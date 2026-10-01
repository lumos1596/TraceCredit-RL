"""One teacher search intervention followed by an on-policy student continuation."""
from __future__ import annotations

import json
import re
import urllib.request
import uuid

import numpy as np
import torch

from verl import DataProto
from .tree_node import TreeNode


def _query_exposes_privileged_answer(query: str, question: str, gold_answer: str) -> bool:
    """Allow an answer entity already stated in the user's question.

    Comparison questions often name both candidates. Searching for either
    candidate does not reveal the privileged label. A query that introduces a
    gold answer absent from the question does reveal it.
    """
    return (bool(gold_answer) and len(gold_answer) > 3
            and gold_answer.casefold() in query.casefold()
            and gold_answer.casefold() not in question.casefold())


def request_rescue_plan(url: str, *, uid: str, question: str, prefix: str,
                        failed_query: str, failed_observation: str,
                        answers: list[str], timeout: float = 600.0) -> dict:
    request = urllib.request.Request(
        url.rstrip('/') + '/rescue',
        data=json.dumps({
            'uid': uid, 'question': question, 'prefix': prefix,
            'failed_query': failed_query, 'failed_observation': failed_observation,
            'answers': answers,
        }, ensure_ascii=False).encode(),
        headers={'Content-Type': 'application/json'}, method='POST',
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    rescue = payload.get('rescue') or {}
    plan = rescue.get('teacher_plan') or {}
    if plan.get('evidence_sufficient') is not True:
        raise ValueError('teacher evidence is insufficient')
    actions = rescue.get('teacher_actions') or []
    if not 1 <= len(actions) <= 2:
        raise ValueError('teacher returned an invalid action count')
    queries = []
    for item in actions:
        query = ' '.join(str(item.get('query', '')).split())
        if not item.get('valid') or not 3 <= len(query.split()) <= 30 or len(query) > 240:
            raise ValueError('teacher returned an invalid search query')
        if re.search(r'<[^>]+>', query):
            raise ValueError('teacher query contains a tagged action')
        if any(_query_exposes_privileged_answer(query, question, answer) for answer in answers):
            raise ValueError('teacher query exposes a privileged answer')
        queries.append(query)
    handoff = ' '.join(str(rescue.get('handoff', '')).split())
    if not handoff:
        raise ValueError('teacher handoff is empty')
    if any(_query_exposes_privileged_answer(handoff, question, answer) for answer in answers):
        raise ValueError('teacher handoff exposes a privileged answer')
    return {'queries': queries, 'handoff': handoff, 'plan': plan}


def _ids(tokenizer, text):
    return torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)


def _valid(tensor, pad_id):
    return tensor[tensor != pad_id].long().cpu()


def _pad_right(tensor, width, value):
    if tensor.shape[1] == width:
        return tensor
    return torch.nn.functional.pad(tensor, (0, width - tensor.shape[1]), value=value)


def pure_em_rows(output: DataProto, tokenizer):
    """Answer EM on every unfiltered trajectory, ignoring shaped format reward."""
    from verl.utils.reward_score.qa_em_format import em_check
    values = []
    pad = tokenizer.pad_token_id
    for row in range(len(output)):
        response = tokenizer.decode(_valid(output.batch['responses'][row], pad), skip_special_tokens=True)
        truth = output.non_tensor_batch['reward_model'][row]['ground_truth']['target']
        answers = re.findall(r'<answer>(.*?)</answer>', response, flags=re.S)
        answer = answers[-1].strip() if answers else None
        values.append(float(bool(answer is not None and em_check(answer, truth))))
    return values


def rescue_all_wrong(output: DataProto, manager, *, teacher_url: str,
                     group_size: int, threshold: float = 0.8):
    """Replace one all-wrong leaf per prompt with teacher action + student suffix.

    The other five leaves remain the original student samples. Teacher tokens are
    excluded from PPO by info_mask and supervised by teacher_rescue_kd_mask.
    """
    if not teacher_url:
        raise ValueError('teacher_rescue_url is required')
    tokenizer = manager.tokenizer
    pad = tokenizer.pad_token_id
    groups = {}
    for index, uid in enumerate(output.non_tensor_batch['uid']):
        groups.setdefault(uid, []).append(index)
    candidate_rows = []
    for rows in groups.values():
        if len(rows) != group_size:
            raise ValueError('teacher rescue requires complete rollout groups')
        if all(float(output.non_tensor_batch['original_score'][i]) < threshold for i in rows):
            candidate_rows.append(rows[0])
    original_width = output.batch['responses'].shape[1]
    raw_em_count = sum(pure_em_rows(output, tokenizer))
    interventions = []
    failures = []
    for row in candidate_rows:
        try:
            response = tokenizer.decode(_valid(output.batch['responses'][row], pad), skip_special_tokens=True)
            match = re.search(r'<search>(.*?)</search>', response, flags=re.S)
            if match is None:
                raise ValueError('failed trajectory has no search action')
            # The first search decision retains the student's preceding think prefix.
            prefix = response[:match.start()]
            if len(prefix) > 3000:
                raise ValueError('student prefix exceeds safe first-action bound')
            failed_query = match.group(1).strip()
            question = tokenizer.decode(_valid(output.batch['prompts'][row], pad), skip_special_tokens=True)
            reward_model = output.non_tensor_batch['reward_model'][row]
            truth = reward_model.get('ground_truth', {}).get('target', '')
            aliases = truth if isinstance(truth, (list, tuple, np.ndarray)) else [truth]
            aliases = [str(alias) for alias in aliases if str(alias or '')]
            if not aliases:
                raise ValueError('gold answer unavailable')
            observation_match = re.search(
                r'<information>(.*?)</information>', response[match.end():], flags=re.S)
            failed_observation = observation_match.group(0) if observation_match else ''
            plan = request_rescue_plan(
                teacher_url, uid=str(output.non_tensor_batch['uid'][row]),
                question=question, prefix=prefix, failed_query=failed_query,
                failed_observation=failed_observation, answers=aliases)
            prefix_ids = _ids(tokenizer, prefix)
            response_parts = [prefix_ids]
            info_parts = [prefix_ids]
            turn_parts = [torch.ones_like(prefix_ids)]
            kd_parts = [torch.zeros_like(prefix_ids, dtype=torch.bool)]
            for action_index, query in enumerate(plan['queries']):
                if action_index:
                    bridge = ('<think>The preceding evidence leaves another required relation unresolved, '
                              'so I will retrieve that specific relation.</think>\n')
                    bridge_ids = _ids(tokenizer, bridge)
                    response_parts.append(bridge_ids)
                    info_parts.append(torch.full_like(bridge_ids, pad))
                    turn_parts.append(torch.zeros_like(bridge_ids))
                    kd_parts.append(torch.ones_like(bridge_ids, dtype=torch.bool))
                action = f'<search>{query}</search>'
                action_ids = _ids(tokenizer, action)
                observation, done, valid, search = manager.execute_predictions(
                    [action], tokenizer.pad_token, [True])
                if done[0] or not valid[0] or not search[0]:
                    raise ValueError('teacher action did not execute as a search')
                observation_ids = _ids(tokenizer, observation[0])
                response_parts.extend((action_ids, observation_ids))
                info_parts.extend((torch.full_like(action_ids, pad),
                                   torch.full_like(observation_ids, pad)))
                turn_parts.extend((torch.zeros_like(action_ids), torch.zeros_like(observation_ids)))
                kd_parts.extend((torch.ones_like(action_ids, dtype=torch.bool),
                                 torch.zeros_like(observation_ids, dtype=torch.bool)))
            handoff = ('<think>The retrieved evidence fully resolves every required relation. Do not search '
                       'again. Synthesize the exact entity or value requested and answer now. '
                       + plan['handoff'] + '</think>\n')
            handoff_ids = _ids(tokenizer, handoff)
            response_parts.append(handoff_ids)
            info_parts.append(torch.full_like(handoff_ids, pad))
            turn_parts.append(torch.zeros_like(handoff_ids))
            kd_parts.append(torch.ones_like(handoff_ids, dtype=torch.bool))
            seed_response = torch.cat(response_parts)
            if seed_response.numel() >= manager.config.max_start_length:
                raise ValueError('teacher seed exceeds response budget')
            # Only original student prefix is an RL action. The search result
            # and the teacher action have no on-policy likelihood ratio.
            seed_info = torch.cat(info_parts)
            seed_turns = torch.cat(turn_parts)
            seed_kd = torch.cat(kd_parts)
            prompt = output.batch['prompts'][row].long().cpu()
            full = torch.cat((_valid(prompt, pad), seed_response))
            rolling = full[-manager.config.max_start_length:]
            rolling = torch.nn.functional.pad(rolling, (manager.config.max_start_length - rolling.numel(), 0), value=pad)
            attn = (rolling != pad).long()
            positions = (attn.cumsum(-1) - 1).clamp_min(0) * attn
            seed = TreeNode(tree_uid=str(uuid.uuid4()), node_uid=str(uuid.uuid4()),
                            prompts=prompt, input_ids=rolling, attention_mask=attn,
                            position_ids=positions, responses=seed_response,
                            responses_with_info_mask=seed_info, turns_mask=seed_turns,
                            is_root=True, depth=len(plan['queries']),
                            valid_action_stats=len(plan['queries']),
                            valid_search_stats=len(plan['queries']), reward_mode=manager.config.reward_mode,
                            tensor_fn=manager.tensor_fn)
            gen_batch = DataProto.from_dict(tensors={
                'input_ids': rolling.unsqueeze(0), 'attention_mask': attn.unsqueeze(0),
                'position_ids': positions.unsqueeze(0), 'prompts': prompt.unsqueeze(0),
            })
            manager.gen_action_chain(
                gen_batch, [seed], torch.tensor([len(plan['queries'])], dtype=torch.int))
            nodes = seed.get_subtree_nodes()
            if not nodes:
                raise ValueError('student produced no suffix')
            leaf = max(nodes, key=lambda node: node.depth)
            response_ids = _valid(leaf.responses, pad)
            if response_ids.numel() <= seed_response.numel():
                raise ValueError('student suffix is empty')
            info_ids = leaf.responses_with_info_mask[:response_ids.numel()].long().cpu()
            if info_ids.numel() != response_ids.numel():
                raise ValueError('rescue info mask length mismatch')
            kd_mask = torch.zeros(response_ids.numel(), dtype=torch.bool)
            kd_mask[:seed_kd.numel()] = seed_kd
            interventions.append((row, response_ids, info_ids, kd_mask, leaf))
        except Exception as exc:
            failures.append(f'row={row}: {type(exc).__name__}: {exc}')
    if failures:
        print('[teacher rescue] skipped: ' + '; '.join(failures[:5]), flush=True)
    kd = torch.zeros_like(output.batch['responses'], dtype=torch.bool)
    if interventions:
        width = max(original_width, *(ids.numel() for _, ids, _, _, _ in interventions))
        aligned_keys = ('responses', 'token_level_scores', 'branch_advantages',
                        'branch_credit_mask', 'turns_mask')
        for key in aligned_keys:
            if key in output.batch:
                output.batch[key] = _pad_right(output.batch[key], width, pad if key in ('responses', 'turns_mask') else 0)
        kd = _pad_right(kd, width, False)
        info_responses = _pad_right(output.batch['info_mask'][:, -original_width:].clone(), width, 0)
        for row, ids, info, mask, leaf in interventions:
            n = ids.numel()
            output.batch['responses'][row].fill_(pad)
            output.batch['responses'][row, :n] = ids
            info_responses[row].zero_()
            info_responses[row, :n] = (info != pad).long()
            kd[row, :n] = mask
            for key in ('token_level_scores', 'branch_advantages', 'branch_credit_mask', 'turns_mask'):
                if key in output.batch:
                    output.batch[key][row].zero_()
            if 'turns_mask' in output.batch:
                output.batch['turns_mask'][row, :n] = leaf.turns_mask[:n]
            for key in output.batch.keys():
                if key.startswith('branch_credit_') or key.startswith('self_opd_'):
                    output.batch[key][row].zero_()
            output.non_tensor_batch['node_uid'][row] = leaf.node_uid
            output.non_tensor_batch['tree_uid'][row] = leaf.tree_uid
            if 'turns_stats' in output.meta_info:
                output.meta_info['turns_stats'][row] = leaf.depth
        output.batch['input_ids'] = torch.cat((output.batch['prompts'], output.batch['responses']), dim=1)
        prompt_attn = manager.tensor_fn.create_attention_mask(output.batch['prompts'])
        response_attn = manager.tensor_fn.create_attention_mask(output.batch['responses'])
        output.batch['attention_mask'] = torch.cat((prompt_attn, response_attn), dim=1)
        output.batch['info_mask'] = torch.cat((prompt_attn, info_responses), dim=1)
        output.batch['position_ids'] = manager.tensor_fn.create_position_ids(output.batch['attention_mask'])
        scores, _ = manager.reward_fn(output)
        for row, ids, _, _, _ in interventions:
            score = float(scores[row])
            output.non_tensor_batch['original_score'][row] = score
            output.batch['token_level_scores'][row, ids.numel() - 1] = score
    output.batch['teacher_rescue_kd_mask'] = kd
    return {
        'eligible': len(candidate_rows), 'intervened': len(interventions),
        'raw_em_count': raw_em_count,
        'failed': len(failures),
        'successes': sum(pure_em_rows(output, tokenizer)[r] for r, *_ in interventions),
        'promoted_groups': sum(float(output.non_tensor_batch['original_score'][r]) >= threshold
                               for r, *_ in interventions),
        'rescued_uids': {output.non_tensor_batch['uid'][r] for r, *_ in interventions},
    }


def request_prefix_rescue(url: str, *, uid: str, question: str, answers: list[str],
                          hops: int = 2, timeout: float = 300.0) -> list[dict]:
    """Call the gold-free scout service; return [{think, query}, ...]."""
    request = urllib.request.Request(
        url.rstrip('/') + '/prefix_rescue',
        data=json.dumps({'uid': uid, 'question': question, 'hops': hops},
                        ensure_ascii=False).encode(),
        headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    rescue = payload.get('rescue') or {}
    if rescue.get('ok') is not True:
        raise ValueError(str(rescue.get('error', 'prefix rescue failed')))
    plan_hops = rescue.get('hops') or []
    if not 1 <= len(plan_hops) <= hops:
        raise ValueError('prefix rescue returned an invalid hop count')
    result = []
    for item in plan_hops:
        think = ' '.join(str(item.get('think', '')).split())
        query = ' '.join(str(item.get('query', '')).split())
        if not think or not 3 <= len(query.split()) <= 30 or len(query) > 240:
            raise ValueError('prefix rescue returned an invalid search query')
        if re.search(r'<[^>]+>', query) or re.search(r'<[^>]+>', think):
            raise ValueError('prefix rescue query contains a tagged action')
        if any(_query_exposes_privileged_answer(query, question, answer) for answer in answers):
            raise ValueError('prefix rescue query exposes a privileged answer')
        result.append({'think': think, 'query': query})
    return result


def _rolling_state(manager, prompt, response):
    """Left-padded (input_ids, attn, pos) for prompt + cumulative response."""
    pad = manager.tokenizer.pad_token_id
    full = torch.cat((_valid(prompt, pad), response))
    width = manager.config.max_start_length
    rolling = full[-width:]
    rolling = torch.nn.functional.pad(
        rolling, (width - rolling.numel(), 0), value=pad)
    attn = (rolling != pad).long()
    positions = (attn.cumsum(-1) - 1).clamp_min(0) * attn
    return rolling, attn, positions


def _build_seeded_prefix_tree(manager, *, prompt, hop_pieces):
    """Two-level seeded tree: root at teacher hop 1, child at teacher hop 2.

    ``hop_pieces`` is one entry per teacher hop, each with cumulative-safe
    response/info/turns/kd pieces already concatenated.  The tree root
    contains hop 1 only, so the ordinary expansion round is allowed to fork
    *before* teacher action 2 (the root itself is an expansion candidate);
    the node carrying the complete 2-hop prefix is one of its children, and
    the initial model chain continues from that node.
    """
    if len(hop_pieces) != 2:
        raise ValueError('forkable prefix tree expects exactly 2 teacher hops')
    tree_uid = str(uuid.uuid4())
    h1, h2 = hop_pieces
    r1 = h1['response']
    r12 = torch.cat((r1, h2['response']))
    i1, i12 = h1['info'], torch.cat((h1['info'], h2['info']))
    t1, t12 = h1['turns'], torch.cat((h1['turns'], h2['turns']))

    rolling1, attn1, pos1 = _rolling_state(manager, prompt, r1)
    root = TreeNode(
        tree_uid=tree_uid, node_uid=str(uuid.uuid4()), prompts=prompt,
        input_ids=rolling1, attention_mask=attn1, position_ids=pos1,
        responses=r1, responses_with_info_mask=i1, turns_mask=t1,
        is_root=True, depth=1, valid_action_stats=1, valid_search_stats=1,
        reward_mode=manager.config.reward_mode,
        tensor_fn=manager.tensor_fn)

    rolling2, attn2, pos2 = _rolling_state(manager, prompt, r12)
    node2 = TreeNode(
        tree_uid=tree_uid, node_uid=str(uuid.uuid4()), prompts=prompt,
        input_ids=rolling2, attention_mask=attn2, position_ids=pos2,
        responses=r12, responses_with_info_mask=i12, turns_mask=t12,
        parent_node=root, is_root=False, depth=2,
        valid_action_stats=2, valid_search_stats=2,
        reward_mode=manager.config.reward_mode,
        tensor_fn=manager.tensor_fn)
    root.add_child(node2)
    meta = {
        'hop1_len': r1.numel(), 'hop2_len': r12.numel(),
        'kd_hop1': h1['kd'], 'kd_hop2': h2['kd'],
        'hop2_uid': node2.node_uid,
    }
    return root, node2, meta


def _leaf_kd_mask(leaf, meta, pad_id):
    """KD mask aligned to a leaf's valid response tokens.

    Hop 1 teacher tokens exist on every leaf; hop 2 teacher tokens exist only
    on leaves whose path passes through the full-prefix node.  Observation
    tokens are already zero in the stored per-hop KD patterns.
    """
    valid_len = int((leaf.responses != pad_id).sum().item())
    kd = torch.zeros(valid_len, dtype=torch.bool)
    l1 = meta['hop1_len']
    kd[:l1] = meta['kd_hop1'][:l1]
    node = leaf.parent_node
    has_hop2 = False
    while node is not None:
        if node.node_uid == meta['hop2_uid']:
            has_hop2 = True
            break
        node = node.parent_node
    if has_hop2:
        l2 = meta['hop2_len']
        kd[l1:l2] = meta['kd_hop2'][:l2 - l1]
    return kd


def _prefix_tree_rollout(manager, seeded_trees):
    """Run the IDENTICAL tree sampling on forked teacher-prefix trees.

    ``seeded_trees`` is a list of (root, node2, meta) from
    ``_build_seeded_prefix_tree``.  The initial chain continues from the
    full-prefix node2; then ``l`` iterations expand ``n`` nodes per tree from
    the root's own candidate set, so continuations may fork at hop 1 (before
    teacher action 2) exactly as in a normal tree rollout; finally ``k``
    leaves are sampled per tree -> m*k leaves per group.
    """
    cfg = manager.config
    node2_list = [item[1] for item in seeded_trees]
    gen_batch = DataProto.from_dict(tensors={
        'input_ids': torch.stack([node.input_ids for node in node2_list]),
        'attention_mask': torch.stack([node.attention_mask for node in node2_list]),
        'position_ids': torch.stack([node.position_ids for node in node2_list]),
        'prompts': torch.stack([node.prompts for node in node2_list]),
    })
    turns_stats = torch.tensor([node.depth for node in node2_list], dtype=torch.int)
    manager.gen_action_chain(gen_batch, node2_list, turns_stats)

    roots = [item[0] for item in seeded_trees]
    picked_depths = []
    for _ in range(cfg.l):
        expansion_node_list = []
        for root in roots:
            picked = root.get_expand_node(
                cfg.n,
                mode=cfg.expand_mode,
                uncertainty_weight=cfg.expand_uncertainty_weight,
                search_weight=cfg.expand_search_weight,
                child_count_weight=cfg.expand_child_count_weight,
                depth_weight=cfg.expand_depth_weight,
                outcome_prior_root=cfg.expand_outcome_prior_root,
                outcome_prior_pre_search=cfg.expand_outcome_prior_pre_search,
                outcome_prior_after_search_depth1=cfg.expand_outcome_prior_after_search_depth1,
                outcome_prior_after_search_depth2=cfg.expand_outcome_prior_after_search_depth2,
                outcome_prior_after_search_depth3plus=cfg.expand_outcome_prior_after_search_depth3plus,
                outcome_prior_concentration_power=cfg.expand_outcome_prior_concentration_power,
            )
            picked_depths.append([node.depth for node in picked])
            expansion_node_list.extend(picked)
        if not expansion_node_list:
            break
        prompts = torch.stack([node.prompts for node in expansion_node_list])
        input_ids = manager.tensor_fn.pad_and_stack(
            [node.input_ids for node in expansion_node_list])
        attention_masks = manager.tensor_fn.create_attention_mask(input_ids)
        position_ids = manager.tensor_fn.create_position_ids(attention_masks)
        expand_batch = DataProto.from_dict(tensors={
            'input_ids': input_ids, 'attention_mask': attention_masks,
            'position_ids': position_ids, 'prompts': prompts})
        expand_turns = torch.tensor(
            [node.depth for node in expansion_node_list], dtype=torch.int)
        manager.gen_action_chain(expand_batch, expansion_node_list, expand_turns)

    leaves = []
    for root in roots:
        root.check_all_nodes_child()
        leaf_count = sum(1 for node in root.get_subtree_nodes() if node.is_leaf)
        if leaf_count < cfg.k:
            topology = '; '.join(
                f"(d{node.depth},leaf={int(node.is_leaf)},kids={len(node.child_node)},"
                f"va={node.valid_action_stats},vs={node.valid_search_stats})"
                for node in [root] + root.get_subtree_nodes()
            )
            raise AssertionError(
                f'seeded tree leaf_count={leaf_count}<k={cfg.k} '
                f'max_turns={cfg.max_turns} picked_depths={picked_depths} '
                f'topology=[{topology}]')
        leaves.extend(root.sample_leaf(cfg.k))
    return leaves


def rescue_all_wrong_prefix(output: DataProto, manager, *, teacher_url: str,
                            group_size: int, hops: int = 2,
                            threshold: float = 0.8):
    """Gold-free teacher prefix + the NORMAL tree rollout on top of it.

    Every all-wrong group gets ``m`` forked trees: the tree root is seeded with
    teacher hop 1 (shared think+search+observation); the full 2-hop teacher
    prefix is one child of that root, the initial model chain continues from
    it, and the ordinary expansion round (same ts_n/ts_l) may also fork
    directly from the root -- i.e. before teacher action 2 -- so the student
    can replace hop 2 with its own search.  Each tree samples ``k`` leaves,
    giving exactly m*k leaves as in a normal rollout.  The group replaces its
    original all-wrong rows; afterwards the normal DAPO selector keeps it only
    if at least one leaf is correct (mixed/all-correct).  All-wrong rescued
    groups are discarded like any other invalid group -- they are NOT
    force-included.  Teacher think/search tokens carry the KD objective and are
    excluded from PPO by info_mask; branch credit is computed on the real
    tree, and the root (teacher hop 1) receives no branch advantage.
    """
    if not teacher_url:
        raise ValueError('teacher_rescue_url is required')
    tokenizer = manager.tokenizer
    pad = tokenizer.pad_token_id
    cfg = manager.config
    if group_size != cfg.m * cfg.k:
        raise ValueError(
            f'prefix rescue group_size={group_size} != tree m*k={cfg.m}*{cfg.k}')
    groups = {}
    for index, uid in enumerate(output.non_tensor_batch['uid']):
        groups.setdefault(uid, []).append(index)
    candidate_rows = []
    for rows in groups.values():
        if len(rows) != group_size:
            raise ValueError('teacher rescue requires complete rollout groups')
        if all(float(output.non_tensor_batch['original_score'][i]) < threshold
               for i in rows):
            candidate_rows.append(rows[0])
    original_width = output.batch['responses'].shape[1]
    raw_em_count = sum(pure_em_rows(output, tokenizer))
    question_re = re.compile(r'Question:\s*(.*?)\s*<\|im_end\|>', flags=re.S)

    # Each entry: (rows, roots, leaves, seed_response, seed_kd)
    interventions = []
    failures = []
    for row in candidate_rows:
        try:
            question_text = tokenizer.decode(
                _valid(output.batch['prompts'][row], pad), skip_special_tokens=True)
            question_match = question_re.search(question_text)
            question = (question_match.group(1).strip() if question_match
                        else question_text.strip())
            if not question:
                raise ValueError('question unavailable')
            reward_model = output.non_tensor_batch['reward_model'][row]
            truth = reward_model.get('ground_truth', {}).get('target', '')
            aliases = truth if isinstance(truth, (list, tuple, np.ndarray)) else [truth]
            aliases = [str(alias) for alias in aliases if str(alias or '')]
            if not aliases:
                raise ValueError('gold answer unavailable')
            plan = request_prefix_rescue(
                teacher_url, uid=str(output.non_tensor_batch['uid'][row]),
                question=question, answers=aliases, hops=hops)

            # One hop piece per teacher action; pieces are per-hop (not
            # cumulative) so the seeded tree can share hop 1 and fork before
            # hop 2.
            hop_pieces = []
            for item in plan:
                think_ids = _ids(tokenizer, f"<think>{item['think']}</think>\n")
                action_ids = _ids(tokenizer, f"<search>{item['query']}</search>")
                observation, done, valid, search = manager.execute_predictions(
                    [f"<search>{item['query']}</search>"], tokenizer.pad_token, [True])
                if done[0] or not valid[0] or not search[0]:
                    raise ValueError('teacher action did not execute as a search')
                observation_ids = _ids(tokenizer, observation[0])
                hop_pieces.append({
                    'response': torch.cat((think_ids, action_ids, observation_ids)),
                    'info': torch.cat((
                        torch.full_like(think_ids, pad),
                        torch.full_like(action_ids, pad),
                        torch.full_like(observation_ids, pad))),
                    'turns': torch.zeros(
                        think_ids.numel() + action_ids.numel() + observation_ids.numel(),
                        dtype=torch.long),
                    'kd': torch.cat((
                        torch.ones_like(think_ids, dtype=torch.bool),
                        torch.ones_like(action_ids, dtype=torch.bool),
                        torch.zeros_like(observation_ids, dtype=torch.bool))),
                })
            if sum(p['response'].numel() for p in hop_pieces) >= manager.config.max_start_length:
                raise ValueError('teacher seed exceeds response budget')
            prompt = output.batch['prompts'][row].long().cpu()

            seeded_trees = [
                _build_seeded_prefix_tree(manager, prompt=prompt, hop_pieces=hop_pieces)
                for _ in range(cfg.m)
            ]
            roots = [item[0] for item in seeded_trees]
            metas = [item[2] for item in seeded_trees]
            leaves = _prefix_tree_rollout(manager, seeded_trees)
            if len(leaves) != group_size:
                raise ValueError(
                    f'prefix tree produced {len(leaves)} leaves, expected {group_size}')
            # Every leaf shares teacher hop 1 (the tree root); leaves forked
            # before hop 2 legitimately diverge afterwards.
            roots_by_tree = {root.tree_uid: root for root in roots}
            for leaf in leaves:
                root = roots_by_tree[leaf.tree_uid]
                hop1_len = int((root.responses != pad).sum().item())
                leaf_valid = leaf.responses[leaf.responses != pad]
                root_valid = root.responses[root.responses != pad]
                if not leaf_valid[:hop1_len].equal(root_valid):
                    raise ValueError('leaf path does not share the teacher hop-1 seed')
            leaf_metas = [meta for meta in metas for _ in range(cfg.k)]
            interventions.append(
                (groups[output.non_tensor_batch['uid'][row]], roots, leaves,
                 leaf_metas))
        except Exception as exc:
            failures.append(f'row={row}: {type(exc).__name__}: {exc}')
    if failures:
        print('[prefix rescue] skipped: ' + '; '.join(failures[:5]), flush=True)

    kd = torch.zeros_like(output.batch['responses'], dtype=torch.bool)
    if interventions:
        width = max(original_width,
                    *(leaf.responses.numel()
                      for _, _, leaves, _ in interventions for leaf in leaves))
        aligned_keys = ('responses', 'token_level_scores', 'branch_advantages',
                        'branch_credit_mask', 'turns_mask')
        for key in aligned_keys:
            if key in output.batch:
                output.batch[key] = _pad_right(
                    output.batch[key], width,
                    pad if key in ('responses', 'turns_mask') else 0)
        kd = _pad_right(kd, width, False)
        info_responses = _pad_right(
            output.batch['info_mask'][:, -original_width:].clone(), width, 0)

        # Pass 1: replace the all-wrong rows with the prefix-tree leaves.
        for rows, _roots, leaves, leaf_metas in interventions:
            for row, leaf, meta in zip(rows, leaves, leaf_metas):
                ids = leaf.responses
                n = ids.numel()
                info_ids = leaf.responses_with_info_mask[:n].long().cpu()
                if info_ids.numel() != n:
                    raise ValueError('rescue info mask length mismatch')
                output.batch['responses'][row].fill_(pad)
                output.batch['responses'][row, :n] = ids
                info_responses[row].zero_()
                info_responses[row, :n] = (info_ids != pad).long()
                leaf_kd = _leaf_kd_mask(leaf, meta, pad)
                kd[row, :leaf_kd.numel()] = leaf_kd
                for key in ('token_level_scores', 'branch_advantages',
                            'branch_credit_mask', 'turns_mask'):
                    if key in output.batch:
                        output.batch[key][row].zero_()
                if 'turns_mask' in output.batch:
                    output.batch['turns_mask'][row, :n] = leaf.turns_mask[:n]
                for key in output.batch.keys():
                    if key.startswith('branch_credit_') or key.startswith('self_opd_'):
                        output.batch[key][row].zero_()
                output.non_tensor_batch['node_uid'][row] = leaf.node_uid
                output.non_tensor_batch['tree_uid'][row] = leaf.tree_uid
                if 'turns_stats' in output.meta_info:
                    output.meta_info['turns_stats'][row] = leaf.depth

        output.batch['input_ids'] = torch.cat(
            (output.batch['prompts'], output.batch['responses']), dim=1)
        prompt_attn = manager.tensor_fn.create_attention_mask(output.batch['prompts'])
        response_attn = manager.tensor_fn.create_attention_mask(output.batch['responses'])
        output.batch['attention_mask'] = torch.cat((prompt_attn, response_attn), dim=1)
        output.batch['info_mask'] = torch.cat((prompt_attn, info_responses), dim=1)
        output.batch['position_ids'] = manager.tensor_fn.create_position_ids(
            output.batch['attention_mask'])

        # Reward the rebuilt batch, then score the real trees and write
        # per-leaf token scores / branch credit exactly like a normal rollout.
        scores, _ = manager.reward_fn(output)
        rescued_groups = 0
        for rows, roots, leaves, _leaf_metas in interventions:
            for row, leaf in zip(rows, leaves):
                leaf.set_leaf_original_score(float(scores[row]))
            for root in roots:
                root.calculate_final_score_from_root()
                if cfg.enable_branch_credit:
                    root.calculate_branch_credit_from_root(
                        normalization=cfg.branch_credit_normalization,
                        clip=cfg.branch_credit_clip,
                        value_mode=cfg.branch_credit_value_mode,
                        correctness_threshold=cfg.branch_credit_correctness_threshold,
                    )
            group_rescued = False
            for row, leaf in zip(rows, leaves):
                score = float(scores[row])
                output.non_tensor_batch['original_score'][row] = score
                group_rescued = group_rescued or score >= threshold
                token_scores = leaf.get_token_level_score_from_leaf()
                n = token_scores.numel()
                output.batch['token_level_scores'][row, :n] = token_scores
                if cfg.enable_branch_credit:
                    branch_adv, branch_mask = leaf.get_branch_credit_from_leaf()
                    output.batch['branch_advantages'][row, :n] = branch_adv
                    output.batch['branch_credit_mask'][row, :n] = branch_mask
            rescued_groups += int(group_rescued)
    else:
        rescued_groups = 0
    output.batch['teacher_rescue_kd_mask'] = kd
    # rescued_uids is intentionally empty: groups without a correct leaf stay
    # all-wrong and are dropped by the ordinary DAPO effective-row selection.
    return {
        'eligible': len(candidate_rows), 'intervened': len(interventions),
        'raw_em_count': raw_em_count,
        'failed': len(failures),
        'successes': rescued_groups,
        'promoted_groups': 0,
        'rescued_uids': set(),
    }
