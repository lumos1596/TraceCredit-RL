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
