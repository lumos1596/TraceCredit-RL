"""TraceCredit-guided reasoning-and-query on-policy distillation.

The privileged teacher is deliberately adapted to this tree-search setup:

* only the best positive-credit query under a parent is distilled;
* the target query is excluded from hindsight, preventing a copy shortcut;
* continuous sibling values and the target's later retrieved evidence replace
  a binary success/failure label;
* the original prompt is anchored while recent state is tail-truncated.

The builder stays independent from the actor and emits fixed-shape tensors so
they survive DataProto concatenation, filtering, and reordering.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

import torch

_NODE_SKILL_FIELDS = ("failure_type", "missing_relation", "next_operation", "stop_condition")
_OPID_SKILL_FIELDS = ("episode_summary", "episode_skill", "step_skills")


def _request_node_skill(url: str, prompt: str, timeout: float = 120.0) -> Optional[dict]:
    body = json.dumps({"prompt": prompt}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/generate", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    skill = payload.get("skill")
    return skill if isinstance(skill, dict) else None


def _validate_node_skill(skill: dict, *, gold_answer: str, queries: Sequence[str], visible_state: str = "") -> bool:
    if set(skill) != set(_NODE_SKILL_FIELDS):
        return False
    values = [str(skill.get(field, "")).strip() for field in _NODE_SKILL_FIELDS]
    combined = "\n".join(values)
    if any(not value for value in values) or not 14 <= len(combined.split()) <= 120:
        return False
    if re.search(r"\b(?:19|20)\d{2}\b|\b\d{1,4}\b", combined):
        return False
    if re.search(r"\b(?:reward|score|gold|branch|hindsight|better action|correct answer)\b", combined, re.I):
        return False
    normalized = _normalized_words(combined)
    answer = _normalized_words(gold_answer)
    if answer and answer in normalized:
        return False
    visible_normalized = _normalized_words(visible_state)
    for query in queries:
        query_norm = _normalized_words(query)
        if (
            len(query_norm.split()) >= 3
            and query_norm in normalized
            and not set(query_norm.split()).issubset(set(visible_normalized.split()))
        ):
            return False
    return True


def _node_skill_prompt(state: str, good_action: str, bad_action: str, evidence: str) -> str:
    return f"""Create one node-level procedural skill for a search agent. Return ONLY JSON.
Schema: {{"failure_type":"short category","missing_relation":"abstract unresolved relation or evidence gap","next_operation":"one executable verification/search operation","stop_condition":"what must be verified before advancing"}}
Do not reveal or infer the final answer. You MAY name entities, works, places, and relations that already appear in the Current visible state, because the agent already knows them. Do not copy either historical action verbatim. Do not quote a retrieved sentence, date, number, newly revealed entity, or answer-bearing fact that appears only in the private action or private evidence sections. Do not mention rewards, scores, branches, gold evidence, hindsight, or which action was better. Produce a concrete verification operation while using only information available in the Current visible state.
Current visible state:
{state[-9000:]}
Historical unsuccessful action:
{bad_action}
Historical more useful action (private; do not copy):
{good_action}
Retrieved evidence (private; do not copy):
{evidence[:1800]}
"""


def _opid_analyzer_prompt(task: str, outcome: bool, candidate_steps: Sequence[int],
                          formatted_steps: str, max_skill_count: int) -> str:
    """Paper Appendix Figure 9 analyzer prompt, adapted only to JSON formatting."""
    return f"""Analyze the following agent episode and return ONLY valid JSON.

You need to complete all three fields:
1. Write a concise episode_summary.
2. Write one episode_skill that extracts the successful trajectory into workflow: the core decision rule and action ordering that made this trajectory work. / Write one episode_skill that extracts the failed trajectory into avoidance rules: the core mistake and warning signs that agent should avoid.
3. Provide concise, action-oriented decision guidance for at most {max_skill_count} critical step(s) from the candidate set as entries in step_skills; use the full episode to infer the guidance, but phrase each skill as advice the policy can act on at that step.

Important constraints:
- Step indexing is 0-based: step 0 is the first step of the trajectory.
- Use the task description together with the episode context to judge progress and mistakes.
- Use the full episode context to identify what each critical step should have done better.
- Each step_skills value should be one short imperative sentence for the policy at that step.
- Write step_skills as policy-facing guidance, not as retrospective explanation of the trajectory.
- Return only these top-level fields: episode_summary, episode_skill, step_skills.
- The chosen steps are exactly the keys present in step_skills.

Return format:
{{"episode_summary":"string","episode_skill":"string","step_skills":{{"0":"skill for step 0","2":"skill for step 2"}}}}

Episode context:
- Task description: {task}
- episode_success: {str(bool(outcome)).lower()}
- Candidate step indices: {json.dumps(list(candidate_steps))}
- Interaction trajectory:
{formatted_steps}
"""


def _validate_opid_skill(skill: dict, candidate_steps: Sequence[int]) -> bool:
    if not isinstance(skill, dict) or set(skill) != set(_OPID_SKILL_FIELDS):
        return False
    if not all(isinstance(skill.get(field), str) and skill[field].strip()
               for field in ("episode_summary", "episode_skill")):
        return False
    step_skills = skill.get("step_skills")
    if not isinstance(step_skills, dict):
        return False
    allowed = {str(index) for index in candidate_steps}
    return all(str(key) in allowed and isinstance(value, str) and value.strip()
               for key, value in step_skills.items())


@dataclass(frozen=True)
class QuerySpan:
    token_ids: Tuple[int, ...]
    response_positions: Tuple[int, ...]


@dataclass(frozen=True)
class PolicySpan:
    """Policy tokens on one tree edge, with reasoning/query supervision masks."""

    action_ids: Tuple[int, ...]
    selected_offsets: Tuple[int, ...]
    student_positions: Tuple[int, ...]
    think_mask: Tuple[bool, ...]
    query_mask: Tuple[bool, ...]


@dataclass(frozen=True)
class _TeacherEvent:
    teacher_ids: Tuple[int, ...]
    teacher_positions: Tuple[int, ...]
    student_positions: Tuple[int, ...]
    think_mask: Tuple[bool, ...]
    query_mask: Tuple[bool, ...]
    weight: float
    raw_advantage: float
    value_gap: float
    target_value: float
    sibling_count: int
    context_truncated: bool
    prompt_retained_fraction: float


def _encode(tokenizer, text: str) -> List[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _find_subsequence(values: Sequence[int], needle: Sequence[int], start: int = 0) -> int:
    if not needle:
        return -1
    stop = len(values) - len(needle) + 1
    for index in range(start, max(start, stop)):
        if list(values[index:index + len(needle)]) == list(needle):
            return index
    return -1


def _valid_length(node) -> int:
    if node.responses is None:
        return 0
    if node.tensor_fn is not None:
        return int(node.tensor_fn.create_attention_mask(node.responses).sum().item())
    return int((node.responses != 0).sum().item())


def _valid_tokens(tensor, attention_mask, pad_token_id: int) -> List[int]:
    if tensor is None:
        return []
    values = tensor.detach().cpu().tolist()
    if attention_mask is not None and len(attention_mask) == len(values):
        mask = attention_mask.detach().cpu().tolist()
        return [int(token) for token, keep in zip(values, mask) if int(keep) != 0]
    return [int(token) for token in values if int(token) != pad_token_id]


def extract_edge_query_span(node, tokenizer) -> Optional[QuerySpan]:
    """Return policy-generated query content introduced by ``parent -> node``."""
    if node.parent_node is None or node.responses is None:
        return None
    start = 0 if node.parent_node.is_root else _valid_length(node.parent_node)
    end = _valid_length(node)
    if end <= start:
        return None
    edge_tokens = node.responses[start:end].detach().cpu().tolist()
    open_ids = _encode(tokenizer, "<search>")
    close_ids = _encode(tokenizer, "</search>")
    open_at = _find_subsequence(edge_tokens, open_ids)
    if open_at < 0:
        return None
    query_start = open_at + len(open_ids)
    close_at = _find_subsequence(edge_tokens, close_ids, query_start)
    if close_at <= query_start:
        return None
    positions = tuple(range(start + query_start, start + close_at))
    token_ids = tuple(int(x) for x in edge_tokens[query_start:close_at])
    if not token_ids:
        return None
    info = node.responses_with_info_mask
    if info is not None:
        info_values = info.detach().cpu()
        if any(int(info_values[pos].item()) == 0 for pos in positions):
            return None
    return QuerySpan(token_ids=token_ids, response_positions=positions)


def extract_edge_policy_span(node, tokenizer, max_action_tokens: int) -> Optional[PolicySpan]:
    """Extract local ``think`` content and query content from one on-policy edge.

    Control tags and environment-provided information are retained in the
    teacher-forcing context but never receive OPD loss.
    """
    if node.parent_node is None or node.responses is None:
        return None
    edge_start = 0 if node.parent_node.is_root else _valid_length(node.parent_node)
    edge_end = _valid_length(node)
    if edge_end <= edge_start:
        return None
    edge = node.responses[edge_start:edge_end].detach().cpu().tolist()
    search_open = _encode(tokenizer, "<search>")
    search_close = _encode(tokenizer, "</search>")
    search_at = _find_subsequence(edge, search_open)
    if search_at < 0:
        return None
    query_start = search_at + len(search_open)
    query_end = _find_subsequence(edge, search_close, query_start)
    if query_end <= query_start:
        return None

    think_open = _encode(tokenizer, "<think>")
    think_close = _encode(tokenizer, "</think>")
    think_at = _find_subsequence(edge, think_open)
    think_start = think_at + len(think_open) if 0 <= think_at < search_at else 0
    think_end = _find_subsequence(edge, think_close, think_start)
    if think_end < think_start or think_end > search_at:
        # Legacy trajectories may omit explicit think tags; treat policy text
        # before <search> as the local planning span.
        think_start, think_end = 0, search_at

    action_start = think_at if 0 <= think_at < search_at else 0
    action_end = query_end
    action_ids = tuple(int(x) for x in edge[action_start:action_end])
    selected = list(range(think_start, think_end)) + list(range(query_start, query_end))
    if not selected or len(selected) > max_action_tokens:
        return None
    info = node.responses_with_info_mask
    absolute = [edge_start + position for position in selected]
    if info is not None:
        info_values = info.detach().cpu()
        if any(int(info_values[position].item()) == 0 for position in absolute):
            return None
    think_count = max(0, think_end - think_start)
    return PolicySpan(
        action_ids=action_ids,
        selected_offsets=tuple(position - action_start for position in selected),
        student_positions=tuple(absolute),
        think_mask=tuple([True] * think_count + [False] * (len(selected) - think_count)),
        query_mask=tuple([False] * think_count + [True] * (len(selected) - think_count)),
    )


def _extract_edge_evidence(node, tokenizer, max_tokens: int) -> str:
    """Extract a compact observation revealed after the target query."""
    if max_tokens <= 0 or node.parent_node is None or node.responses is None:
        return ""
    start = 0 if node.parent_node.is_root else _valid_length(node.parent_node)
    end = _valid_length(node)
    edge_tokens = node.responses[start:end].detach().cpu().tolist()
    open_ids = _encode(tokenizer, "<information>")
    close_ids = _encode(tokenizer, "</information>")
    open_at = _find_subsequence(edge_tokens, open_ids)
    if open_at < 0:
        return ""
    evidence_start = open_at + len(open_ids)
    close_at = _find_subsequence(edge_tokens, close_ids, evidence_start)
    if close_at < 0:
        close_at = len(edge_tokens)
    evidence_ids = edge_tokens[evidence_start:close_at][:max_tokens]
    return tokenizer.decode(evidence_ids, skip_special_tokens=False).strip()


def _redact_answer(text: str, gold_answer: Optional[str]) -> str:
    """Remove literal answer aliases from privileged text before tokenization."""
    answer = " ".join(str(gold_answer or "").split())
    if not answer:
        return text
    pattern = re.compile(r"\s+".join(re.escape(part) for part in answer.split()), re.IGNORECASE)
    return pattern.sub("[REDACTED]", text)


def _normalized_words(text: str) -> str:
    return " ".join(re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE))


def _path_from_root(leaf) -> List:
    path = []
    node = leaf
    while node is not None and not node.is_root:
        path.append(node)
        node = node.parent_node
    path.reverse()
    return path


def _anchored_prefix(parent, tokenizer, budget: int) -> Tuple[List[int], bool, float]:
    """Keep the original prompt and the most recent state within ``budget``."""
    if budget <= 0:
        return [], True, 0.0
    pad_id = int(tokenizer.pad_token_id)
    prompt = _valid_tokens(parent.prompts, None, pad_id)
    state = _valid_tokens(parent.input_ids, parent.attention_mask, pad_id)

    # Root/early rolling states normally start with the prompt.  Remove that
    # exact duplicate before composing prompt + recent state.
    recent = state[len(prompt):] if prompt and state[:len(prompt)] == prompt else state
    total_length = len(prompt) + len(recent)
    truncated = total_length > budget

    if len(prompt) >= budget:
        # Chat prompts end in the user question; retaining the tail anchors the
        # task even when an unusually long system message consumes the budget.
        kept_prompt = prompt[-budget:]
        return kept_prompt, truncated, len(kept_prompt) / max(1, len(prompt))

    recent_budget = budget - len(prompt)
    kept_recent = recent[-recent_budget:] if recent_budget else []
    return prompt + kept_recent, truncated, 1.0 if prompt else 0.0


def _query_siblings(parent, tokenizer, max_query_tokens: int, max_action_tokens: int):
    siblings = []
    for sibling in parent.child_node:
        span = extract_edge_query_span(sibling, tokenizer)
        if span is None or len(span.token_ids) > max_query_tokens:
            continue
        policy_span = extract_edge_policy_span(sibling, tokenizer, max_action_tokens)
        if policy_span is None:
            continue
        siblings.append((sibling, span, policy_span))
    return siblings


def _render_answer_cheat(gold_answer: str) -> str:
    """Render an answer-conditioned backward-query instruction.

    Merely inserting the final answer conflicts with the base task instruction:
    a capable model should answer immediately once the answer is known.  Make
    the privileged task explicit instead: infer the missing bridge fact from
    the current state and emit the search query that would recover it.
    """
    answer = " ".join(str(gold_answer).split())
    return (
        "\n<information>(reference) "
        f'The correct final answer is "{answer}"'
        "\n</information>\n"
        "The final answer above is privileged hindsight, not permission to "
        "answer now. Work backward from it and the original question. Identify "
        "the missing bridge fact at the current search state, then produce the "
        "single next search-engine query most likely to retrieve that fact. "
        "Output only that query inside the already-open search action.\n<search>"
    )


def _dense_teacher_event(
    node,
    tokenizer,
    *,
    max_teacher_length: int,
    max_query_tokens: int,
    max_action_tokens: int,
    gold_answer: Optional[str],
) -> Optional[_TeacherEvent]:
    """Answer-conditioned event for a single query, without contrast gating."""
    parent = node.parent_node
    if parent is None or not gold_answer:
        return None
    query_span = extract_edge_query_span(node, tokenizer)
    span = extract_edge_policy_span(node, tokenizer, max_action_tokens)
    if query_span is None or len(query_span.token_ids) > max_query_tokens or span is None:
        return None
    suffix = _encode(tokenizer, _render_answer_cheat(gold_answer))
    target_ids = list(span.action_ids)
    required = len(suffix) + len(target_ids)
    if required > max_teacher_length:
        return None
    base, context_truncated, prompt_fraction = _anchored_prefix(
        parent, tokenizer, max_teacher_length - required
    )
    teacher_ids = base + suffix + target_ids
    target_start = len(base) + len(suffix)
    teacher_positions = tuple(target_start + offset - 1 for offset in span.selected_offsets)
    return _TeacherEvent(
        teacher_ids=tuple(teacher_ids),
        teacher_positions=teacher_positions,
        student_positions=span.student_positions,
        think_mask=span.think_mask,
        query_mask=span.query_mask,
        weight=1.0,
        raw_advantage=0.0,
        value_gap=0.0,
        target_value=0.0,
        sibling_count=1,
        context_truncated=bool(context_truncated),
        prompt_retained_fraction=float(prompt_fraction),
    )



def _paper_teacher_events(
    leaf,
    tokenizer,
    *,
    max_teacher_length: int,
    max_query_tokens: int,
    max_action_tokens: int,
    min_value_gap: float,
    analyzer_url: str,
    max_skill_count: int,
    global_failure_only: bool = False,
) -> List[_TeacherEvent]:
    """Build Appendix-B critical-first events from one completed trajectory."""
    if not analyzer_url:
        return []
    path = _path_from_root(leaf)
    if not path:
        return []
    episode_success = float(getattr(leaf, "node_value", 0.0)) >= 0.8
    if global_failure_only and episode_success:
        return []

    candidate_steps = []
    formatted = []
    for step_index, node in enumerate(path):
        parent = node.parent_node
        siblings = _query_siblings(parent, tokenizer, max_query_tokens, max_action_tokens)
        sibling_values = [float(getattr(sibling, "node_value", 0.0))
                          for sibling, _, _ in siblings]
        if (len(sibling_values) >= 2
                and max(sibling_values) - min(sibling_values) >= min_value_gap):
            candidate_steps.append(step_index)
        start = 0 if parent.is_root else _valid_length(parent)
        end = _valid_length(node)
        edge_ids = [
            int(token) for token in node.responses[start:end].detach().cpu().tolist()
        ]
        formatted.append(f"[Step {step_index}]\n"
                         + tokenizer.decode(edge_ids, skip_special_tokens=False))

    pad_id = int(tokenizer.pad_token_id)
    root = path[0].parent_node
    task_ids = _valid_tokens(root.prompts, None, pad_id)
    task = tokenizer.decode(task_ids, skip_special_tokens=False)
    prompt = _opid_analyzer_prompt(
        task=task,
        outcome=episode_success,
        candidate_steps=candidate_steps,
        formatted_steps="\n".join(formatted),
        max_skill_count=max_skill_count,
    )
    skill = _request_node_skill(analyzer_url, prompt)
    if skill is None or not _validate_opid_skill(skill, candidate_steps):
        return []

    step_skills = {str(key): str(value).strip()
                   for key, value in skill["step_skills"].items()}
    episode_skill = str(skill["episode_skill"]).strip()
    events = []
    for step_index, node in enumerate(path):
        parent = node.parent_node
        query_span = extract_edge_query_span(node, tokenizer)
        span = extract_edge_policy_span(node, tokenizer, max_action_tokens)
        if query_span is None or len(query_span.token_ids) > max_query_tokens or span is None:
            continue
        routed_skill = (
            episode_skill if global_failure_only
            else step_skills.get(str(step_index), episode_skill)
        )
        instruction = (
            "\n<information>Procedural guidance: "
            + routed_skill
            + "</information>\n"
        )
        suffix = _encode(tokenizer, instruction)
        target_ids = list(span.action_ids)
        required = len(suffix) + len(target_ids)
        if required > max_teacher_length:
            continue
        base, context_truncated, prompt_fraction = _anchored_prefix(
            parent, tokenizer, max_teacher_length - required
        )
        teacher_ids = base + suffix + target_ids
        target_start = len(base) + len(suffix)
        events.append(_TeacherEvent(
            teacher_ids=tuple(teacher_ids),
            teacher_positions=tuple(target_start + offset - 1
                                    for offset in span.selected_offsets),
            student_positions=span.student_positions,
            think_mask=span.think_mask,
            query_mask=span.query_mask,
            weight=1.0,
            raw_advantage=0.0,
            value_gap=0.0,
            target_value=float(getattr(node, "node_value", 0.0)),
            sibling_count=len(getattr(parent, "child_node", [])),
            context_truncated=bool(context_truncated),
            prompt_retained_fraction=float(prompt_fraction),
        ))
    return events

def _teacher_event(
    node,
    tokenizer,
    *,
    max_teacher_length: int,
    max_query_tokens: int,
    max_action_tokens: int,
    max_evidence_tokens: int,
    min_value_gap: float,
    min_raw_advantage: float,
    max_advantage_weight: float,
    teacher_context: str = "hindsight",
    gold_answer: Optional[str] = None,
    analyzer_url: str = "",
) -> Optional[_TeacherEvent]:
    parent = node.parent_node
    if parent is None:
        return None
    siblings = _query_siblings(parent, tokenizer, max_query_tokens, max_action_tokens)
    if len(siblings) < 2:
        return None

    values = [float(getattr(sibling, "node_value", 0.0)) for sibling, _, _ in siblings]
    value_gap = max(values) - min(values)
    if not math.isfinite(value_gap) or value_gap < min_value_gap:
        return None

    # One deterministic top target per parent.  TraceCredit already supplies
    # negative gradients for worse branches, so OPD focuses on the best query.
    top_index = max(range(len(siblings)), key=lambda index: values[index])
    target_node, target_query_span, target_span = siblings[top_index]
    if node is not target_node:
        return None

    total_other_count = sum(
        max(1, int(getattr(sibling, "descendant_leaf_count", 1)))
        for index, (sibling, _, _) in enumerate(siblings)
        if index != top_index
    )
    if total_other_count <= 0:
        return None
    sibling_baseline = sum(
        max(1, int(getattr(sibling, "descendant_leaf_count", 1))) * values[index]
        for index, (sibling, _, _) in enumerate(siblings)
        if index != top_index
    ) / total_other_count
    raw_advantage = values[top_index] - sibling_baseline
    if not math.isfinite(raw_advantage) or raw_advantage <= min_raw_advantage:
        return None

    mean_value = sum(values) / len(values)
    std = math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))
    normalized_advantage = raw_advantage / max(std, 1e-6)
    descendant_count = max(1, int(getattr(node, "descendant_leaf_count", 1)))
    weight = min(max_advantage_weight, normalized_advantage) / descendant_count

    if teacher_context == "answer":
        # Answer-conditioned teacher: only the audited gold-answer cheat sheet
        # precedes the query; no hindsight block is constructed.
        if not gold_answer:
            return None
        instruction = _render_answer_cheat(gold_answer)
    elif teacher_context == "node_skill":
        if not analyzer_url:
            return None
        low_index = min(range(len(siblings)), key=lambda index: values[index])
        bad_span = siblings[low_index][2]
        pad_id = int(tokenizer.pad_token_id)
        state_ids = _valid_tokens(parent.input_ids, parent.attention_mask, pad_id)
        state = _redact_answer(
            tokenizer.decode(state_ids, skip_special_tokens=False), gold_answer
        )
        good_action = tokenizer.decode(target_span.action_ids, skip_special_tokens=False)
        bad_action = tokenizer.decode(bad_span.action_ids, skip_special_tokens=False)
        evidence = _redact_answer(
            _extract_edge_evidence(node, tokenizer, max_evidence_tokens), gold_answer
        )
        queries = [tokenizer.decode(span.token_ids, skip_special_tokens=False)
                   for _, span, _ in siblings]
        skill = _request_node_skill(
            analyzer_url, _node_skill_prompt(state, good_action, bad_action, evidence)
        )
        if skill is None or not _validate_node_skill(
            skill, gold_answer=str(gold_answer or ""), queries=queries, visible_state=state
        ):
            return None
        rendered = json.dumps(
            {field: str(skill[field]).strip() for field in _NODE_SKILL_FIELDS},
            ensure_ascii=False, separators=(",", ":"),
        )
        instruction = (
            "\n<analyzer_skill>\n" + rendered + "\n</analyzer_skill>\n"
            "Use this answer-free procedural skill to choose the next action "
            "from the current visible state.\n"
        )
    else:
        # Leave-one-out hindsight: the target query is never present in the
        # privileged block.  Other sibling queries act as explicit alternatives.
        sibling_rows = []
        for index, (sibling, span, _) in enumerate(siblings):
            if index == top_index:
                continue
            sibling_rows.append(
                f"- anonymized alternative branch {len(sibling_rows) + 1}: "
                f"value {values[index]:.4f}"
            )
        evidence = _redact_answer(
            _extract_edge_evidence(node, tokenizer, max_evidence_tokens), gold_answer
        )
        evidence_text = evidence if evidence else "(no retrieval evidence was returned)"
        instruction = (
            "\n<hindsight>\n"
            "A better branch from this exact search state later retrieved:\n"
            f"{evidence_text}\n"
            f"Its branch value was {values[top_index]:.4f}; its local advantage "
            f"over the alternatives was {raw_advantage:.4f}.\n"
            "Outcomes of other attempted searches from the same state "
            "(query text withheld to prevent copying):\n"
            + "\n".join(sibling_rows)
            + "\n</hindsight>\n"
            "Use this hindsight only to correct the direction of the agent's "
            "private reasoning and next search. Never state or imply a final "
            "answer, and never claim access to future information. Continue as "
            "an agent that only knows the visible current state.\n"
        )
        normalized_answer = _normalized_words(str(gold_answer or ""))
        if normalized_answer and normalized_answer in _normalized_words(instruction):
            return None
    suffix = _encode(tokenizer, instruction)
    target_ids = list(target_span.action_ids)
    required = len(suffix) + len(target_ids)
    if required > max_teacher_length:
        return None
    context_budget = max_teacher_length - required
    base, context_truncated, prompt_fraction = _anchored_prefix(
        parent, tokenizer, context_budget
    )
    teacher_ids = base + suffix + target_ids
    target_start = len(base) + len(suffix)
    teacher_positions = tuple(target_start + offset - 1 for offset in target_span.selected_offsets)
    return _TeacherEvent(
        teacher_ids=tuple(teacher_ids),
        teacher_positions=teacher_positions,
        student_positions=target_span.student_positions,
        think_mask=target_span.think_mask,
        query_mask=target_span.query_mask,
        weight=float(weight),
        raw_advantage=float(raw_advantage),
        value_gap=float(value_gap),
        target_value=float(values[top_index]),
        sibling_count=len(siblings),
        context_truncated=bool(context_truncated),
        prompt_retained_fraction=float(prompt_fraction),
    )


def build_self_opd_batch(
    final_node_list: Sequence,
    tokenizer,
    *,
    max_events: int = 3,
    max_teacher_length: int = 1024,
    max_query_tokens: int = 64,
    max_action_tokens: int = 256,
    max_evidence_tokens: int = 128,
    min_value_gap: float = 0.25,
    min_raw_advantage: float = 0.0,
    max_advantage_weight: float = 3.0,
    teacher_context: str = "hindsight",
    gold_answers: Optional[Sequence] = None,
    event_selection: str = "contrast",
    analyzer_url: str = "",
) -> Dict[str, torch.Tensor]:
    """Create batch-aligned TraceCredit-guided OPD tensors."""
    if max_events <= 0 or max_teacher_length <= 1 or max_query_tokens <= 0 or max_action_tokens <= 0:
        raise ValueError("Self-OPD bounds must be positive and teacher length must exceed one")
    if max_evidence_tokens < 0:
        raise ValueError("self_opd max_evidence_tokens must be non-negative")
    if min_value_gap < 0 or min_raw_advantage < 0 or max_advantage_weight <= 0:
        raise ValueError("Self-OPD gap/advantage bounds are invalid")
    if teacher_context not in ("hindsight", "answer", "node_skill", "opid_paper", "opid_global_failure"):
        raise ValueError(f"unknown self_opd teacher_context: {teacher_context!r}")
    if event_selection not in ("contrast", "all"):
        raise ValueError(f"unknown self_opd event_selection: {event_selection!r}")
    if event_selection == "all" and teacher_context not in ("answer", "opid_paper", "opid_global_failure"):
        raise ValueError("event_selection='all' requires answer or opid_paper context")
    if gold_answers is not None and len(gold_answers) != len(final_node_list):
        raise ValueError("gold_answers must align with final_node_list")

    batch_size = len(final_node_list)
    pad_id = int(tokenizer.pad_token_id)
    teacher_ids = torch.full((batch_size, max_events, max_teacher_length), pad_id, dtype=torch.long)
    teacher_attention = torch.zeros_like(teacher_ids)
    teacher_positions = torch.full((batch_size, max_events, max_action_tokens), -1, dtype=torch.long)
    student_positions = torch.full_like(teacher_positions, -1)
    token_mask = torch.zeros_like(teacher_positions, dtype=torch.bool)
    think_mask = torch.zeros_like(teacher_positions, dtype=torch.bool)
    query_mask = torch.zeros_like(teacher_positions, dtype=torch.bool)
    event_mask = torch.zeros((batch_size, max_events), dtype=torch.bool)
    event_weight = torch.zeros((batch_size, max_events), dtype=torch.float32)
    raw_advantage = torch.zeros_like(event_weight)
    value_gap = torch.zeros_like(event_weight)
    target_value = torch.zeros_like(event_weight)
    sibling_count = torch.zeros((batch_size, max_events), dtype=torch.long)
    context_truncated = torch.zeros_like(event_mask)
    prompt_retained_fraction = torch.zeros_like(event_weight)

    emitted_node_ids = set()
    for batch_index, leaf in enumerate(final_node_list):
        events = []
        gold_answer = gold_answers[batch_index] if gold_answers is not None else None
        if teacher_context in ("opid_paper", "opid_global_failure"):
            events = _paper_teacher_events(
                leaf,
                tokenizer,
                max_teacher_length=max_teacher_length,
                max_query_tokens=max_query_tokens,
                max_action_tokens=max_action_tokens,
                min_value_gap=min_value_gap,
                analyzer_url=analyzer_url,
                max_skill_count=max_events,
                global_failure_only=(teacher_context == "opid_global_failure"),
            )
        for node in ([] if teacher_context in ("opid_paper", "opid_global_failure")
                     else _path_from_root(leaf)):
            if event_selection == "all":
                # Tree leaves share ancestors; distill each query node once,
                # on the first leaf whose path reaches it.  Positions index the
                # shared trajectory prefix, so any descendant leaf is valid.
                if id(node) in emitted_node_ids:
                    continue
                event = _dense_teacher_event(
                    node,
                    tokenizer,
                    max_teacher_length=max_teacher_length,
                    max_query_tokens=max_query_tokens,
                    max_action_tokens=max_action_tokens,
                    gold_answer=gold_answer,
                )
                if event is not None:
                    emitted_node_ids.add(id(node))
                    events.append(event)
                continue
            event = _teacher_event(
                node,
                tokenizer,
                max_teacher_length=max_teacher_length,
                max_query_tokens=max_query_tokens,
                max_action_tokens=max_action_tokens,
                max_evidence_tokens=max_evidence_tokens,
                min_value_gap=min_value_gap,
                min_raw_advantage=min_raw_advantage,
                max_advantage_weight=max_advantage_weight,
                teacher_context=teacher_context,
                gold_answer=gold_answer,
                analyzer_url=analyzer_url,
            )
            if event is not None:
                events.append(event)
        events = events[-max_events:]
        for event_index, event in enumerate(events):
            length = len(event.teacher_ids)
            query_length = len(event.teacher_positions)
            offset = max_teacher_length - length
            teacher_ids[batch_index, event_index, offset:] = torch.tensor(event.teacher_ids)
            teacher_attention[batch_index, event_index, offset:] = 1
            teacher_positions[batch_index, event_index, :query_length] = (
                torch.tensor(event.teacher_positions) + offset
            )
            student_positions[batch_index, event_index, :query_length] = torch.tensor(
                event.student_positions
            )
            token_mask[batch_index, event_index, :query_length] = True
            think_mask[batch_index, event_index, :query_length] = torch.tensor(event.think_mask)
            query_mask[batch_index, event_index, :query_length] = torch.tensor(event.query_mask)
            event_mask[batch_index, event_index] = True
            event_weight[batch_index, event_index] = event.weight
            raw_advantage[batch_index, event_index] = event.raw_advantage
            value_gap[batch_index, event_index] = event.value_gap
            target_value[batch_index, event_index] = event.target_value
            sibling_count[batch_index, event_index] = event.sibling_count
            context_truncated[batch_index, event_index] = event.context_truncated
            prompt_retained_fraction[batch_index, event_index] = event.prompt_retained_fraction

    return {
        "self_opd_teacher_input_ids": teacher_ids,
        "self_opd_teacher_attention_mask": teacher_attention,
        "self_opd_teacher_query_positions": teacher_positions,
        "self_opd_student_query_positions": student_positions,
        "self_opd_token_mask": token_mask,
        "self_opd_think_mask": think_mask,
        "self_opd_query_mask": query_mask,
        "self_opd_event_mask": event_mask,
        "self_opd_event_weight": event_weight,
        "self_opd_raw_advantage": raw_advantage,
        "self_opd_value_gap": value_gap,
        "self_opd_target_value": target_value,
        "self_opd_sibling_count": sibling_count,
        "self_opd_context_truncated": context_truncated,
        "self_opd_prompt_retained_fraction": prompt_retained_fraction,
    }
