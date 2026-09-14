from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Callable, Tuple
from pydantic import BaseModel
import json
import random
from collections import deque
import torch
from verl import DataProto
import random
import os

DEBUG: bool = os.environ.get('TREE_SEARCH_DEBUG', '').lower() == 'true'

def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


class TreeNode:
    EXPAND_MODES = frozenset({'random', 'uncertainty_balanced', 'outcome_prior'})

    def __init__(
        self,
        tree_uid: str, # equal to the original question prompt uid
        node_uid: str,
        prompts: Optional[torch.Tensor] = None, # prompts are same as the original input_ids
        input_ids: Optional[torch.Tensor] = None, # input_ids are the left_part input during rollout
        attention_mask: Optional[torch.Tensor] = None, # attention_mask are the left_part input during rollout
        position_ids: Optional[torch.Tensor] = None, # position_ids are the left_part input during rollout
        responses: Optional[torch.Tensor] = None, # responses are the right_part output during rollout
        responses_with_info_mask: Optional[torch.Tensor] = None, # responses_with_info_mask are the right_part output during rollout
        turns_mask: Optional[torch.Tensor] = None, # turn_mask are the right_part output during rollout
        log_prob_node: Optional[float] = 0.0,
        log_prob_list: Optional[list[float]] = [],
        parent_node: Optional['TreeNode'] = None,
        is_root: bool = False,
        is_active: bool = True,
        valid_action_stats: int = 0,
        valid_search_stats: int = 0,
        depth: int = 0,
        is_leaf: bool = False,
        correct_leaf_in_subtree: int = 0,
        reward_mode: str = 'base', 
        tensor_fn = None,
        margin = 0.1,
    ):

        self.tree_uid: int = tree_uid
        self.node_uid: int = node_uid

        self.prompts: torch.Tensor = prompts
        self.input_ids: torch.Tensor = input_ids
        self.attention_mask: torch.Tensor = attention_mask
        self.position_ids: torch.Tensor = position_ids
        self.responses: torch.Tensor = responses
        self.responses_with_info_mask: torch.Tensor = responses_with_info_mask
        self.turns_mask: torch.Tensor = turns_mask

        self.log_prob_node: float = log_prob_node
        self.log_prob_list: list[float] = log_prob_list

        self.parent_node = parent_node
        self._child_node = []

        self.is_root = is_root
        self.is_active = is_active
        self.valid_action_stats = valid_action_stats
        self.valid_search_stats = valid_search_stats
        self.depth = depth
        self.is_leaf = is_leaf

        self.original_score = 0.
        self.final_score = 0.

        self.subtree_leaf_score = 0.

        # Auditable statistics for the signed sibling-counterfactual credit.
        # They are populated only when ``calculate_branch_credit_from_root``
        # is called, so the legacy base/tree_diff paths retain their behavior.
        self.descendant_leaf_count = 0
        self.node_value = 0.0
        self.sibling_baseline = 0.0
        self.edge_credit_raw = 0.0
        self.edge_credit_normalized = 0.0
        self.branch_credit_stats: Dict[str, float] = {}

        # Selection metadata is intentionally kept on the tree, rather than
        # in the training batch.  It makes expansion decisions inspectable in
        # rollout snapshots without changing the policy/reward data path.
        self.last_selection_uncertainty = 0.0
        self.last_selection_outcome_prior = 0.0
        self.last_selection_weight = 1.0
        self.last_selection_after_search = False
        self.last_selection_count = 0
        self.last_expand_selection_stats: Dict[str, Any] = {}
        self.expand_selection_history: List[Dict[str, Any]] = []

        self.reward_mode = reward_mode
        self.margin = margin

        self.tensor_fn = tensor_fn

    @property
    def child_node(self) -> list['TreeNode']:
        return self._child_node

    @child_node.setter
    def child_node(self, value: list['TreeNode']):
        """
        Debug
        """
        print(f"!!! WARNING: Direct assignment to child_node on node {self.node_uid}. New list has {len(value)} children.")
        
        import traceback
        traceback.print_stack()
        
        # Check
        for child in value:
            if child is self:
                raise ValueError(f"CRITICAL: Attempted to directly assign a list containing the node itself as its own child. Node UID: {self.node_uid}")

        self._child_node = value
        
    def add_child(self, child_node: 'TreeNode'):
        if child_node is self:
            raise ValueError("A node cannot be its own child")
        if child_node.node_uid == self.node_uid:
            raise ValueError("node_uid same!! A node cannot be its own child")
        self._child_node.append(child_node)
    
    def get_subtree_nodes(self):
        """
        Dynamically get all descendant nodes by traversing the tree
        """
        nodes = []
        nodes_to_visit = list(self.child_node) # from child nodes
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            nodes.append(current_node)
            nodes_to_visit.extend(current_node.child_node)
        return nodes

    def get_subtree_leaves_num(self):
        num = 0
        nodes_to_visit = list(self.child_node)
        while nodes_to_visit:
            current_node = nodes_to_visit.pop(0)
            if current_node.is_leaf:
                num += 1
            nodes_to_visit.extend(current_node.child_node)
        return num

    @staticmethod
    def _validate_expand_weight(name: str, value: float) -> float:
        """Return a finite, non-negative selector weight."""
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be a finite non-negative number, got {value!r}') from exc
        if not math.isfinite(numeric_value) or numeric_value < 0.0:
            raise ValueError(f'{name} must be a finite non-negative number, got {value!r}')
        return numeric_value

    @classmethod
    def _validate_expand_selector(
            cls,
            mode: str,
            uncertainty_weight: float,
            search_weight: float,
            child_count_weight: float,
            depth_weight: float,
        ) -> Tuple[float, float, float, float]:
        if mode not in cls.EXPAND_MODES:
            raise ValueError(
                f'expand mode must be one of {sorted(cls.EXPAND_MODES)}, got {mode!r}'
            )
        return (
            cls._validate_expand_weight('uncertainty_weight', uncertainty_weight),
            cls._validate_expand_weight('search_weight', search_weight),
            cls._validate_expand_weight('child_count_weight', child_count_weight),
            cls._validate_expand_weight('depth_weight', depth_weight),
        )

    @staticmethod
    def _validate_outcome_prior(name: str, value: float) -> float:
        """Return a finite empirical probability in the closed interval [0, 1]."""
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{name} must be a finite probability in [0, 1], got {value!r}') from exc
        if not math.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
            raise ValueError(f'{name} must be a finite probability in [0, 1], got {value!r}')
        return numeric_value

    @classmethod
    def _validate_outcome_priors(
            cls,
            root: float,
            pre_search: float,
            after_search_depth1: float,
            after_search_depth2: float,
            after_search_depth3plus: float,
            concentration_power: float,
        ) -> Tuple[float, float, float, float, float, float]:
        priors = (
            cls._validate_outcome_prior('outcome_prior_root', root),
            cls._validate_outcome_prior('outcome_prior_pre_search', pre_search),
            cls._validate_outcome_prior(
                'outcome_prior_after_search_depth1', after_search_depth1
            ),
            cls._validate_outcome_prior(
                'outcome_prior_after_search_depth2', after_search_depth2
            ),
            cls._validate_outcome_prior(
                'outcome_prior_after_search_depth3plus', after_search_depth3plus
            ),
        )
        concentration_power = cls._validate_expand_weight(
            'outcome_prior_concentration_power', concentration_power
        )
        if concentration_power == 0.0:
            raise ValueError('outcome_prior_concentration_power must be greater than zero')
        return (*priors, concentration_power)

    @staticmethod
    def _historical_outcome_prior(
            node: 'TreeNode',
            root: float,
            pre_search: float,
            after_search_depth1: float,
            after_search_depth2: float,
            after_search_depth3plus: float,
        ) -> float:
        """Look up the calibrated chance that this parent yields mixed correctness."""
        if node.depth <= 0:
            return root
        if node.valid_search_stats < 1:
            return pre_search
        if node.depth == 1:
            return after_search_depth1
        if node.depth == 2:
            return after_search_depth2
        return after_search_depth3plus

    @staticmethod
    def _selection_uncertainty(node: 'TreeNode') -> float:
        """Estimate next-action uncertainty from a node's generated children.

        Every child was generated as an action from ``node``.  Invalid or
        missing action log-probabilities are ignored; if no finite value is
        available, zero is a conservative finite fallback.
        """
        finite_log_probs = []
        for child in node.child_node:
            try:
                log_prob = float(child.log_prob_node)
            except (TypeError, ValueError):
                continue
            if math.isfinite(log_prob):
                finite_log_probs.append(log_prob)

        if not finite_log_probs:
            return 0.0

        # Inference log-probabilities are normally non-positive.  Clamp an
        # anomalous positive average to zero so selector weights stay valid.
        uncertainty = -sum(finite_log_probs) / len(finite_log_probs)
        return max(0.0, uncertainty) if math.isfinite(uncertainty) else 0.0

    @staticmethod
    def _depth_preference(node: 'TreeNode', max_candidate_depth: int) -> float:
        """Favor non-root, middle-depth nodes while retaining finite weights."""
        if node.depth <= 0:
            return 0.0
        if max_candidate_depth <= 1:
            return 1.0

        normalized_depth = min(max(node.depth / max_candidate_depth, 0.0), 1.0)
        middle_preference = max(0.0, 1.0 - abs(2.0 * normalized_depth - 1.0))
        # Non-root endpoints still receive a small preference; middle-depth
        # nodes receive the full value.
        return 0.25 + 0.75 * middle_preference

    @staticmethod
    def _weighted_sample_without_replacement(
            candidate_set: List['TreeNode'],
            weights: List[float],
            n: int,
        ) -> List['TreeNode']:
        """Sample up to ``n`` candidates with weighted roulette draws."""
        remaining_candidates = list(candidate_set)
        remaining_weights = list(weights)
        result = []
        for _ in range(min(n, len(remaining_candidates))):
            selected = random.choices(remaining_candidates, weights=remaining_weights, k=1)[0]
            selected_index = remaining_candidates.index(selected)
            result.append(selected)
            remaining_candidates.pop(selected_index)
            remaining_weights.pop(selected_index)
        return result

    def _record_expand_selection(
            self,
            mode: str,
            requested_count: int,
            candidate_set: List['TreeNode'],
            candidate_metadata: Dict[int, Dict[str, float]],
            result: List['TreeNode'],
            fallback_with_replacement_count: int = 0,
        ) -> None:
        """Persist auditable selector metadata on the expansion root."""
        selection_counts = {}
        for node in result:
            node_id = id(node)
            selection_counts[node_id] = selection_counts.get(node_id, 0) + 1

        for node in candidate_set:
            metadata = candidate_metadata[id(node)]
            node.last_selection_uncertainty = metadata['uncertainty']
            node.last_selection_outcome_prior = metadata['outcome_prior']
            node.last_selection_weight = metadata['weight']
            node.last_selection_after_search = bool(metadata['after_search'])
            node.last_selection_count = selection_counts.get(id(node), 0)

        selected_count = len(result)
        duplicate_count = selected_count - len(selection_counts)
        if selected_count:
            mean_selected_depth = sum(node.depth for node in result) / selected_count
            fraction_selected_after_search = (
                sum(bool(candidate_metadata[id(node)]['after_search']) for node in result)
                / selected_count
            )
            mean_selection_uncertainty = (
                sum(candidate_metadata[id(node)]['uncertainty'] for node in result)
                / selected_count
            )
            mean_selection_outcome_prior = (
                sum(candidate_metadata[id(node)]['outcome_prior'] for node in result)
                / selected_count
            )
        else:
            mean_selected_depth = 0.0
            fraction_selected_after_search = 0.0
            mean_selection_uncertainty = 0.0
            mean_selection_outcome_prior = 0.0

        stats = {
            'mode': mode,
            'requested_count': requested_count,
            'candidate_count': len(candidate_set),
            'selected_count': selected_count,
            'duplicate_count': duplicate_count,
            'mean_selected_depth': mean_selected_depth,
            'fraction_selected_after_search': fraction_selected_after_search,
            'mean_selection_uncertainty': mean_selection_uncertainty,
            'mean_selection_outcome_prior': mean_selection_outcome_prior,
            'fallback_with_replacement_count': fallback_with_replacement_count,
            'selected_node_uids': [node.node_uid for node in result],
        }
        self.last_expand_selection_stats = stats
        self.expand_selection_history.append(dict(stats))

    def get_expand_node(
            self,
            n: int = 1,
            mode: str = 'random',
            uncertainty_weight: float = 1.0,
            search_weight: float = 1.0,
            child_count_weight: float = 1.0,
            depth_weight: float = 1.0,
            outcome_prior_root: float = 0.23255813953488372,
            outcome_prior_pre_search: float = 0.17880794701986755,
            outcome_prior_after_search_depth1: float = 0.20754716981132076,
            outcome_prior_after_search_depth2: float = 0.11363636363636363,
            outcome_prior_after_search_depth3plus: float = 0.09090909090909091,
            outcome_prior_concentration_power: float = 2.0,
        ) -> List['TreeNode']:
        """Select expansion nodes from this root and its non-leaf descendants.

        ``random`` deliberately retains legacy ``random.choices`` sampling
        with replacement.  ``uncertainty_balanced`` uses weighted sampling
        without replacement while enough candidates exist. ``outcome_prior``
        uses calibrated historical mixed-correctness rates and samples with
        replacement so several continuations can share a promising prefix.
        """
        (
            uncertainty_weight,
            search_weight,
            child_count_weight,
            depth_weight,
        ) = self._validate_expand_selector(
            mode,
            uncertainty_weight,
            search_weight,
            child_count_weight,
            depth_weight,
        )
        (
            outcome_prior_root,
            outcome_prior_pre_search,
            outcome_prior_after_search_depth1,
            outcome_prior_after_search_depth2,
            outcome_prior_after_search_depth3plus,
            outcome_prior_concentration_power,
        ) = self._validate_outcome_priors(
            outcome_prior_root,
            outcome_prior_pre_search,
            outcome_prior_after_search_depth1,
            outcome_prior_after_search_depth2,
            outcome_prior_after_search_depth3plus,
            outcome_prior_concentration_power,
        )

        candidate_set = [self]
        for node in self.get_subtree_nodes():
            if not node.is_leaf:
                candidate_set.append(node)

        max_candidate_depth = max(node.depth for node in candidate_set)
        candidate_metadata = {}
        weights = []
        for node in candidate_set:
            uncertainty = self._selection_uncertainty(node)
            after_search = float(node.valid_search_stats >= 1)
            child_preference = 1.0 / (1.0 + len(node.child_node))
            depth_preference = self._depth_preference(node, max_candidate_depth)
            outcome_prior = self._historical_outcome_prior(
                node,
                outcome_prior_root,
                outcome_prior_pre_search,
                outcome_prior_after_search_depth1,
                outcome_prior_after_search_depth2,
                outcome_prior_after_search_depth3plus,
            )
            if mode == 'uncertainty_balanced':
                # A tiny shared base retains stochastic exploration even when
                # every optional weight is disabled.
                weight = (
                    1e-6
                    + uncertainty_weight * uncertainty
                    + search_weight * after_search
                    + child_count_weight * child_preference
                    + depth_weight * depth_preference
                )
                # Valid finite configuration values can still overflow when
                # multiplied by an unusually large finite uncertainty.
                if not math.isfinite(weight):
                    weight = 1e300
            elif mode == 'outcome_prior':
                # The empirical probability is the auditable score.  Raising
                # it to a configurable power controls how strongly rollout
                # budget is concentrated, while a tiny floor keeps every
                # candidate reachable when a calibrated bucket is zero.
                weight = 1e-6 + outcome_prior ** outcome_prior_concentration_power
            else:
                weight = 1.0
            candidate_metadata[id(node)] = {
                'uncertainty': uncertainty,
                'after_search': after_search,
                'outcome_prior': outcome_prior,
                'weight': weight,
            }
            weights.append(weight)

        if mode == 'random':
            # Keep this exact legacy call so existing seeded runs retain the
            # same with-replacement behavior.
            result = random.choices(candidate_set, k=n)
            fallback_with_replacement_count = 0
        elif mode == 'uncertainty_balanced':
            result = self._weighted_sample_without_replacement(candidate_set, weights, n)
            fallback_with_replacement_count = max(0, n - len(candidate_set))
            if fallback_with_replacement_count:
                result.extend(
                    random.choices(
                        candidate_set,
                        weights=weights,
                        k=fallback_with_replacement_count,
                    )
                )
        else:
            result = random.choices(candidate_set, weights=weights, k=n)
            fallback_with_replacement_count = 0

        self._record_expand_selection(
            mode=mode,
            requested_count=n,
            candidate_set=candidate_set,
            candidate_metadata=candidate_metadata,
            result=result,
            fallback_with_replacement_count=fallback_with_replacement_count,
        )
        assert len(result) == n, f"get_expand_node error, len(result)={len(result)} != n={n}"
        return result

    def sample_leaf(self, n: int = 1) -> List['TreeNode']:
        """
        Sample n leaves, then prune the tree (drop the unselected nodes)
        """

        candidate_uid_set = []

        for node in self.get_subtree_nodes():
            if node.is_leaf:
                candidate_uid_set.append(node.node_uid)

        if len(candidate_uid_set) < n:
            dprint(f"root={self.node_uid}, candidate_uid_set={candidate_uid_set}")
            subtree_nodes = self.get_subtree_nodes()
            subtree_node_uids = [node.node_uid for node in subtree_nodes]
            dprint(f"all subtree nodes={subtree_node_uids}")
        assert len(candidate_uid_set) >= n, f"root={self.node_uid}, candidate_uid_set len={len(candidate_uid_set)} < n={n}"

        random.shuffle(candidate_uid_set)
        dprint(f'original candidate_uid_set len={len(candidate_uid_set)}')
        candidate_uid_set = candidate_uid_set[:n]
        dprint(f'sampled candidate_uid_set len={len(candidate_uid_set)}')
        self._prune_subtree(candidate_uid_set)

        result = []
        for node in self.get_subtree_nodes():
            if node.is_leaf:
                result.append(node)
        return result

    def set_leaf_original_score(self, score: float):
        """
        Set the original score of the leaf
        """
        self.original_score = score

    @staticmethod
    def dfs_subtree_leaf_score(tmp_node: 'TreeNode') -> float:
        """
        Do dfs and compute the subtree leaf original score
        """
        subtree_leaf_score = tmp_node.original_score
        for node in tmp_node.child_node:
            subtree_leaf_score += TreeNode.dfs_subtree_leaf_score(node)
        tmp_node.subtree_leaf_score = subtree_leaf_score
        return subtree_leaf_score

    def calculate_final_score_from_root(self):
        """
        Calculate the diff-based final score from the root (not used by the base tree-rollout mode).
        """

        # First do dfs and compute the subtree leaf original score
        TreeNode.dfs_subtree_leaf_score(self)

        # Then compute the final score for each node
        total_leaf_num = self.get_subtree_leaves_num()
        global_score_mean = self.subtree_leaf_score / total_leaf_num
        for node in self.get_subtree_nodes():
            # TreeRL global score
            curr_leaf_num = 1 if node.is_leaf else node.get_subtree_leaves_num()
            subtree_nodes = node.get_subtree_nodes()
            subtree_node_uids = [x.node_uid for x in subtree_nodes]
            assert curr_leaf_num > 0, f"node_uid={node.node_uid} have no leaves, subtree_node_uids={subtree_node_uids}"
            curr_score_mean = node.subtree_leaf_score / curr_leaf_num    
            global_score = curr_score_mean - global_score_mean
            global_score = 0.
            # TreeRL local score
            parent_leaf_num = node.parent_node.get_subtree_leaves_num()
            parent_score_mean = node.parent_node.subtree_leaf_score / parent_leaf_num
            local_score = curr_score_mean - parent_score_mean
            
            diff_score = global_score + local_score
            diff_score = max(diff_score - self.margin, 0.)

            # final score = diff_score + curr_score_mean
            final_score = diff_score + curr_score_mean
            node.final_score = final_score / math.sqrt(curr_leaf_num)

    def calculate_branch_credit_from_root(
        self,
        normalization: str = 'sibling_std',
        clip: Optional[float] = 3.0,
        eps: float = 1e-6,
        value_mode: str = 'reward',
        correctness_threshold: float = 0.8,
    ) -> Dict[str, float]:
        """Compute signed counterfactual credit for every retained tree edge.

        Node values are means over descendant leaf rewards.  For a child, the
        counterfactual baseline is the leaf-count-weighted mean of all *other*
        child subtrees under the same parent.  ``sibling_std`` uses the
        population standard deviation of the child subtree values; a
        zero-variance sibling set deliberately receives zero local credit.

        ``value_mode='correctness'`` converts the shaped terminal reward to a
        binary answer-correctness signal before computing node values.  This
        prevents sibling standardization from promoting a small format-only
        reward gap to the same scale as an answer-correctness gap.

        The method is intended to run after leaf rewards are assigned and
        after final-leaf pruning, so only training trajectories influence the
        counterfactual values.
        """
        if normalization not in {'none', 'sibling_std'}:
            raise ValueError(
                f'unsupported branch credit normalization: {normalization!r}'
            )
        if clip is not None and clip < 0:
            raise ValueError(f'branch credit clip must be non-negative, got {clip}')
        if value_mode not in {'reward', 'correctness'}:
            raise ValueError(f'unsupported branch credit value mode: {value_mode!r}')
        if not math.isfinite(correctness_threshold):
            raise ValueError(
                'branch credit correctness threshold must be finite, got '
                f'{correctness_threshold}'
            )

        def leaf_value(node: 'TreeNode') -> float:
            reward = float(node.original_score)
            if value_mode == 'correctness':
                return float(reward >= correctness_threshold)
            return reward

        def populate_values(node: 'TreeNode') -> Tuple[int, float]:
            if not node.child_node:
                node.descendant_leaf_count = 1
                node.node_value = leaf_value(node)
                return 1, node.node_value

            leaf_count = 0
            reward_sum = 0.0
            for child in node.child_node:
                child_count, child_value = populate_values(child)
                leaf_count += child_count
                reward_sum += child_count * child_value
            if leaf_count == 0:
                raise ValueError(f'node {node.node_uid!r} has no descendant leaves')
            node.descendant_leaf_count = leaf_count
            node.node_value = reward_sum / leaf_count
            return leaf_count, node.node_value

        populate_values(self)
        stats = {
            'edges_with_siblings': 0,
            'edges_without_siblings': 0,
            'zero_variance_parents': 0,
            'parents_with_multiple_children': 0,
            'clipped_edges': 0,
            'total_edges': 0,
        }

        for parent in [self] + self.get_subtree_nodes():
            children = parent.child_node
            if not children:
                continue
            total_count = sum(child.descendant_leaf_count for child in children)
            total_reward = sum(
                child.descendant_leaf_count * child.node_value for child in children
            )
            has_siblings = len(children) > 1
            std = 0.0
            zero_variance = False
            if has_siblings:
                stats['parents_with_multiple_children'] += 1
                mean_value = sum(child.node_value for child in children) / len(children)
                variance = sum(
                    (child.node_value - mean_value) ** 2 for child in children
                ) / len(children)
                std = math.sqrt(variance)
                zero_variance = normalization == 'sibling_std' and std <= eps
                if zero_variance:
                    stats['zero_variance_parents'] += 1

            for child in children:
                stats['total_edges'] += 1
                if not has_siblings:
                    stats['edges_without_siblings'] += 1
                    baseline = child.node_value
                    raw_credit = 0.0
                else:
                    stats['edges_with_siblings'] += 1
                    other_count = total_count - child.descendant_leaf_count
                    if other_count <= 0:
                        raise ValueError(
                            f'node {child.node_uid!r} has siblings but no sibling leaves'
                        )
                    baseline = (
                        total_reward
                        - child.descendant_leaf_count * child.node_value
                    ) / other_count
                    raw_credit = child.node_value - baseline

                if normalization == 'sibling_std':
                    normalized = 0.0 if zero_variance or not has_siblings else raw_credit / (std + eps)
                else:
                    normalized = raw_credit

                unclipped = normalized
                if clip is not None:
                    normalized = max(-float(clip), min(float(clip), normalized))
                if normalized != unclipped:
                    stats['clipped_edges'] += 1

                child.sibling_baseline = float(baseline)
                child.edge_credit_raw = float(raw_credit)
                child.edge_credit_normalized = float(normalized)

        stats['zero_variance_parent_fraction'] = (
            stats['zero_variance_parents'] / stats['parents_with_multiple_children']
            if stats['parents_with_multiple_children'] else 0.0
        )
        stats['clipped_fraction'] = (
            stats['clipped_edges'] / stats['edges_with_siblings']
            if stats['edges_with_siblings'] else 0.0
        )
        self.branch_credit_stats = stats
        return stats

    def get_branch_credit_from_leaf(
        self,
        policy_token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map path-edge credits to newly generated policy-token intervals.

        ``responses`` on every node is cumulative.  Consequently, the tokens
        introduced by edge ``parent -> child`` occupy the interval between
        their valid cumulative lengths.  The supplied policy mask should be
        one for model-generated tokens and zero for observations/padding.
        """
        if self.responses is None:
            raise ValueError('cannot align branch credit without leaf responses')
        tensor_fn = self.tensor_fn
        if tensor_fn is None:
            raise ValueError('cannot align branch credit without tensor_fn')

        valid_response_mask = tensor_fn.create_attention_mask(self.responses).bool()
        if policy_token_mask is None:
            if self.responses_with_info_mask is None:
                raise ValueError(
                    'policy_token_mask or responses_with_info_mask is required'
                )
            policy_token_mask = tensor_fn.create_attention_mask(
                self.responses_with_info_mask
            )
        policy_token_mask = policy_token_mask.to(
            device=self.responses.device, dtype=torch.bool
        )
        if policy_token_mask.shape != self.responses.shape:
            raise ValueError(
                'policy_token_mask must have the same shape as leaf responses: '
                f'{tuple(policy_token_mask.shape)} != {tuple(self.responses.shape)}'
            )
        policy_token_mask = policy_token_mask & valid_response_mask

        advantages = torch.zeros_like(self.responses, dtype=torch.float32)
        credit_mask = torch.zeros_like(self.responses, dtype=torch.bool)
        path = []
        node = self
        while node is not None and not node.is_root:
            path.append(node)
            node = node.parent_node
        path.reverse()

        for child in path:
            parent = child.parent_node
            if parent is None or parent.is_root or parent.responses is None:
                start = 0
            else:
                start = int(tensor_fn.create_attention_mask(parent.responses).sum().item())
            end = int(tensor_fn.create_attention_mask(child.responses).sum().item())
            end = min(end, self.responses.numel())
            if end <= start:
                continue
            interval_mask = policy_token_mask[start:end]
            advantages[start:end][interval_mask] = child.edge_credit_normalized
            credit_mask[start:end][interval_mask] = True

        return advantages, credit_mask

    def get_token_level_score_from_leaf(self):
        """
        Get the token-level score from the leaf
        """
        final_token_level_scores = torch.zeros_like(self.responses, dtype=torch.float32)

        # Diff-based Reward
        if self.reward_mode == 'tree_diff':
            valid_response_length_list = []
            scores_list = []
            node = self.parent_node
            while node:
                if node.is_root:
                    break
                valid_response_length = self.tensor_fn.create_attention_mask(node.responses).sum()
                valid_response_length_list.append(valid_response_length)
                scores_list.append(node.final_score)
                node = node.parent_node
            scores_list.append(0.)
            valid_response_length_list.append(0)
            valid_response_length_list.reverse()
            scores_list.reverse()
            for i in range(1, len(valid_response_length_list)):
                score = scores_list[i]
                
                l = valid_response_length_list[i-1]
                r = valid_response_length_list[i] - 1

                if l < r:
                    final_token_level_scores[l:r] = score
        # The default tree-rollout path uses base mode.
        else:
            valid_response_length = self.tensor_fn.create_attention_mask(self.responses).sum()
            final_token_level_scores[valid_response_length-1] = self.original_score

        return final_token_level_scores

    def _prune_subtree(self, candidate_uid_set: List[int]) -> bool:
        """
        Drop all the leaves not in cdandidate_uid_set
        """
        surviving_children = []
        dprint(f'start, node={self.node_uid}, child_node len={len(self._child_node)}')

        for child in self.child_node:
            dprint(f'node={self.node_uid}, iter for child={child.node_uid}')
            if child._prune_subtree(candidate_uid_set):
                surviving_children.append(child)
        
        # update child node
        self._child_node = surviving_children
                
        # Determine whether the current node should be retained. The conditions for retention are:
        # 1. It is in candidate_uid_set
        # 2. Or, it still has child nodes after pruning
        should_keep_this_node = self.node_uid in candidate_uid_set or (len(self._child_node) > 0)

        dprint(f'end, node={self.node_uid}, is keeped={should_keep_this_node}')

        return should_keep_this_node

    def check_all_nodes_child(self):
        for node in self.get_subtree_nodes():
            node_child_list = node.child_node
            node_child_node_uid_list = []
            for node_child_node in node_child_list:
                node_child_node_uid_list.append(node_child_node.node_uid)
            if node.node_uid in node_child_node_uid_list:
                dprint(f'error!! node.uid in node_child_node_uid_list!!! is_root={node.is_root}, is_leaf={node.is_leaf}, node_uid={node.node_uid}')

    def delete_tree_from_root(self):
        """
        Delete the tree from root
        """
        all_nodes_list = self.get_subtree_nodes()
        all_nodes_list.append(self)

        for node in all_nodes_list:
            node.prompts = None
            node.input_ids = None
            node.attention_mask = None
            node.position_ids = None
            node.responses = None
            node.responses_with_info_mask = None

            node._child_node.clear()
            node.parent_node = None

            node.log_prob_list.clear()

        
