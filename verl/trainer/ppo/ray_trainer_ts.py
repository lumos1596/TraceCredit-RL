# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import re
import json
from collections import defaultdict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import re
from search_r1.llm_agent.generation_ts import LLMGenerationTreeSearchManager, GenerationTreeSearchConfig
from search_r1.llm_agent.teacher_rescue import rescue_all_wrong, pure_em_rows
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def select_dapo_effective_rows(uids, scores, group_size, max_groups=None, correct_threshold=0.8, rescue_uids=()):
    """Return rows from EM-mixed groups, following DAPO dynamic sampling.

    A trajectory is treated as EM-correct when its shaped QA score is at least
    0.8 (valid-format correct answers score 1.0; correct answers with malformed
    formatting score 0.8). Group order follows first appearance for
    deterministic truncation when the target is reached.
    """
    grouped_rows = {}
    for row, uid in enumerate(uids):
        grouped_rows.setdefault(uid, []).append(row)

    effective_uids = []
    rescue_uids = set(rescue_uids)
    rescued_groups = 0
    all_correct = 0
    all_wrong = 0
    for uid, rows in grouped_rows.items():
        if len(rows) != group_size:
            raise ValueError(f'uid {uid!r} has {len(rows)} trajectories; expected {group_size}')
        correct = sum(float(scores[row]) >= correct_threshold for row in rows)
        if correct == 0:
            all_wrong += 1
            if uid in rescue_uids:
                effective_uids.append(uid)
                rescued_groups += 1
        elif correct == group_size:
            all_correct += 1
        else:
            effective_uids.append(uid)

    if max_groups is not None:
        effective_uids = effective_uids[:max_groups]
    selected = [row for uid in effective_uids for row in grouped_rows[uid]]
    return selected, {
        'total_groups': len(grouped_rows),
        'effective_groups': len(effective_uids),
        'all_correct_groups': all_correct,
        'all_wrong_groups': all_wrong,
        'rescued_groups': rescued_groups,
    }


def subset_dataproto_rows(data, rows):
    """Select batch rows and keep batch-aligned metadata in sync."""
    rows = list(rows)
    indices = torch.as_tensor(rows, dtype=torch.long)
    indices_np = np.asarray(rows)
    original_length = len(data)
    meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == original_length:
            meta_info[key] = value[indices]
        elif isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == original_length:
            meta_info[key] = value[indices_np]
        elif isinstance(value, list) and len(value) == original_length:
            meta_info[key] = [value[index] for index in rows]
        elif isinstance(value, tuple) and len(value) == original_length:
            meta_info[key] = tuple(value[index] for index in rows)
        else:
            meta_info[key] = value
    return DataProto(
        batch=data.batch[indices],
        non_tensor_batch={key: value[indices_np] for key, value in data.non_tensor_batch.items()},
        meta_info=meta_info,
    )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl', kl_fix=""):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
        if kl_fix == "detach":
            kld = kld.clamp(-10, 10)
            kld = kld.detach()
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1,
                      branch_credit_coef=1.0):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'rloo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        response_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'tree':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        tree_index = data.non_tensor_batch['tree_uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        inter_tree_advantages, inter_tree_returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        inner_tree_advantages, inner_tree_returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=tree_index)

        data.batch['inter_tree_advantages'] = inter_tree_advantages
        data.batch['inter_tree_returns'] = inter_tree_returns
        data.batch['inner_tree_advantages'] = inner_tree_advantages
        data.batch['inner_tree_returns'] = inner_tree_returns
        
        data.batch['advantages'] = inter_tree_advantages + inner_tree_advantages
        data.batch['returns'] = inter_tree_returns + inner_tree_returns
    elif adv_estimator == 'branch_credit':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        response_mask = data.batch['attention_mask'][:, -response_length:].bool()
        if 'branch_advantages' not in data.batch or 'branch_credit_mask' not in data.batch:
            raise ValueError(
                'branch_credit estimator requires rollout tensors '
                'branch_advantages and branch_credit_mask'
            )
        global_advantages, global_returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
        )
        branch_mask = data.batch['branch_credit_mask'].bool() & response_mask
        local_advantages = data.batch['branch_advantages'].float() * branch_mask
        coef = float(branch_credit_coef)
        data.batch['global_advantages'] = global_advantages
        data.batch['local_branch_advantages'] = local_advantages
        data.batch['advantages'] = global_advantages + coef * local_advantages
        data.batch['returns'] = global_returns + coef * local_advantages
    elif adv_estimator == 'tree_2norm':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        tree_index = data.non_tensor_batch['tree_uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        tree_advantages, tree_returns = core_algos.compute_tree_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        tree_index=tree_index)
        
        data.batch['advantages'] = tree_advantages
        data.batch['returns'] = tree_returns
    elif adv_estimator == 'tree_inner':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        tree_index = data.non_tensor_batch['tree_uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        inner_tree_advantages, inner_tree_returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=tree_index)
        data.batch['inner_tree_advantages'] = inner_tree_advantages
        data.batch['inner_tree_returns'] = inner_tree_returns
        
        data.batch['advantages'] = inner_tree_advantages
        data.batch['returns'] = inner_tree_returns
    elif adv_estimator == 'tree_per_token':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        tree_index = data.non_tensor_batch['tree_uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        inter_tree_advantages, inter_tree_returns = core_algos.compute_grpo_per_token_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        inner_tree_advantages, inner_tree_returns = core_algos.compute_grpo_per_token_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=tree_index)

        data.batch['inter_tree_advantages'] = inter_tree_advantages
        data.batch['inter_tree_returns'] = inter_tree_returns
        data.batch['inner_tree_advantages'] = inner_tree_advantages
        data.batch['inner_tree_returns'] = inner_tree_returns
        
        data.batch['advantages'] = inter_tree_advantages + inner_tree_advantages
        data.batch['returns'] = inter_tree_returns + inner_tree_returns
    elif adv_estimator == 'no_estimator':
        token_level_rewards = data.batch['token_level_rewards']
        data.batch['advantages'] = token_level_rewards
        data.batch['returns'] = token_level_rewards
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True, reward_mode='tree'):
    # TODO: add response length
    if reward_mode == 'tree':
        sequence_score = batch.batch['token_level_scores'].mean(-1)
        sequence_reward = batch.batch['token_level_rewards'].mean(-1)
    else:
        sequence_score = batch.batch['token_level_scores'].sum(-1)
        sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if 'inner_tree_advantages' in batch.batch:
        valid_inner_tree_adv = torch.masked_select(batch.batch['inner_tree_advantages'], response_mask)
    else:
        valid_inner_tree_adv = None
    if 'inter_tree_advantages' in batch.batch:
        valid_inter_tree_adv = torch.masked_select(batch.batch['inter_tree_advantages'], response_mask)
    else:
        valid_inter_tree_adv = None

    branch_metrics = {}
    if 'local_branch_advantages' in batch.batch:
        branch_mask = batch.batch['branch_credit_mask'].bool() & response_mask
        policy_mask = batch.batch.get('loss_mask', response_mask).bool() & response_mask
        local_advantages = batch.batch['local_branch_advantages']
        global_advantages = batch.batch['global_advantages']
        valid_local = local_advantages[branch_mask]
        policy_local = local_advantages[policy_mask]
        policy_global = global_advantages[policy_mask]
        if valid_local.numel() > 0:
            branch_metrics.update({
                'branch_credit/mean': valid_local.mean().detach().item(),
                'branch_credit/std': valid_local.std(unbiased=False).detach().item(),
                'branch_credit/min': valid_local.min().detach().item(),
                'branch_credit/max': valid_local.max().detach().item(),
                'branch_credit/positive_fraction': (valid_local > 0).float().mean().detach().item(),
                'branch_credit/negative_fraction': (valid_local < 0).float().mean().detach().item(),
                'branch_credit/nonzero_fraction': (valid_local != 0).float().mean().detach().item(),
            })
        else:
            branch_metrics.update({
                'branch_credit/mean': 0.0,
                'branch_credit/std': 0.0,
                'branch_credit/min': 0.0,
                'branch_credit/max': 0.0,
                'branch_credit/positive_fraction': 0.0,
                'branch_credit/negative_fraction': 0.0,
                'branch_credit/nonzero_fraction': 0.0,
            })
        policy_tokens = policy_mask.sum().clamp_min(1)
        global_norm = torch.linalg.vector_norm(policy_global.float())
        local_norm = torch.linalg.vector_norm(policy_local.float())
        branch_metrics.update({
            'branch_credit/token_coverage': (
                (branch_mask & policy_mask).sum().float() / policy_tokens
            ).detach().item(),
            'branch_credit/global_norm': global_norm.detach().item(),
            'branch_credit/local_norm': local_norm.detach().item(),
            'branch_credit/local_to_global_ratio': (
                local_norm / global_norm.clamp_min(1e-12)
            ).detach().item(),
        })
        edge_stat_names = (
            'edges_with_siblings',
            'edges_without_siblings',
            'zero_variance_parents',
            'parents_with_multiple_children',
            'clipped_edges',
        )
        edge_stats = {
            name: batch.batch[f'branch_credit_{name}'].sum().detach().item()
            for name in edge_stat_names
            if f'branch_credit_{name}' in batch.batch
        }
        if edge_stats:
            branch_metrics.update({
                'branch_credit/edges_with_siblings': edge_stats.get('edges_with_siblings', 0.0),
                'branch_credit/edges_without_siblings': edge_stats.get('edges_without_siblings', 0.0),
                'branch_credit/zero_variance_parent_fraction': (
                    edge_stats.get('zero_variance_parents', 0.0)
                    / max(edge_stats.get('parents_with_multiple_children', 0.0), 1.0)
                ),
                'branch_credit/clipped_fraction': (
                    edge_stats.get('clipped_edges', 0.0)
                    / max(edge_stats.get('edges_with_siblings', 0.0), 1.0)
                ),
            })

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            'critic/inter_tree_advantages/mean':
                torch.mean(valid_inter_tree_adv).detach().item(),
            'critic/inter_tree_advantages/max':
                torch.max(valid_inter_tree_adv).detach().item(),
            'critic/inter_tree_advantages/min':
                torch.min(valid_inter_tree_adv).detach().item(),
        } if valid_inter_tree_adv is not None else {}),
        **({
            'critic/inner_tree_advantages/mean':
                torch.mean(valid_inner_tree_adv).detach().item(),
            'critic/inner_tree_advantages/max':
                torch.max(valid_inner_tree_adv).detach().item(),
            'critic/inner_tree_advantages/min':
                torch.min(valid_inner_tree_adv).detach().item(),
        } if valid_inner_tree_adv is not None else {}),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        **branch_metrics,
    }

    # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        valid_actions = np.asarray(batch.meta_info['valid_action_stats'], dtype=np.float32)
        turns = np.asarray(batch.meta_info['turns_stats'], dtype=np.float32)
        valid_action_ratio = np.divide(valid_actions, turns, out=np.zeros_like(valid_actions), where=turns != 0)
        metrics['env/ratio_of_valid_action'] = float(valid_action_ratio.mean())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())


    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout', 'update_policy']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None,
                 debug=False,
                 ):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        ##### for test
        if not debug:
            self._create_dataloader()
            self._init_logger()
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(k): RayPPOTrainer._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [RayPPOTrainer._json_safe(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        return value

    def _dump_tree_rollouts(self, output: DataProto, generation_manager, chunk_idx=None):
        dump_dir = self.config.trainer.get('rollout_dump_dir', None)
        if not dump_dir:
            return

        os.makedirs(dump_dir, exist_ok=True)
        chunk_suffix = '' if chunk_idx is None else f'_chunk_{chunk_idx:02d}'
        selected_path = os.path.join(dump_dir, f'step_{self.global_steps:06d}{chunk_suffix}_selected.jsonl')
        trees_path = os.path.join(dump_dir, f'step_{self.global_steps:06d}{chunk_suffix}_trees.jsonl')
        response_mask = self.tokenizer.pad_token_id != output.batch['responses']

        with open(selected_path, 'w', encoding='utf-8') as handle:
            for i in range(len(output)):
                prompt_ids = output.batch['prompts'][i]
                prompt_ids = prompt_ids[prompt_ids != self.tokenizer.pad_token_id]
                response_ids = output.batch['responses'][i][response_mask[i]]
                reward_model = output.non_tensor_batch.get('reward_model', np.array([None] * len(output), dtype=object))[i]
                record = {
                    'global_step': self.global_steps,
                    'sample_index': i,
                    'uid': output.non_tensor_batch.get('uid', np.array([None] * len(output), dtype=object))[i],
                    'tree_uid': output.non_tensor_batch['tree_uid'][i],
                    'node_uid': output.non_tensor_batch['node_uid'][i],
                    'data_source': output.non_tensor_batch.get('data_source', np.array([None] * len(output), dtype=object))[i],
                    'ground_truth': reward_model.get('ground_truth') if isinstance(reward_model, dict) else None,
                    'original_score': output.non_tensor_batch['original_score'][i],
                    'turn_count': output.meta_info['turns_stats'][i],
                    'is_active': output.meta_info['active_mask'][i],
                    'valid_action_count': output.meta_info['valid_action_stats'][i],
                    'valid_search_count': output.meta_info['valid_search_stats'][i],
                    'prompt': self.tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=False),
                    'response': self.tokenizer.decode(response_ids.tolist(), skip_special_tokens=False),
                    'response_token_count': int(response_mask[i].sum().item()),
                    'trajectory_limit': int(self.config.data.max_start_length),
                    'was_trajectory_clipped': int(response_mask[i].sum().item()) >= int(self.config.data.max_start_length),
                }
                handle.write(json.dumps(self._json_safe(record), ensure_ascii=False) + '\n')

        with open(trees_path, 'w', encoding='utf-8') as handle:
            for snapshot in generation_manager.last_tree_snapshots:
                snapshot = {'global_step': self.global_steps, **snapshot}
                handle.write(json.dumps(self._json_safe(snapshot), ensure_ascii=False) + '\n')

        print(f'[rollout dump] selected={selected_path}, trees={trees_path}')

    @staticmethod
    def _merge_chunk_meta_info(data: list[DataProto]) -> dict:
        """Merge batch-aligned metadata after chunking while retaining scalars."""
        if not data:
            return {}

        def values_equal(left, right):
            if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
                return torch.equal(left, right)
            if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
                return np.array_equal(left, right)
            return left == right

        keys = set().union(*(item.meta_info.keys() for item in data))
        merged = {}
        for key in keys:
            values = [item.meta_info[key] for item in data if key in item.meta_info]
            if len(values) != len(data):
                raise ValueError(f'metadata key {key!r} is missing from one or more rollout chunks')

            first = values[0]
            if all(isinstance(value, torch.Tensor) for value in values) and first.ndim > 0:
                merged[key] = torch.cat(values, dim=0)
            elif all(isinstance(value, np.ndarray) for value in values) and first.ndim > 0:
                merged[key] = np.concatenate(values, axis=0)
            elif all(isinstance(value, (list, tuple)) for value in values):
                merged[key] = [item for value in values for item in value]
            elif all(values_equal(value, first) for value in values[1:]):
                merged[key] = first
            else:
                raise ValueError(f'cannot safely merge metadata key {key!r}')
        return merged

    def _concat_rollout_chunks_cpu(self, chunks: list[DataProto]) -> DataProto:
        """Move rollout chunks to CPU and concatenate tensors and aligned metadata."""
        if not chunks:
            raise ValueError('cannot concatenate an empty rollout chunk list')

        cpu_chunks = []
        for chunk in chunks:
            chunk = chunk.to('cpu')
            cpu_chunks.append(chunk)

        # Different chunks can have different maximum response lengths. Pad
        # sequence tensors before torch.cat; right-padding preserves the
        # existing response/attention-mask convention.
        batch_keys = set().union(*(chunk.batch.keys() for chunk in cpu_chunks))
        pad_token_id = int(self.tokenizer.pad_token_id)
        pad_keys = {'responses', 'prompts', 'input_ids'}
        for key in batch_keys:
            tensors = [chunk.batch[key] for chunk in cpu_chunks]
            if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
                raise TypeError(f'rollout batch key {key!r} is not a tensor')
            if all(tensor.shape == tensors[0].shape for tensor in tensors[1:]):
                continue
            if any(tensor.ndim < 2 for tensor in tensors):
                raise ValueError(f'cannot concatenate non-sequence tensor key {key!r}')
            if any(tensor.shape[2:] != tensors[0].shape[2:] for tensor in tensors[1:]):
                raise ValueError(f'non-leading dimensions differ for rollout key {key!r}')
            max_length = max(tensor.shape[1] for tensor in tensors)
            pad_value = pad_token_id if key in pad_keys else 0
            for chunk, tensor in zip(cpu_chunks, tensors):
                if tensor.shape[1] == max_length:
                    continue
                padding = tensor.new_full(
                    (tensor.shape[0], max_length - tensor.shape[1], *tensor.shape[2:]),
                    pad_value,
                )
                chunk.batch[key] = torch.cat([tensor, padding], dim=1)

        non_tensor_keys = set().union(*(chunk.non_tensor_batch.keys() for chunk in cpu_chunks))
        for key in non_tensor_keys:
            values = []
            for chunk_idx, chunk in enumerate(cpu_chunks):
                if key not in chunk.non_tensor_batch:
                    raise ValueError(
                        f'non-tensor key {key!r} is missing from rollout chunk {chunk_idx}'
                    )

                value = chunk.non_tensor_batch[key]
                expected_length = len(chunk)
                try:
                    value_length = len(value)
                except TypeError as exc:
                    raise ValueError(
                        f'non-tensor key {key!r} in rollout chunk {chunk_idx} '
                        f'is not batch-aligned (value has no batch dimension)'
                    ) from exc
                if value_length != expected_length:
                    raise ValueError(
                        f'non-tensor key {key!r} in rollout chunk {chunk_idx} '
                        f'is not batch-aligned: got length {value_length}, '
                        f'expected {expected_length}'
                    )
                values.append(value)

            # Nested dimensions belong to individual samples. Store each row
            # as one object so DataProto.concat cannot concatenate those
            # dimensions across the batch axis.
            if any(isinstance(value, np.ndarray) and value.ndim > 1 for value in values):
                for chunk, value in zip(cpu_chunks, values):
                    per_sample_values = np.empty(len(chunk), dtype=object)
                    for sample_idx in range(len(chunk)):
                        per_sample_values[sample_idx] = value[sample_idx]
                    chunk.non_tensor_batch[key] = per_sample_values

        merged = DataProto.concat(cpu_chunks)
        merged.meta_info = self._merge_chunk_meta_info(cpu_chunks)
        return merged

    def _collect_tree_rollout_chunk(self, batch_dict, generation_manager, gen_config, chunk_idx=None,
                                    accumulation_steps=1, timing_raw=None):
        """Collect one Tree rollout chunk and recompute its old-policy log probabilities."""
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        prompt_count = len(batch)
        ts_m = int(self.config.actor_rollout_ref.rollout.ts_m)
        ts_k = int(self.config.actor_rollout_ref.rollout.ts_k)

        # Repeat once per tree. The final output contains ts_k selected leaves per tree.
        batch = batch.repeat(repeat_times=ts_m, interleave=True)
        batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
        gen_batch = batch.pop(
            batch_keys=['input_ids', 'attention_mask', 'position_ids'],
            non_tensor_batch_keys=['uid', 'data_source', 'reward_model'],
        )
        gen_batch.batch['prompts'] = gen_batch.batch['input_ids']

        first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()
        chunk_timing = {}
        generation_manager.timing_raw = chunk_timing
        with _timer('gen', chunk_timing):
            final_gen_batch_output = generation_manager.run_llm_loop_tree_search(
                gen_batch=gen_batch,
                initial_input_ids=first_input_ids,
            )
        if timing_raw is not None:
            for key, value in chunk_timing.items():
                timing_raw[key] = timing_raw.get(key, 0.0) + value
        dump_chunk_idx = None if accumulation_steps == 1 else chunk_idx
        self._dump_tree_rollouts(final_gen_batch_output, generation_manager, chunk_idx=dump_chunk_idx)
        if self.config.trainer.get('teacher_rescue_enabled', False):
            rescue_stats = rescue_all_wrong(
                final_gen_batch_output, generation_manager,
                teacher_url=str(self.config.trainer.get('teacher_rescue_url', '')),
                group_size=ts_m * ts_k,
            )
            final_gen_batch_output.meta_info['teacher_rescue_stats'] = rescue_stats

        expected_trajectories = prompt_count * ts_m * ts_k
        if len(final_gen_batch_output) != expected_trajectories:
            raise RuntimeError(
                f'Tree rollout produced {len(final_gen_batch_output)} trajectories for '
                f'{prompt_count} prompts; expected {expected_trajectories} '
                f'({prompt_count}*ts_m={ts_m}*ts_k={ts_k})'
            )

        for key in final_gen_batch_output.batch.keys():
            # Token/id tensors must be integral, but rollout-side training
            # signals retain fractional values. Converting Self-OPD event
            # weights, advantages, and value gaps to long silently zeros them.
            if (key != 'token_level_scores'
                    and not key.startswith('branch_')
                    and not key.startswith('self_opd_')
                    and key != 'teacher_rescue_kd_mask'):
                final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

        for key in (
            'self_opd_event_weight',
            'self_opd_raw_advantage',
            'self_opd_value_gap',
        ):
            if key in final_gen_batch_output.batch and not torch.is_floating_point(
                    final_gen_batch_output.batch[key]):
                raise TypeError(f'{key} must remain floating point after rollout collection')

        with torch.no_grad():
            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
            final_gen_batch_output = final_gen_batch_output.union(output)

        # Repeat the original batch to align its prompt metadata with selected leaves.
        batch = batch.repeat(repeat_times=ts_k, interleave=True)
        batch = batch.union(final_gen_batch_output)
        if len(batch) != expected_trajectories:
            raise RuntimeError(
                f'aligned Tree batch has {len(batch)} trajectories; expected {expected_trajectories}'
            )

        print(
            f'[rollout accumulation] optimizer_step={self.global_steps} '
            f'chunk={chunk_idx or 1}/{accumulation_steps} '
            f'prompts={prompt_count} trajectories={len(batch)}'
        )
        return batch, prompt_count, len(batch)

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=False,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        """
        Validate each prompt with one or more agent trajectories.

        The plain validation agent shares the training search environment and
        length limits and does not use Tree expansion.  The default remains one
        deterministic sample; pass@n evaluation can opt into repeated sampled
        trajectories with trainer.val_num_samples and trainer.val_do_sample.
        """
        import torch
        reward_tensor_lst = []
        pure_em_lst = []
        data_source_lst = []
        val_num_samples = int(self.config.trainer.get('val_num_samples', 1))
        val_do_sample = bool(self.config.trainer.get('val_do_sample', val_num_samples > 1))
        requested_pass_k = tuple(dict.fromkeys(
            k for k in (1, 2, 4, 5, val_num_samples) if 1 <= k <= val_num_samples
        ))
        pass_at_k_values = {k: {} for k in requested_pass_k}

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url=self.config.retriever.url,
            topk=self.config.retriever.topk,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation=True,
        )

        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print('validation generation end')

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                _, reward_tensor = self.val_reward_fn(test_batch)

                reward_tensor_lst.append(reward_tensor)
                pure_em_lst.extend(pure_em_rows(test_batch, self.tokenizer))
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
        else:
            for batch_dict in self.val_dataloader:
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)

                # The tree-search vLLM adapter expects one generated row per
                # input row. Repeat prompts explicitly instead of rollout.n > 1;
                # interleaving keeps each prompt's samples contiguous.
                if val_num_samples > 1:
                    test_batch = test_batch.repeat(
                        repeat_times=val_num_samples, interleave=True
                    )

                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': val_do_sample,
                    'validate': True,
                }
                with _timer('step', timing_raw):
                    first_input_ids = test_gen_batch.batch['input_ids'][
                        :, -gen_config.max_start_length:
                    ].clone().long()
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                        )

                    if len(final_gen_batch_output) != len(test_batch):
                        raise RuntimeError(
                            f'Validation produced {len(final_gen_batch_output)} trajectories for '
                            f'{len(test_batch)} prompts; expected exactly one trajectory per prompt'
                        )

                    test_batch = test_batch.union(final_gen_batch_output)
                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()

                    _, reward_tensor = self.val_reward_fn(test_batch)

                    if val_num_samples > 1:
                        sample_rewards = reward_tensor.sum(-1).detach().cpu().view(
                            -1, val_num_samples
                        )
                        sample_sources = np.asarray(
                            test_batch.non_tensor_batch.get(
                                'data_source', ['unknown'] * reward_tensor.shape[0]
                            )
                        ).reshape(-1, val_num_samples)[:, 0]
                        correct_counts = (sample_rewards >= 1.0).sum(dim=1).tolist()
                        from math import comb
                        for source, correct_count in zip(sample_sources, correct_counts):
                            source = str(source)
                            for k in requested_pass_k:
                                score = 1.0 if val_num_samples - correct_count < k else (
                                    1.0 - comb(val_num_samples - correct_count, k)
                                    / comb(val_num_samples, k)
                                )
                                pass_at_k_values[k].setdefault(source, []).append(score)

                    reward_tensor_lst.append(reward_tensor)
                    pure_em_lst.extend(pure_em_rows(test_batch, self.tokenizer))
                    data_source_lst.append(test_batch.non_tensor_batch.get(
                        'data_source', ['unknown'] * reward_tensor.shape[0]
                    ))

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {'val/pure_em': float(np.mean(pure_em_lst)) if pure_em_lst else 0.0}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        if val_num_samples > 1:
            for k, source_values in pass_at_k_values.items():
                all_values = []
                for data_source, values in source_values.items():
                    metric_dict[f'val/pass@{k}/{data_source}'] = np.mean(values)
                    all_values.extend(values)
                if all_values:
                    metric_dict[f'val/pass@{k}/micro'] = np.mean(all_values)

        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True  
        elif self.config.algorithm.adv_estimator in ['grpo', 'tree', 'branch_credit', 'tree_inner', 'no_estimator', 'tree_per_token', 'tree_2norm', 'rloo']:
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _fit_with_rollout_accumulation(self, accumulation_steps: int):
        """Train with several small Tree rollouts followed by one optimizer update."""
        pprint(f'Using rollout accumulation: {accumulation_steps} chunks per optimizer step')

        logger = self.logger
        self.global_steps = int(self.config.trainer.get('resume_global_step', 0))
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # Keep the existing trainer's step numbering: validation is step 0 and
        # the first optimizer update is logged at step 1.
        self.global_steps += 1

        gen_config = GenerationTreeSearchConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url=self.config.retriever.url,
            topk=self.config.retriever.topk,
            m=self.config.actor_rollout_ref.rollout.ts_m,
            n=self.config.actor_rollout_ref.rollout.ts_n,
            l=self.config.actor_rollout_ref.rollout.ts_l,
            k=self.config.actor_rollout_ref.rollout.ts_k,
            reward_mode=self.config.actor_rollout_ref.rollout.reward_mode,
            expand_mode=self.config.actor_rollout_ref.rollout.expand_mode,
            expand_uncertainty_weight=float(
                self.config.actor_rollout_ref.rollout.get('expand_uncertainty_weight', 1.0)
            ),
            expand_search_weight=float(
                self.config.actor_rollout_ref.rollout.get('expand_search_weight', 1.0)
            ),
            expand_child_count_weight=float(
                self.config.actor_rollout_ref.rollout.get('expand_child_count_weight', 1.0)
            ),
            expand_depth_weight=float(
                self.config.actor_rollout_ref.rollout.get('expand_depth_weight', 1.0)
            ),
            expand_outcome_prior_root=float(
                self.config.actor_rollout_ref.rollout.get(
                    'expand_outcome_prior_root', 0.23255813953488372
                )
            ),
            expand_outcome_prior_pre_search=float(
                self.config.actor_rollout_ref.rollout.get(
                    'expand_outcome_prior_pre_search', 0.17880794701986755
                )
            ),
            expand_outcome_prior_after_search_depth1=float(
                self.config.actor_rollout_ref.rollout.get(
                    'expand_outcome_prior_after_search_depth1', 0.20754716981132076
                )
            ),
            expand_outcome_prior_after_search_depth2=float(
                self.config.actor_rollout_ref.rollout.get(
                    'expand_outcome_prior_after_search_depth2', 0.11363636363636363
                )
            ),
            expand_outcome_prior_after_search_depth3plus=float(
                self.config.actor_rollout_ref.rollout.get(
                    'expand_outcome_prior_after_search_depth3plus', 0.09090909090909091
                )
            ),
            expand_outcome_prior_concentration_power=float(
                self.config.actor_rollout_ref.rollout.get(
                    'expand_outcome_prior_concentration_power', 2.0
                )
            ),
            enable_branch_credit=self.config.algorithm.adv_estimator == 'branch_credit',
            branch_credit_normalization=self.config.algorithm.get(
                'branch_credit_normalization', 'sibling_std'
            ),
            branch_credit_clip=float(self.config.algorithm.get('branch_credit_clip', 3.0)),
            branch_credit_no_sibling=self.config.algorithm.get(
                'branch_credit_no_sibling', 'zero'
            ),
            branch_credit_value_mode=self.config.algorithm.get(
                'branch_credit_value_mode', 'reward'
            ),
            branch_credit_correctness_threshold=float(self.config.algorithm.get(
                'branch_credit_correctness_threshold', 0.8
            )),
            enable_self_opd=bool(self.config.actor_rollout_ref.actor.get(
                'self_opd_enabled', False
            )),
            self_opd_max_events=int(self.config.algorithm.get('self_opd_max_events', 3)),
            self_opd_max_teacher_length=int(self.config.algorithm.get(
                'self_opd_max_teacher_length', 1024
            )),
            self_opd_max_query_tokens=int(self.config.algorithm.get(
                'self_opd_max_query_tokens', 64
            )),
            self_opd_max_evidence_tokens=int(self.config.algorithm.get(
                'self_opd_max_evidence_tokens', 128
            )),
            self_opd_min_value_gap=float(self.config.algorithm.get(
                'self_opd_min_value_gap', 0.25
            )),
            self_opd_min_raw_advantage=float(self.config.algorithm.get(
                'self_opd_min_raw_advantage', 0.0
            )),
            self_opd_max_advantage_weight=float(self.config.algorithm.get(
                'self_opd_max_advantage_weight', 3.0
            )),
            self_opd_teacher_context=str(self.config.algorithm.get(
                'self_opd_teacher_context', 'hindsight'
            )),
            self_opd_event_selection=str(self.config.algorithm.get(
                'self_opd_event_selection', 'contrast'
            )),
            self_opd_analyzer_url=str(self.config.algorithm.get(
                'self_opd_analyzer_url', ''
            )),
        )
        generation_manager = LLMGenerationTreeSearchManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            reward_fn=self.reward_fn,
            config=gen_config,
        )

        train_iterator = iter(self.train_dataloader)
        data_epoch = 0
        ts_m = int(self.config.actor_rollout_ref.rollout.ts_m)
        ts_k = int(self.config.actor_rollout_ref.rollout.ts_k)
        configured_chunk_prompt_count = int(self.config.data.train_batch_size)

        while self.global_steps < self.total_training_steps:
            print(
                f'[########]epoch {data_epoch}, optimizer_step {self.global_steps}, '
                f'accumulation={accumulation_steps}'
            )
            metrics = {}
            timing_raw = {}
            rollout_chunks = []
            prompt_count = 0
            trajectory_count = 0

            with _timer('step', timing_raw):
                dapo_enabled = bool(self.config.trainer.get('dapo_dynamic_sampling', False))
                target_effective = int(self.config.trainer.get('dapo_target_effective_prompts', 24))
                max_chunks = int(self.config.trainer.get('dapo_max_chunks', 48))
                if dapo_enabled and (target_effective < 1 or max_chunks < 1):
                    raise ValueError('DAPO target_effective_prompts and max_chunks must both be positive')

                attempted_prompts = 0
                attempted_trajectories = 0
                all_correct_groups = 0
                all_wrong_groups = 0
                rescue_eligible = rescue_intervened = rescue_successes = rescue_failures = 0
                raw_correct_trajectories = 0
                chunk_idx = 0
                collection_limit = max_chunks if dapo_enabled else accumulation_steps
                while chunk_idx < collection_limit:
                    if dapo_enabled and prompt_count >= target_effective:
                        break
                    chunk_idx += 1
                    try:
                        batch_dict = next(train_iterator)
                    except StopIteration:
                        data_epoch += 1
                        train_iterator = iter(self.train_dataloader)
                        try:
                            batch_dict = next(train_iterator)
                        except StopIteration as exc:
                            raise RuntimeError('training dataloader yielded no batches') from exc

                    chunk, chunk_prompts, chunk_trajectories = self._collect_tree_rollout_chunk(
                        batch_dict=batch_dict,
                        generation_manager=generation_manager,
                        gen_config=gen_config,
                        chunk_idx=chunk_idx,
                        accumulation_steps=collection_limit,
                        timing_raw=timing_raw,
                    )
                    if chunk_prompts != configured_chunk_prompt_count:
                        raise RuntimeError(
                            f'rollout chunk {chunk_idx} has {chunk_prompts} prompts; '
                            f'expected configured train_batch_size={configured_chunk_prompt_count}'
                        )
                    attempted_prompts += chunk_prompts
                    attempted_trajectories += chunk_trajectories
                    rescue_stats = chunk.meta_info.get('teacher_rescue_stats', {})
                    raw_correct_trajectories += rescue_stats.get('raw_em_count', sum(
                        float(score) >= 0.8 for score in chunk.non_tensor_batch['original_score']
                    ))
                    rescue_eligible += int(rescue_stats.get('eligible', 0))
                    rescue_intervened += int(rescue_stats.get('intervened', 0))
                    rescue_successes += int(rescue_stats.get('successes', 0))
                    rescue_failures += int(rescue_stats.get('failed', 0))

                    if dapo_enabled:
                        remaining = target_effective - prompt_count
                        selected_rows, group_stats = select_dapo_effective_rows(
                            chunk.non_tensor_batch['uid'],
                            chunk.non_tensor_batch['original_score'],
                            group_size=ts_m * ts_k,
                            max_groups=remaining,
                            rescue_uids=chunk.meta_info.get('teacher_rescue_stats', {}).get('rescued_uids', ()),
                        )
                        all_correct_groups += group_stats['all_correct_groups']
                        all_wrong_groups += group_stats['all_wrong_groups'] + int(rescue_stats.get('promoted_groups', 0))
                        selected_groups = group_stats['effective_groups']
                        if not selected_rows:
                            continue
                        chunk = subset_dataproto_rows(chunk, selected_rows)
                        prompt_count += selected_groups
                        trajectory_count += len(chunk)
                    else:
                        prompt_count += chunk_prompts
                        trajectory_count += chunk_trajectories
                    chunk.meta_info.pop('teacher_rescue_stats', None)
                    rollout_chunks.append(chunk)

                if dapo_enabled and prompt_count < target_effective:
                    raise RuntimeError(
                        f'DAPO sampling found only {prompt_count}/{target_effective} effective groups '
                        f'after {chunk_idx} chunks; raise trainer.dapo_max_chunks or inspect rewards'
                    )

                expected_prompts = target_effective if dapo_enabled else configured_chunk_prompt_count * accumulation_steps
                expected_trajectories = expected_prompts * ts_m * ts_k
                if prompt_count != expected_prompts or trajectory_count != expected_trajectories:
                    raise RuntimeError(
                        f'rollout accumulation mismatch: prompts={prompt_count}/{expected_prompts}, '
                        f'trajectories={trajectory_count}/{expected_trajectories}'
                    )

                # All rollout tensors are CPU-resident before this operation.
                # Group UIDs are concatenated unchanged and advantages are
                # intentionally computed only after all chunks are present.
                batch = self._concat_rollout_chunks_cpu(rollout_chunks)
                if len(batch) != expected_trajectories:
                    raise RuntimeError(
                        f'concatenated rollout has {len(batch)} trajectories; '
                        f'expected {expected_trajectories}'
                    )
                if len(batch.non_tensor_batch.get('uid', [])) != expected_trajectories:
                    raise RuntimeError('concatenated rollout lost uid grouping metadata')

                metrics.update({
                    'rollout_accumulation/chunks': chunk_idx,
                    'rollout_accumulation/prompts': prompt_count,
                    'rollout_accumulation/trajectories': len(batch),
                })
                if dapo_enabled:
                    metrics.update({
                        'dapo/attempted_chunks': chunk_idx,
                        'dapo/attempted_prompts': attempted_prompts,
                        'dapo/attempted_trajectories': attempted_trajectories,
                        'dapo/effective_prompts': prompt_count,
                        'dapo/all_correct_groups': all_correct_groups,
                        'dapo/all_wrong_groups': all_wrong_groups,
                        'dapo/group_acceptance_rate': prompt_count / attempted_prompts,
                        'teacher_rescue/raw_rollout_em': raw_correct_trajectories / max(attempted_trajectories, 1),
                        'teacher_rescue/all_wrong_prompt_fraction': all_wrong_groups / max(attempted_prompts, 1),
                        'teacher_rescue/mixed_prompt_fraction': (attempted_prompts - all_wrong_groups - all_correct_groups) / max(attempted_prompts, 1),
                        'teacher_rescue/eligible_prompts': rescue_eligible,
                        'teacher_rescue/interventions': rescue_intervened,
                        'teacher_rescue/intervention_fraction': rescue_intervened / max(attempted_prompts, 1),
                        'teacher_rescue/successes': rescue_successes,
                        'teacher_rescue/success_fraction': rescue_successes / max(rescue_intervened, 1),
                        'teacher_rescue/failed_interventions': rescue_failures,
                    })
                print(
                    f'[rollout accumulation] optimizer_step={self.global_steps} '
                    f'chunks={chunk_idx} prompts={prompt_count} '
                    f'trajectories={len(batch)}; actor_update=pending'
                )

                self._balance_batch(batch, metrics=metrics)
                batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                for key in batch.batch.keys():
                    if (key not in {'old_log_probs', 'token_level_scores'}
                            and not key.startswith('branch_')
                            and not key.startswith('self_opd_')
                            and key != 'teacher_rescue_kd_mask'):
                        batch.batch[key] = batch.batch[key].long()

                if self.use_reference_policy:
                    with _timer('ref', timing_raw):
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                if self.use_critic:
                    with _timer('values', timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with _timer('adv', timing_raw):
                    batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        branch_credit_coef=self.config.algorithm.get('branch_credit_coef', 1.0),
                    )

                if self.use_critic:
                    with _timer('update_critic', timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    metrics.update(reduce_metrics(critic_output.meta_info['metrics']))

                actor_update_calls = 0
                if self.config.trainer.critic_warmup <= self.global_steps:
                    with _timer('update_actor', timing_raw):
                        if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                            batch, metrics = self._create_loss_mask(batch, metrics)
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_update_calls += 1
                    metrics.update(reduce_metrics(actor_output.meta_info['metrics']))

                if actor_update_calls > 1:
                    raise RuntimeError(
                        f'expected at most one actor update for optimizer step {self.global_steps}, '
                        f'got {actor_update_calls}'
                    )
                metrics['rollout_accumulation/actor_update_calls'] = actor_update_calls
                print(
                    f'[rollout accumulation] optimizer_step={self.global_steps} '
                    f'actor_update_calls={actor_update_calls}'
                )

                if (self.config.trainer.save_freq > 0 and
                        self.global_steps % self.config.trainer.save_freq == 0):
                    with _timer('save_checkpoint', timing_raw):
                        self._save_checkpoint()

                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                    with _timer('testing', timing_raw):
                        val_metrics: dict = self._validate()
                    metrics.update(val_metrics)

            metrics.update(compute_data_metrics(
                batch=batch,
                use_critic=self.use_critic,
                reward_mode=self.config.actor_rollout_ref.rollout.reward_mode,
            ))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            print(self.global_steps, metrics)
            logger.log(data=metrics, step=self.global_steps)

            self.global_steps += 1
            if self.global_steps >= self.total_training_steps:
                if self.val_reward_fn is not None and self.config.trainer.get('val_at_end', True):
                    val_metrics = self._validate()
                    pprint(f'Final validation metrics: {val_metrics}')
                    logger.log(data=val_metrics, step=self.global_steps)

                if self.config.trainer.get('save_at_end', True):
                    with _timer('save_checkpoint', timing_raw):
                        self._save_checkpoint()
                return

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        rollout_accumulation_steps = int(self.config.trainer.get('rollout_accumulation_steps', 1))
        if rollout_accumulation_steps < 1:
            raise ValueError(
                f'trainer.rollout_accumulation_steps must be >= 1, got {rollout_accumulation_steps}'
            )
        if rollout_accumulation_steps > 1:
            return self._fit_with_rollout_accumulation(rollout_accumulation_steps)

        pprint(f'Check config changes, kl_fix={self.config.algorithm.kl_fix}, use_kl_in_reward={self.config.algorithm.use_kl_in_reward}')

        logger = self.logger
        self.global_steps = 0
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        # Agent config preparation
        gen_config = GenerationTreeSearchConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
            m = self.config.actor_rollout_ref.rollout.ts_m,
            n = self.config.actor_rollout_ref.rollout.ts_n,
            l = self.config.actor_rollout_ref.rollout.ts_l,
            k = self.config.actor_rollout_ref.rollout.ts_k,
            reward_mode = self.config.actor_rollout_ref.rollout.reward_mode,
            expand_mode = self.config.actor_rollout_ref.rollout.expand_mode,
            expand_uncertainty_weight=float(
                self.config.actor_rollout_ref.rollout.get('expand_uncertainty_weight', 1.0)
            ),
            expand_search_weight=float(
                self.config.actor_rollout_ref.rollout.get('expand_search_weight', 1.0)
            ),
            expand_child_count_weight=float(
                self.config.actor_rollout_ref.rollout.get('expand_child_count_weight', 1.0)
            ),
            expand_depth_weight=float(
                self.config.actor_rollout_ref.rollout.get('expand_depth_weight', 1.0)
            ),
            expand_outcome_prior_root=float(
                self.config.actor_rollout_ref.rollout.get(
                    'expand_outcome_prior_root', 0.23255813953488372
                )
            ),
            expand_outcome_prior_pre_search=float(
                self.config.actor_rollout_ref.rollout.get(
                    'expand_outcome_prior_pre_search', 0.17880794701986755
                )
            ),
            expand_outcome_prior_after_search_depth1=float(
                self.config.actor_rollout_ref.rollout.get(
                    'expand_outcome_prior_after_search_depth1', 0.20754716981132076
                )
            ),
            expand_outcome_prior_after_search_depth2=float(
                self.config.actor_rollout_ref.rollout.get(
                    'expand_outcome_prior_after_search_depth2', 0.11363636363636363
                )
            ),
            expand_outcome_prior_after_search_depth3plus=float(
                self.config.actor_rollout_ref.rollout.get(
                    'expand_outcome_prior_after_search_depth3plus', 0.09090909090909091
                )
            ),
            expand_outcome_prior_concentration_power=float(
                self.config.actor_rollout_ref.rollout.get(
                    'expand_outcome_prior_concentration_power', 2.0
                )
            ),
            enable_branch_credit=self.config.algorithm.adv_estimator == 'branch_credit',
            branch_credit_normalization=self.config.algorithm.get(
                'branch_credit_normalization', 'sibling_std'
            ),
            branch_credit_clip=float(self.config.algorithm.get('branch_credit_clip', 3.0)),
            branch_credit_no_sibling=self.config.algorithm.get(
                'branch_credit_no_sibling', 'zero'
            ),
            branch_credit_value_mode=self.config.algorithm.get(
                'branch_credit_value_mode', 'reward'
            ),
            branch_credit_correctness_threshold=float(self.config.algorithm.get(
                'branch_credit_correctness_threshold', 0.8
            )),
            enable_self_opd=bool(self.config.actor_rollout_ref.actor.get(
                'self_opd_enabled', False
            )),
            self_opd_max_events=int(self.config.algorithm.get('self_opd_max_events', 3)),
            self_opd_max_teacher_length=int(self.config.algorithm.get(
                'self_opd_max_teacher_length', 1024
            )),
            self_opd_max_query_tokens=int(self.config.algorithm.get(
                'self_opd_max_query_tokens', 64
            )),
            self_opd_max_evidence_tokens=int(self.config.algorithm.get(
                'self_opd_max_evidence_tokens', 128
            )),
            self_opd_min_value_gap=float(self.config.algorithm.get(
                'self_opd_min_value_gap', 0.25
            )),
            self_opd_min_raw_advantage=float(self.config.algorithm.get(
                'self_opd_min_raw_advantage', 0.0
            )),
            self_opd_max_advantage_weight=float(self.config.algorithm.get(
                'self_opd_max_advantage_weight', 3.0
            )),
            self_opd_teacher_context=str(self.config.algorithm.get(
                'self_opd_teacher_context', 'hindsight'
            )),
            self_opd_event_selection=str(self.config.algorithm.get(
                'self_opd_event_selection', 'contrast'
            )),
            self_opd_analyzer_url=str(self.config.algorithm.get(
                'self_opd_analyzer_url', ''
            )),
        )

        generation_manager = LLMGenerationTreeSearchManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            reward_fn=self.reward_fn,
            config=gen_config,
        )

        # start training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'[########]epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                ## Repeat for tree search, [BS, ] -> [BS*m, ]
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.ts_m, interleave=True)
                batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                # pop those keys for generation
                gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['uid', 'data_source', 'reward_model'],
                    )
                gen_batch.batch['prompts'] = gen_batch.batch['input_ids']

                ####################
                # original code here

                with _timer('step', timing_raw):

                    # if not self.config.do_search:
                    #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    #     batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    #                                             dtype=object)
                    #     # repeat to align with repeated responses in rollout
                    #     batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    #     batch = batch.union(gen_batch_output)

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################
                # with _timer('step', timing_raw):
                    if True:
                        first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                        with _timer('gen', timing_raw):
                            generation_manager.timing_raw = timing_raw
                            final_gen_batch_output = generation_manager.run_llm_loop_tree_search(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                            )
                            self._dump_tree_rollouts(final_gen_batch_output, generation_manager)

                        # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        for key in final_gen_batch_output.batch.keys():
                            if (key != 'token_level_scores'
                                    and not key.startswith('branch_')
                                    and not key.startswith('self_opd_')):
                                final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        with torch.no_grad():
                            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                            final_gen_batch_output = final_gen_batch_output.union(output)

                        # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        #                                         dtype=object)
                        # batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                                            
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.ts_k, interleave=True)
                        batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    for key in batch.batch.keys():
                        if (key not in {'old_log_probs', 'token_level_scores'}
                                and not key.startswith('branch_')
                                and not key.startswith('self_opd_')):
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        # if self.use_rm:
                        #     # we first compute reward model score
                        #     reward_tensor = self.rm_wg.compute_rm_score(batch)
                        #     batch = batch.union(reward_tensor)

                        # # we combine with rule-based rm
                        # reward_tensor = self.reward_fn(batch)
                        # batch.batch['token_level_scores'] = reward_tensor

                        # # compute rewards. apply_kl_penalty if available
                        # # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                        # if self.config.algorithm.use_kl_in_reward:
                        #     batch, kl_metrics = apply_kl_penalty(batch,
                        #                                          kl_ctrl=self.kl_ctrl,
                        #                                          kl_penalty=self.config.algorithm.kl_penalty,
                        #                                          kl_fix=self.config.algorithm.kl_fix)
                        #     metrics.update(kl_metrics)
                        # else:
                        
                        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  branch_credit_coef=self.config.algorithm.get('branch_credit_coef', 1.0),
                                                  )

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # Save first so a long or interrupted validation cannot
                    # prevent the scheduled checkpoint from being persisted.
                    if (self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, reward_mode=self.config.actor_rollout_ref.rollout.reward_mode))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                print(self.global_steps, metrics)
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)

                    if self.config.trainer.get('save_at_end', True):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return
    
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics
