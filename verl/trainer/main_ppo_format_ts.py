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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em, qa_em_format, qa_f1_format
from verl.trainer.ppo.ray_trainer_ts import RayPPOTrainer
import re
import numpy as np
import json
import os
import urllib.request

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'web_questions', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'strategyqa']:
        return qa_em_format.compute_score_em
    else:
        return qa_f1_format.compute_score_f1


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, structure_format_score=0., final_format_score=0., retrieval_score=0., format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.structure_format_score = structure_format_score
        self.final_format_score = final_format_score
        self.retrieval_score = retrieval_score
        self.semantic_reward_url = os.environ.get('SEMANTIC_REWARD_URL', '').rstrip('/')

    def _semantic_judgments(self, items):
        if not self.semantic_reward_url or not items:
            return [False] * len(items)
        values = []
        for start in range(0, len(items), 64):
            batch = items[start:start + 64]
            request = urllib.request.Request(
                self.semantic_reward_url + '/semantic_judge',
                data=json.dumps({'items': batch}, ensure_ascii=False).encode(),
                headers={'Content-Type': 'application/json'}, method='POST')
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    payload = json.load(response)
                judgments = payload['judgments']
                if len(judgments) != len(batch):
                    raise ValueError('semantic judge returned wrong batch size')
                values.extend(bool(item.get('equivalent')) for item in judgments)
            except Exception as error:
                print(f'[semantic reward] strict-EM fallback: {type(error).__name__}: {error}', flush=True)
                values.extend([False] * len(batch))
        return values

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        all_scores = []

        already_print_data_sources = {}
        semantic_candidates = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, 
                                     structure_format_score=self.structure_format_score, 
                                     final_format_score=self.final_format_score, 
                                     retrieval_score=self.retrieval_score,
                                     format_score=self.format_score,
                                     data_source=data_source)

            reward_tensor[i, valid_response_length - 1] = score
            all_scores.append(score)

            if self.semantic_reward_url and data_source in [
                    'nq', 'triviaqa', 'popqa', 'web_questions', 'hotpotqa',
                    '2wikimultihopqa', 'musique', 'bamboogle', 'strategyqa']:
                answer = qa_em_format.extract_solution(sequences_str)
                targets = ground_truth.get('target', [])
                targets = [targets] if isinstance(targets, str) else [str(value) for value in targets]
                if answer is not None and not qa_em_format.em_check(answer, targets):
                    question_match = re.search(r'Question:\s*(.*?)<\|im_end\|>', sequences_str, flags=re.S)
                    question = question_match.group(1).strip() if question_match else sequences_str[:1800]
                    semantic_candidates.append((i, valid_response_length, sequences_str, {
                        'question': question, 'reference_answers': targets, 'model_answer': answer,
                    }))

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        accepted = self._semantic_judgments([item for *_meta, item in semantic_candidates])
        for (i, valid_response_length, sequences_str, _item), equivalent in zip(semantic_candidates, accepted):
            if not equivalent:
                continue
            valid_format = qa_em_format.is_valid_sequence(sequences_str)[0]
            semantic_score = 1.0 if valid_format else 1.0 - self.structure_format_score
            all_scores[i] = semantic_score
            reward_tensor[i].zero_()
            reward_tensor[i, valid_response_length - 1] = semantic_score
        if semantic_candidates:
            print(f'[semantic reward] accepted={sum(accepted)}/{len(accepted)}', flush=True)
        return all_scores, reward_tensor


import os

import ray
import hydra


def _optional_positive_int_env(name):
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == '':
        return None
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f'{name} must be a positive integer, got {raw_value!r}') from exc
    if value <= 0:
        raise ValueError(f'{name} must be a positive integer, got {raw_value!r}')
    return value


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        print("Ray is not initialized! run ray.init()...")
        ray_init_kwargs = {}
        ray_num_cpus = _optional_positive_int_env('TREE_GRPO_RAY_NUM_CPUS')
        ray_object_store_mib = _optional_positive_int_env(
            'TREE_GRPO_RAY_OBJECT_STORE_MEMORY_MIB'
        )
        if ray_num_cpus is not None:
            ray_init_kwargs['num_cpus'] = ray_num_cpus
        if ray_object_store_mib is not None:
            ray_init_kwargs['object_store_memory'] = ray_object_store_mib * 1024 * 1024
        print(f'Ray resource overrides: {ray_init_kwargs or "defaults"}')
        ray.init(
            runtime_env={
                'env_vars': {
                    'TOKENIZERS_PARALLELISM': 'true',
                    'NCCL_DEBUG': 'WARN',
                }
            },
            **ray_init_kwargs,
        )
    print("Ray already initialized. Get remote ...")
    ray.get(main_task.remote(config))


# Do not retry the controller task inside a damaged Ray session.  Its named
# placement group survives a memory-kill long enough to collide with an
# in-session retry; the detached service performs a clean process restart.
@ray.remote(max_retries=0)
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer_ts import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
    }
    if not config.trainer.get("val_only", False):
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }
    if not config.trainer.get("val_only", False):
        mapping[Role.RefPolicy] = global_pool_id

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, 
                              structure_format_score=config.reward_model.structure_format_score, 
                              final_format_score=config.reward_model.final_format_score,
                              retrieval_score=config.reward_model.retrieval_score)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    print(f"=================init_workers================")
    trainer.init_workers()
    init_actor_checkpoint = config.trainer.get('init_actor_checkpoint', None)
    if init_actor_checkpoint:
        resume_actor_state = bool(config.trainer.get('resume_actor_state', False))
        load_mode = 'full-state resume' if resume_actor_state else 'actor weights only'
        print(
            f"=================load_actor_checkpoint: {init_actor_checkpoint} "
            f"({load_mode})================"
        )
        trainer.actor_rollout_wg.load_checkpoint(
            init_actor_checkpoint,
            del_local_after_load=False,
            load_optimizer=resume_actor_state,
            load_scheduler=resume_actor_state,
            load_rng=resume_actor_state,
        )
    print(f"=================fit================")
    trainer.fit()


if __name__ == '__main__':
    main()
