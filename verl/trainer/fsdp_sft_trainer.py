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
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import logging
import importlib.util
import math
import re
import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, AutoConfig
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from tensordict import TensorDict
from torch.utils.data import DataLoader, DistributedSampler

from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.dataset import SFTDataset
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.tracking import Tracking

from torch.distributed.device_mesh import DeviceMesh

import verl.utils.hdfs_io as hdfs_io
from verl.utils.debug import log_gpu_memory_usage

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))

VALIDATION_LOSS_CHUNK_SIZE = 512


def extract_step(path):
    match = re.search(r'global_step_(\d+)', path)
    if match:
        return int(match.group(1))
    return None


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import ListConfig, DictConfig
    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


def compute_masked_ce_chunked(flattened_logits,
                              flattened_labels,
                              loss_mask,
                              chunk_size=VALIDATION_LOSS_CHUNK_SIZE):
    """Compute a masked token CE sum in bounded chunks.

    The inputs are already flattened to ``[num_tokens, vocab_size]`` and
    ``[num_tokens]``.  Only the current chunk's unreduced CE tensor is kept,
    so validation does not materialize an unreduced loss for the whole batch.
    The returned token count uses the same mask values and dtype as the
    original one-shot implementation.
    """
    if flattened_logits.ndim != 2:
        raise ValueError('flattened_logits must have shape [num_tokens, vocab_size]')
    if flattened_labels.ndim != 1 or loss_mask.ndim != 1:
        raise ValueError('flattened_labels and loss_mask must have shape [num_tokens]')
    if flattened_logits.shape[0] != flattened_labels.shape[0] or flattened_logits.shape[0] != loss_mask.shape[0]:
        raise ValueError('logits, labels, and loss_mask must contain the same number of tokens')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')

    device = flattened_logits.device
    # Cross entropy requires integral targets on the logits device.  Moving
    # only the compact labels/mask here also makes the helper independently
    # usable in CPU tests and keeps validation's device semantics explicit.
    flattened_labels = flattened_labels.to(device=device, dtype=torch.long)
    loss_mask = loss_mask.to(device=device)
    valid_token_count = torch.sum(loss_mask)

    # A completely empty target or an all-zero mask has a well-defined zero
    # masked loss, while the original division would otherwise produce NaN.
    if flattened_logits.shape[0] == 0 or torch.count_nonzero(loss_mask) == 0:
        return flattened_logits.new_zeros(()), valid_token_count

    loss_sum = None
    for start in range(0, flattened_logits.shape[0], chunk_size):
        end = min(start + chunk_size, flattened_logits.shape[0])
        chunk_loss = F.cross_entropy(flattened_logits[start:end],
                                     flattened_labels[start:end],
                                     reduction='none')
        chunk_masked_sum = torch.sum(chunk_loss * loss_mask[start:end])
        loss_sum = chunk_masked_sum if loss_sum is None else loss_sum + chunk_masked_sum

    return loss_sum, valid_token_count


class FSDPSFTTrainer(object):

    def __init__(self, config, device_mesh: DeviceMesh):
        self.config = config
        self.device_mesh = device_mesh
        self.rank = self.device_mesh.get_rank()
        self.resume_path = self.config.trainer.get('resume_path', None)
        configured_model_path = self.config.model.get(
            'partial_pretrain', self.config.model.get('path', None))
        self.model_source = self._resolve_model_source(configured_model_path)
        # build tokenizer first
        local_model_path = copy_local_path_from_hdfs(src=self.model_source, verbose=True)
        from verl.utils import hf_tokenizer
        self.tokenizer = hf_tokenizer(local_model_path,
                                      trust_remote_code=self.config.model.get('trust_remote_code', False))
        if self.config.data.get('chat_template', None) is not None:
            raise ValueError('Apply Chat template from config is not supported yet.')

        # normalize dp size
        self._normalize_config_bsz()

        self._build_dataloader()
        # build model
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.rank == 0:
            print(self.config)

    def _resolve_model_source(self, configured_model_path):
        """Use a resume checkpoint as the HF model source when it is a model directory."""
        if self.resume_path:
            resume_path = copy_local_path_from_hdfs(src=self.resume_path, verbose=True)
            self.resume_path = resume_path
            if os.path.isdir(resume_path):
                return resume_path
            logger.warning('resume_path=%s is not an HF model directory; using the configured model source.',
                           resume_path)
        return configured_model_path

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size()
        if self.rank == 0:
            print(f'Normalize batch size by dp {dp_size}')

        assert self.config.data.train_batch_size % dp_size == 0
        assert self.config.data.micro_batch_size % dp_size == 0

        self.config.data.train_batch_size //= dp_size
        self.config.data.micro_batch_size //= dp_size

    def _build_dataloader(self):
        config = self.config
        # build dataset
        self.train_dataset = SFTDataset(parquet_files=config.data.train_files,
                                        tokenizer=self.tokenizer,
                                        prompt_key=config.data.prompt_key,
                                        prompt_dict_keys=config.data.get('prompt_dict_keys', None),
                                        response_key=config.data.response_key,
                                        response_dict_keys=config.data.get('response_dict_keys', None),
                                        max_length=config.data.max_length,
                                        truncation=config.data.truncation)
        self.val_dataset = SFTDataset(parquet_files=config.data.val_files,
                                      tokenizer=self.tokenizer,
                                      prompt_key=config.data.prompt_key,
                                      prompt_dict_keys=config.data.get('prompt_dict_keys', None),
                                      response_key=config.data.response_key,
                                      response_dict_keys=config.data.get('response_dict_keys', None),
                                      max_length=config.data.max_length,
                                      truncation=config.data.truncation)

        # build dataloader
        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()
        self.train_sampler = DistributedSampler(self.train_dataset,
                                                shuffle=True,
                                                num_replicas=world_size,
                                                rank=rank,
                                                drop_last=True)
        num_workers = int(config.data.get('num_workers', 8))
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=config.data.train_batch_size,
                                           sampler=self.train_sampler,
                                           num_workers=num_workers,
                                           pin_memory=True,
                                           drop_last=True)

        self.val_sampler = DistributedSampler(self.val_dataset,
                                              shuffle=True,
                                              num_replicas=world_size,
                                              rank=rank,
                                              drop_last=True)
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=config.data.micro_batch_size,
                                         sampler=self.val_sampler,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         drop_last=True)

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_local_path_from_hdfs(src=self.model_source, verbose=True)

        if self.config.model.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib
            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage('Before model allocation', logger=logger)

        trust_remote_code = self.config.model.get('trust_remote_code', False)
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)

        # This may be very large
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings)

        attention_implementation = self.config.model.get(
            'attn_implementation',
            self.config.model.get('attention_implementation', 'flash_attention_2'))
        if attention_implementation == 'flash_attention_2' and importlib.util.find_spec('flash_attn') is None:
            logger.warning('flash_attn is unavailable; falling back to SDPA attention.')
            attention_implementation = 'sdpa'

        with init_context():
            try:
                self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    config=config,
                    torch_dtype=torch.float32,
                    attn_implementation=attention_implementation,
                    trust_remote_code=trust_remote_code)
            except (ImportError, OSError, ValueError) as exc:
                if attention_implementation != 'flash_attention_2':
                    raise
                logger.warning('FlashAttention initialization failed (%s); falling back to SDPA.', exc)
                self.model = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    config=config,
                    torch_dtype=torch.float32,
                    attn_implementation='sdpa',
                    trust_remote_code=trust_remote_code)

            if self.config.model.get('lora_rank', 0) > 0:
                try:
                    from peft import LoraConfig, TaskType, get_peft_model
                except ImportError as exc:
                    raise ImportError('PEFT is required only when model.lora_rank > 0.') from exc
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    'task_type': TaskType.CAUSAL_LM,
                    'r': self.config.model.lora_rank,
                    'lora_alpha': self.config.model.lora_alpha,
                    'target_modules': convert_to_regular_types(self.config.model.target_modules),
                    'bias': "none"
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.get('enable_gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        log_gpu_memory_usage('After model allocation', logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16,
                                         reduce_dtype=torch.float32,
                                         buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(self.model,
                                                config=self.config.model.fsdp_config.wrap_policy,
                                                is_lora=self.config.model.get('lora_rank', 0) > 0)
        if self.rank == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        self.fsdp_model = FSDP(module=self.model,
                               auto_wrap_policy=auto_wrap_policy,
                               param_init_fn=init_fn,
                               sharding_strategy=ShardingStrategy.FULL_SHARD,
                               mixed_precision=mixed_precision,
                               device_mesh=self.device_mesh,
                               sync_module_states=True,
                               device_id=torch.cuda.current_device(),
                               cpu_offload=cpu_offload,
                               use_orig_params=False)

        log_gpu_memory_usage('After FSDP wrapping', logger=logger)

        self.optimizer = optim.AdamW(self.fsdp_model.parameters(),
                                     lr=self.config.optim.lr,
                                     betas=self.config.optim.betas,
                                     weight_decay=self.config.optim.weight_decay)

        log_gpu_memory_usage('After initialize optimizer', logger=logger)

        steps_per_epoch = len(self.train_dataloader)
        total_steps = steps_per_epoch * self.config.trainer.total_epochs
        configured_total_steps = self.config.trainer.get('total_training_steps', None)
        if configured_total_steps is not None:
            total_steps = int(configured_total_steps)
        if total_steps <= 0:
            raise ValueError('The SFT schedule requires at least one optimizer step.')

        if self.rank == 0:
            print(
                f'Number of steps/epoch {steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {total_steps}'
            )

        warmup_ratio = self.config.optim.get(
            'warmup_steps_ratio', self.config.optim.get('warmup_ratio', 0.0))
        num_warmup_steps = int(total_steps * warmup_ratio)

        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=total_steps)
        self.resume_state = self._load_resume_state()

    def _load_resume_state(self):
        if not self.resume_path:
            return None

        state_path = os.path.join(self.resume_path, 'trainer_state.pt')
        if not os.path.exists(state_path):
            logger.warning('No trainer_state.pt found under resume_path=%s; starting progress from zero.',
                           self.resume_path)
            return None

        state_box = [None]
        if self.rank == 0:
            # This is a locally generated trainer state. PyTorch 2.6 defaults
            # to weights_only=True, which rejects OmegaConf objects stored in
            # the full resume state.
            state_box[0] = torch.load(state_path, map_location='cpu', weights_only=False)
        torch.distributed.broadcast_object_list(state_box, src=0)
        state = state_box[0]
        optimizer_state = state.get('optimizer')
        if optimizer_state is not None:
            try:
                optimizer_state = FSDP.optim_state_dict_to_load(
                    self.fsdp_model, self.optimizer, optimizer_state)
            except (AttributeError, TypeError):
                pass
            self.optimizer.load_state_dict(optimizer_state)
        if state.get('scheduler') is not None:
            self.lr_scheduler.load_state_dict(state['scheduler'])
        logger.info('Resumed trainer progress from %s: step=%s epoch=%s batch=%s',
                    state_path, state.get('global_step', 0), state.get('epoch', 0),
                    state.get('batch_index', state.get('batch_idx', 0)))
        return state

    def _compute_loss(self, batch):
        loss_mask = batch.pop('loss_mask')[:, :-1].reshape(-1).cuda()
        labels = batch['input_ids'][:, 1:].cuda()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.fsdp_model(input_ids=batch['input_ids'],
                                     attention_mask=batch['attention_mask'],
                                     position_ids=batch['position_ids'],
                                     use_cache=False)  # prevent model thinks it it generating

        logits = output.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels.contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss * loss_mask

        valid_token_this_rank = torch.sum(loss_mask)

        if self.config.data.get('balance_dp_token', self.config.data.get('balance', False)):
            torch.distributed.all_reduce(valid_token_this_rank)  # becomes total valid tokens in all ranks
            dp_size = torch.distributed.get_world_size()
        else:
            dp_size = 1

        loss = torch.sum(loss) / valid_token_this_rank * dp_size  # possible bugs here for dp
        return loss

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage('Before optimizer zero_grad', logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage('After optimizer zero_grad', logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss(batch=micro_batch) / n_micro_batches
            loss.backward()
            step_loss += loss.item()

        self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)

        log_gpu_memory_usage('Before optimizer step', logger=logger)

        self.optimizer.step()

        log_gpu_memory_usage('After optimizer step', logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage('After offload weights', logger=logger)

        step_loss = torch.tensor(step_loss).cuda()
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        return {'train/loss': step_loss.detach().item(), 'train/lr': lr}

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss_mask = batch.pop('loss_mask')[:, :-1].reshape(-1).cuda()
            labels = batch['input_ids'][:, 1:].cuda()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.fsdp_model(input_ids=batch['input_ids'],
                                         attention_mask=batch['attention_mask'],
                                         position_ids=batch['position_ids'],
                                         use_cache=False)

            # Keep the existing forward path, but bound the unreduced CE
            # workspace to a small token chunk during validation.  Release
            # the model output before entering the chunk loop where possible.
            shift_logits = output.logits[..., :-1, :].contiguous().view(-1, self.model.config.vocab_size)
            shift_labels = labels.contiguous().view(-1).to(shift_logits.device)
            del output
            loss_sum, valid_token_this_rank = compute_masked_ce_chunked(
                shift_logits,
                shift_labels,
                loss_mask,
                chunk_size=VALIDATION_LOSS_CHUNK_SIZE)

            if self.config.data.get('balance_dp_token', self.config.data.get('balance', False)):
                torch.distributed.all_reduce(valid_token_this_rank)  # becomes total valid tokens in all ranks
                dp_size = torch.distributed.get_world_size()
            else:
                dp_size = 1

            if valid_token_this_rank.item() == 0:
                loss = loss_sum.new_zeros(())
            else:
                loss = loss_sum / valid_token_this_rank * dp_size
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        return loss

    def _validation_frequency(self):
        return int(self.config.trainer.get('validation_freq', len(self.train_dataloader)))

    def _save_frequency(self):
        return int(self.config.trainer.get('save_freq', len(self.train_dataloader)))

    def _log_frequency(self):
        return int(self.config.trainer.get('log_freq', 1))

    def _release_optimizer_grads_before_validation(self):
        """Release gradients after the completed update before validation.

        ``training_step`` has already called ``optimizer.step`` when this is
        used.  The next training step clears gradients again, so setting them
        to ``None`` here changes only the validation-time memory footprint.
        """
        self.optimizer.zero_grad(set_to_none=True)

    def _run_validation(self, step, epoch, tracking):
        self.val_sampler.set_epoch(epoch=epoch)
        val_losses = []
        for data in self.val_dataloader:
            data = TensorDict(data, batch_size=self.config.data.micro_batch_size).cuda()
            val_losses.append(self.validation_step(data))
        if val_losses:
            avg_val_loss = torch.mean(torch.stack(val_losses))
        else:
            avg_val_loss = torch.tensor(float('nan'), device='cuda')
        val_loss = avg_val_loss.detach().item()
        is_best = not math.isnan(val_loss) and (
            self.best_val_loss is None or val_loss < self.best_val_loss)
        if is_best:
            self.best_val_loss = val_loss
        if self.rank == 0:
            tracking.log(data={
                'val/loss': val_loss,
                'train/global_step': step,
                'train/epoch': epoch,
                'train/best_val_loss': self.best_val_loss if self.best_val_loss is not None else float('nan'),
            }, step=step)
        torch.distributed.barrier()
        return val_loss, is_best

    def _full_state_dicts(self):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.fsdp_model.state_dict()
            try:
                optimizer_state = FSDP.full_optim_state_dict(
                    self.fsdp_model, self.optimizer, rank0_only=True)
            except AttributeError:
                optimizer_state = FSDP.optim_state_dict(self.fsdp_model, self.optimizer)
        return state_dict, optimizer_state

    def _write_hf_checkpoint(self, path, state_dict, trainer_state):
        if self.rank != 0:
            return
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path, state_dict=state_dict)
        self.tokenizer.save_pretrained(path)
        torch.save(trainer_state, os.path.join(path, 'trainer_state.pt'))
        if self.config.trainer.get('default_hdfs_dir', None):
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

    def _update_checkpoint_alias(self, alias_name, target_path):
        """Atomically point best/last at one physical periodic checkpoint."""
        if self.rank != 0:
            return
        base_dir = self.config.trainer.default_local_dir
        alias_path = os.path.join(base_dir, alias_name)
        temporary_alias = f'{alias_path}.tmp'
        if os.path.lexists(temporary_alias):
            os.unlink(temporary_alias)
        if os.path.lexists(alias_path) and not os.path.islink(alias_path):
            raise FileExistsError(
                f'Cannot replace legacy checkpoint directory {alias_path!r} with a symlink. '
                'Move or remove that directory before continuing.')
        relative_target = os.path.relpath(target_path, start=base_dir)
        os.symlink(relative_target, temporary_alias)
        os.replace(temporary_alias, alias_path)

    def save_checkpoint(self, step, epoch, batch_index, save_periodic=True, save_best=False):
        state_dict, optimizer_state = self._full_state_dicts()
        trainer_state = {
            'optimizer': optimizer_state,
            'scheduler': self.lr_scheduler.state_dict(),
            'global_step': step,
            'epoch': epoch,
            'batch_idx': batch_index,
            'batch_index': batch_index,
            'best_val_loss': self.best_val_loss,
        }
        base_dir = self.config.trainer.default_local_dir
        # Every save has one physical, resumable checkpoint. best/last are
        # lightweight aliases so model and optimizer state are not triplicated.
        checkpoint_path = os.path.join(base_dir, f'global_step_{step}')
        if self.rank == 0:
            self._write_hf_checkpoint(checkpoint_path, state_dict, trainer_state)
            self._update_checkpoint_alias('last', checkpoint_path)
            if save_best:
                self._update_checkpoint_alias('best', checkpoint_path)
        torch.distributed.barrier()

    def _finish_tracking(self):
        if self.rank != 0 or self.tracking is None:
            return
        for backend_logger in list(self.tracking.logger.values()):
            finish = getattr(backend_logger, 'finish', None)
            if callable(finish):
                finish()
        # Prevent Tracking.__del__ from finishing the W&B run a second time.
        self.tracking.logger.clear()
        self.tracking = None

    def fit(self):
        self.tracking = None
        if self.rank == 0:
            self.tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=convert_to_regular_types(self.config))

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        configured_total_steps = self.config.trainer.get('total_training_steps', None)
        if configured_total_steps is not None:
            total_training_steps = configured_total_steps
        self.total_training_steps = total_training_steps
        validation_freq = self._validation_frequency()
        save_freq = self._save_frequency()
        log_freq = self._log_frequency()

        resume = self.resume_state or {}
        global_step = int(resume.get('global_step', 0))
        start_epoch = int(resume.get('epoch', 0))
        start_batch = int(resume.get('batch_index', resume.get('batch_idx', 0)))
        self.best_val_loss = resume.get('best_val_loss', None)
        if self.rank == 0:
            print(f'Total training steps: {self.total_training_steps}')
            self.tracking.log(data={
                'data/train_size': len(self.train_dataset),
                'data/val_size': len(self.val_dataset),
                'data/max_length': int(self.config.data.max_length),
                'data/global_train_batch_size': (
                    int(self.config.data.train_batch_size) * self.device_mesh.size()),
            }, step=global_step)

        try:
            if self.config.trainer.get('validate_before_training',
                                       self.config.trainer.get('val_before_train', False)):
                _, is_best = self._run_validation(global_step, start_epoch, self.tracking)
                if is_best:
                    self.save_checkpoint(global_step, start_epoch, start_batch,
                                         save_periodic=False, save_best=True)

            stop_training = global_step >= self.total_training_steps
            if stop_training:
                self.save_checkpoint(global_step, start_epoch, start_batch,
                                     save_periodic=False, save_best=False)
                return
            for epoch in range(start_epoch, self.config.trainer.total_epochs):
                self.train_sampler.set_epoch(epoch=epoch)
                first_batch = start_batch if epoch == start_epoch else 0
                for batch_index, data in enumerate(self.train_dataloader):
                    if batch_index < first_batch:
                        continue
                    data = TensorDict(data, batch_size=self.config.data.train_batch_size).cuda()
                    metric = self.training_step(data)
                    global_step += 1
                    metric.update({
                        'train/global_step': global_step,
                        'train/epoch': epoch,
                        'train/log_freq': log_freq,
                        'train/best_val_loss': self.best_val_loss if self.best_val_loss is not None else float('nan'),
                    })
                    should_log = log_freq > 0 and global_step % log_freq == 0
                    if self.rank == 0 and (should_log or global_step >= self.total_training_steps):
                        self.tracking.log(data=metric, step=global_step)

                    should_validate = validation_freq > 0 and global_step % validation_freq == 0
                    should_stop = global_step >= self.total_training_steps
                    is_best = False
                    if should_validate or should_stop:
                        self._release_optimizer_grads_before_validation()
                        _, is_best = self._run_validation(global_step, epoch, self.tracking)
                    should_save = save_freq > 0 and global_step % save_freq == 0
                    if should_save or should_stop:
                        self.save_checkpoint(global_step, epoch, batch_index + 1,
                                             save_periodic=True, save_best=is_best)
                    elif is_best:
                        self.save_checkpoint(global_step, epoch, batch_index + 1,
                                             save_periodic=False, save_best=True)
                    if should_stop:
                        stop_training = True
                        break
                if stop_training:
                    break
                start_batch = 0
            if not stop_training:
                # A non-positive total_training_steps means train to the configured epoch limit.
                self.save_checkpoint(global_step, self.config.trainer.total_epochs, 0,
                                     save_periodic=True, save_best=False)
        finally:
            self._finish_tracking()


from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
import hydra

from torch.distributed.device_mesh import init_device_mesh

from verl.utils.distributed import initialize_global_process_group


@hydra.main(config_path='config', config_name='sft_trainer', version_base=None)
def main(config):
    local_rank, rank, world_size = initialize_global_process_group()
    try:
        device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(world_size,), mesh_dim_names=('dp',))
        trainer = FSDPSFTTrainer(config=config, device_mesh=device_mesh)
        trainer.fit()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
