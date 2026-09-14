import inspect

import torch
from torch.nn import functional as F

from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer, compute_masked_ce_chunked


def test_validation_releases_gradients_only_at_validation_boundary():
    fit_source = inspect.getsource(FSDPSFTTrainer.fit)
    release_source = inspect.getsource(FSDPSFTTrainer._release_optimizer_grads_before_validation)

    training_step_pos = fit_source.index('metric = self.training_step(data)')
    release_pos = fit_source.index('self._release_optimizer_grads_before_validation()')
    validation_pos = fit_source.index('_run_validation(global_step, epoch, self.tracking)')

    assert training_step_pos < release_pos < validation_pos
    assert fit_source.count('self._release_optimizer_grads_before_validation()') == 1
    assert 'self.optimizer.zero_grad(set_to_none=True)' in release_source
    assert 'if should_validate or should_stop:' in fit_source
    boundary = fit_source[fit_source.index('if should_validate or should_stop:'):validation_pos]
    assert 'self._release_optimizer_grads_before_validation()' in boundary
    assert fit_source.count('self.optimizer.zero_grad') == 0


def test_chunked_masked_ce_matches_one_shot_for_non_divisible_chunk():
    torch.manual_seed(7)
    logits = torch.randn(13, 17, dtype=torch.float32)
    labels = torch.randint(0, logits.shape[-1], (13,), dtype=torch.long)
    loss_mask = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0], dtype=torch.float32)

    expected = torch.sum(F.cross_entropy(logits, labels, reduction='none') * loss_mask)
    actual, valid_tokens = compute_masked_ce_chunked(logits, labels, loss_mask, chunk_size=5)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(valid_tokens, loss_mask.sum())


def test_chunked_masked_ce_handles_empty_and_zero_masks():
    logits = torch.randn(4, 9, dtype=torch.float32)
    labels = torch.zeros(4, dtype=torch.long)
    zero_mask = torch.zeros(4, dtype=torch.float32)

    actual, valid_tokens = compute_masked_ce_chunked(logits, labels, zero_mask, chunk_size=3)
    assert actual.item() == 0.0
    assert valid_tokens.item() == 0.0

    empty_logits = torch.empty(0, 9, dtype=torch.float32)
    empty_labels = torch.empty(0, dtype=torch.long)
    empty_mask = torch.empty(0, dtype=torch.float32)
    actual, valid_tokens = compute_masked_ce_chunked(empty_logits, empty_labels, empty_mask, chunk_size=3)
    assert actual.item() == 0.0
    assert valid_tokens.item() == 0.0
