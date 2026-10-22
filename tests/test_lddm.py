from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from hydra import compose, initialize_config_dir

import algo
import samplers


class Tokenizer:
  vocab_size = 11
  bos_token_id = 0

  def __len__(self):
    return self.vocab_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason='DiT requires CUDA')
@pytest.mark.parametrize('name, trainer', [('lddm_u', algo.LDDM_U),
                                         ('lddm_m', algo.LDDM_M)])
@pytest.mark.parametrize('amp', [False, True])
def test_preliminary_forward_has_no_graph(name, trainer, amp):
  directory = str(Path(__file__).resolve().parents[1] / 'configs')
  with initialize_config_dir(version_base=None, config_dir=directory):
    cfg = compose(config_name='config', overrides=[
      f'algo={name}', 'algo.self_cond_rate=1.0', 'model=small',
      'model.length=8', 'model.hidden_size=32', 'model.cond_dim=16',
      'model.n_blocks=1', 'model.n_heads=4', 'model.dropout=0.0',
      'trainer.devices=1', 'trainer.num_nodes=1', 'loader.global_batch_size=2'])
  torch.manual_seed(17)
  assert 'loophole' not in cfg.algo
  model = trainer(cfg, Tokenizer()).cuda().eval()
  # Nonzero projections make an accidental cross-pass gradient observable.
  torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.1)
  torch.nn.init.ones_(model.backbone.latent_ln.weight)
  tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8],
                         [2, 3, 4, 5, 6, 7, 8, 9]], device='cuda')
  hidden_states = []
  logits = []

  def record_projection(module, inputs, output):
    del module
    hidden_states.append(inputs[0])
    logits.append(output)
    if inputs[0].requires_grad:
      inputs[0].retain_grad()
    if output.requires_grad:
      output.retain_grad()

  hook = model.backbone.output_layer.register_forward_hook(record_projection)
  try:
    with patch.object(model, 'forward', wraps=model.forward) as forward:
      with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
        loss, _ = model.nll(tokens, None, None, train_mode=True)
    assert forward.call_count == 2
    assert forward.call_args_list[0].kwargs['context'].carry is None
    carry = forward.call_args_list[1].kwargs['context'].carry
    assert not carry.requires_grad
    assert carry.grad_fn is None
    for output in (hidden_states[0], logits[0]):
      assert not output.requires_grad
      assert output.grad_fn is None
    assert all(output.dtype == torch.bfloat16 for output in logits)
    torch.testing.assert_close(carry, hidden_states[0], rtol=0, atol=0)
    assert loss.shape == tokens.shape and torch.isfinite(loss).all()
    loss.mean().backward()
  finally:
    hook.remove()

  # Shared parameters alone cannot distinguish the two backbone forwards.
  assert hidden_states[0].grad is None
  assert logits[0].grad is None
  for output in (hidden_states[1], logits[1]):
    assert output.grad is not None
    assert torch.isfinite(output.grad).all()
    assert output.grad.abs().sum() > 0
  for parameter in (model.backbone.vocab_embed.embedding,
                    model.backbone.output_layer.linear.weight):
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert parameter.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='DiT requires CUDA')
@pytest.mark.parametrize('name, trainer', [('lddm_u', algo.LDDM_U),
                                         ('lddm_m', algo.LDDM_M)])
def test_sampler_uses_discrete_posterior(name, trainer):
  directory = str(Path(__file__).resolve().parents[1] / 'configs')
  with initialize_config_dir(version_base=None, config_dir=directory):
    cfg = compose(config_name='config', overrides=[
      f'algo={name}', 'sampler=lddm', 'model=small', 'model.length=8',
      'model.hidden_size=32', 'model.cond_dim=16', 'model.n_blocks=1',
      'model.n_heads=4', 'model.dropout=0.0', 'trainer.devices=1',
      'trainer.num_nodes=1', 'loader.global_batch_size=2'])
  model = trainer(cfg, Tokenizer()).cuda().eval()
  assert isinstance(model.sampler, samplers.LDDMSampler)
  prefix = torch.tensor([[1, 2, 3], [3, 2, 1]], device='cuda')
  lengths = torch.tensor([1, 3], device='cuda')
  with patch('samplers.sample_posterior', wraps=samplers.sample_posterior) as posterior:
    tokens, metadata = model.generate_samples(
      num_samples=2, num_steps=3, prefix_tokens=prefix, prefix_lengths=lengths)
  assert metadata == {'nfe': 3} and posterior.call_count == 3
  assert all(call.args[0] == model.diffusion_type for call in posterior.call_args_list)
  assert torch.equal(tokens[0, :1], prefix[0, :1])
  assert torch.equal(tokens[1, :3], prefix[1, :3])
