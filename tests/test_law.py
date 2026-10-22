from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import lightning as L
from hydra import compose, initialize_config_dir

import algo
import samplers
from models.dit_loop import CarryRegularization, apply_temperature


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='DiT requires CUDA')


class Tokenizer:
  vocab_size = 11
  bos_token_id = 0

  def __len__(self):
    return self.vocab_size


def config(*overrides):
  directory = str(Path(__file__).resolve().parents[1] / 'configs')
  with initialize_config_dir(version_base=None, config_dir=directory):
    return compose(config_name='config', overrides=[
      'algo=law', 'model=small-loop', 'sampler=law',
      'model.length=8', 'model.hidden_size=32', 'model.cond_dim=16',
      'model.n_blocks=1', 'model.n_heads=4', 'model.dropout=0.0',
      'trainer.devices=1', 'trainer.num_nodes=1', 'loader.global_batch_size=2',
      'eval.generate_samples=False', *overrides])


def build(*overrides, device='cpu'):
  torch.manual_seed(17)
  model = algo.LAW(config(*overrides), Tokenizer()).to(device)
  # A zero output projection would hide graph failures behind zero derivatives.
  torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=0.1)
  model.eval()
  return model


def batch(device='cuda'):
  tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8],
                         [2, 3, 4, 5, 6, 7, 8, 9]], device=device)
  valid = torch.ones_like(tokens)
  valid[:, :2] = 0
  return tokens, valid


@pytest.mark.parametrize('count', [1, 2, 3])
def test_fixed_pass_count(count):
  weights = [0] * (count - 1) + [5]
  model = build(f'algo.pass_count_probabilities={weights}')
  assert model.config.algo.pass_count_probabilities == weights
  torch.testing.assert_close(model._pass_count_probabilities,
                             torch.tensor([0] * (count - 1) + [1], dtype=torch.float64))
  assert {model._sample_pass_count() for _ in range(20)} == {count}


def test_normalization_and_same_seed_pass_count_sequences():
  model = build('algo.pass_count_probabilities=[2,3,5]', 'seed=73')
  identical = build('algo.pass_count_probabilities=[2,3,5]', 'seed=73')
  different = build('algo.pass_count_probabilities=[2,3,5]', 'seed=74')
  assert model.config.algo.pass_count_probabilities == [2, 3, 5]
  torch.testing.assert_close(model._pass_count_probabilities,
                             torch.tensor([.2, .3, .5], dtype=torch.float64))
  first = [model._sample_pass_count() for _ in range(2000)]
  torch.manual_seed(8234)
  second = [identical._sample_pass_count() for _ in range(2000)]
  assert first == second
  assert first != [different._sample_pass_count() for _ in range(2000)]
  frequencies = torch.bincount(torch.tensor(first), minlength=4)[1:] / len(first)
  torch.testing.assert_close(frequencies, torch.tensor([.2, .3, .5]),
                             atol=.04, rtol=0)


def test_pass_count_generator_reuse_and_advancing_draws():
  model = build('algo.pass_count_probabilities=[2,3,5]')
  generator = model._pass_count_generator
  assert generator.device == torch.device('cpu')
  initial_state = generator.get_state()
  rng_state = torch.get_rng_state()
  with patch('torch.Generator', side_effect=AssertionError('generator recreated')):
    first = [model._sample_pass_count() for _ in range(30)]
    middle_state = generator.get_state()
    second = [model._sample_pass_count() for _ in range(30)]
  assert model._pass_count_generator is generator
  assert not torch.equal(initial_state, middle_state)
  assert not torch.equal(middle_state, generator.get_state())
  assert first != second
  torch.testing.assert_close(torch.get_rng_state(), rng_state, rtol=0, atol=0)


@pytest.mark.parametrize('name, model_class', [
  ('law', algo.LAW), ('law_carry_ablation', algo.LAWCarryAblation)])
def test_pass_count_checkpoint_continuation(tmp_path, name, model_class):
  cfg = config(f'algo={name}', f'sampler={name}',
               'algo.pass_count_probabilities=[2,3,5]', 'seed=73')
  model = model_class(cfg, Tokenizer())
  model._trainer = SimpleNamespace(
    accumulate_grad_batches=1,
    train_dataloader=SimpleNamespace(sampler=object()))
  prefix = [model._sample_pass_count() for _ in range(30)]
  checkpoint = {'state_dict': model.state_dict(), 'loops': {'fit_loop': {
    'epoch_progress': {'current': {'completed': 0}},
    'epoch_loop.batch_progress': {'total': {}, 'current': {}},
    'epoch_loop.automatic_optimization.optim_progress': {'optimizer': {'step': {
      'total': {'completed': 30}, 'current': {'completed': 30}}}},
    'epoch_loop.state_dict': {}}}}
  model.on_save_checkpoint(checkpoint)
  path = tmp_path / 'pass_counts.ckpt'
  torch.save(checkpoint, path)
  expected = [model._sample_pass_count() for _ in range(30)]
  assert expected != prefix

  restored = model_class(cfg, Tokenizer())
  checkpoint = torch.load(path, map_location='cpu', weights_only=False)
  restored.on_load_checkpoint(checkpoint)
  restored.load_state_dict(checkpoint['state_dict'])
  assert [restored._sample_pass_count() for _ in range(30)] == expected
  assert torch.equal(restored._pass_count_generator.get_state(),
                     model._pass_count_generator.get_state())
  del checkpoint['pass_count_generator_state']
  with pytest.raises(KeyError, match='pass_count_generator_state'):
    restored.on_load_checkpoint(checkpoint)


def distributed_counts(rank, rendezvous):
  torch.distributed.init_process_group(
    'gloo', init_method=rendezvous, rank=rank, world_size=2)
  try:
    torch.manual_seed(73 + rank)
    model = algo.LAW(config('algo.pass_count_probabilities=[2,3,5]', 'seed=73'),
                     Tokenizer())
    counts = torch.tensor([model._sample_pass_count() for _ in range(30)])
    gathered = [torch.empty_like(counts) for _ in range(2)]
    torch.distributed.all_gather(gathered, counts)
    assert torch.equal(gathered[0], gathered[1])
  finally:
    torch.distributed.destroy_process_group()


def test_two_rank_pass_counts(tmp_path):
  torch.multiprocessing.spawn(distributed_counts,
                              args=(f'file://{tmp_path / "counts"}',),
                              nprocs=2, join=True)


@pytest.mark.parametrize('override', [
  'algo.loss=other', 'algo.pass_objective=last',
  'algo.stop_gradient_between_passes=none', 'model.type=dit',
  'algo.diffusion_type=absorbing', 'algo.dt=0',
  'algo.carry_regularization.gaussian_std=-1',
  'algo.carry_regularization.rotation_fraction=1.1'])
def test_invalid_policy(override):
  with pytest.raises(ValueError):
    build(override)


def test_time_boundary():
  for dt in (.5, .6, .499999999):
    with pytest.raises(ValueError, match=r'\(K - 1\) \* dt < 1'):
      build('algo.pass_count_probabilities=[0,0,1]', f'algo.dt={dt}')
  model = build('algo.pass_count_probabilities=[0,0,1]', 'algo.dt=0.1')
  t = torch.tensor([0., .1, .5])
  sigmas, alpha, _, start = model._pass_schedule(t, 3)
  assert (start > .2).all()
  assert (start - 2 * model.config.algo.dt > 0).all()
  torch.testing.assert_close(start[1:], torch.tensor([.28, .6]))
  for i, sigma in enumerate(sigmas):
    _, expected_alpha = model.noise(start - i * .1)
    torch.testing.assert_close(sigma, -expected_alpha.log())
  torch.testing.assert_close(alpha[:, 0], model.noise(start)[1])
  model.config.algo.advance_time_between_passes = False
  sigmas, _, _, start = model._pass_schedule(t[1:], 3)
  torch.testing.assert_close(start, t[1:])
  assert all(torch.equal(sigmas[0], sigma) for sigma in sigmas)


def test_four_pass_float32_final_time_is_positive():
  model = build('algo.pass_count_probabilities=[0,0,0,1]', 'algo.dt=0.3333333')
  with patch.object(model.noise, 'forward', wraps=model.noise.forward) as noise:
    model._pass_schedule(torch.tensor([0., .001, .5]), 4)
  noise.assert_called_once()
  times = noise.call_args.args[0]
  assert times.dtype == torch.float32 and times.shape == (4, 3)
  assert (times[-1] > 0).all()
  assert (times[0] <= 1).all()


def test_time_boundary_uses_float32_offset_multiplication():
  with pytest.raises(ValueError, match=r'\(K - 1\) \* dt < 1'):
    build(f'algo.pass_count_probabilities={[0] * 25 + [1]}', 'algo.dt=0.0399999988')


@pytest.mark.parametrize('count', [1, 2, 3])
@pytest.mark.parametrize('advance', [False, True])
@pytest.mark.parametrize('time_conditioning', [False, True])
@pytest.mark.parametrize('batch_size', [1, 4])
def test_pass_schedule_matches_scalar_reference(count, advance, time_conditioning,
                                                batch_size):
  model = build(f'algo.advance_time_between_passes={advance}',
                f'algo.time_conditioning={time_conditioning}', 'algo.dt=0.1')
  t = torch.tensor([0., .1, .5, 1.])[:batch_size]
  dt = model.config.algo.dt if advance else 0
  t_min = (count - 1) * dt
  lower = torch.nextafter(t.new_tensor(t_min), t.new_tensor(1.0))
  start = (t_min + t * (1 - t_min)).clamp_min(lower)
  dalpha, alpha = model.noise(start)
  sigmas = []
  for i in range(count):
    _, pass_alpha = model.noise(start - i * dt)
    sigmas.append(model._process_sigma(
      model._sigma_from_alphat(pass_alpha.unsqueeze(-1))))
  expected = (torch.stack(sigmas), alpha.unsqueeze(-1), dalpha.unsqueeze(-1), start)
  with patch.object(model.noise, 'forward', wraps=model.noise.forward) as noise:
    with patch.object(model, '_process_sigma', wraps=model._process_sigma) as process:
      actual = model._pass_schedule(t, count)
  noise.assert_called_once()
  assert noise.call_args.args[0].shape == (count, batch_size)
  process.assert_called_once()
  assert process.call_args.args[0].shape == (count * batch_size, 1)
  assert actual[0].shape == (count, batch_size)
  for result, reference in zip(actual, expected):
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


@CUDA
@pytest.mark.parametrize('count', [1, 2, 3])
@pytest.mark.parametrize('objective', ['mean', 'last_only'])
@pytest.mark.parametrize('stop', [True, False])
@pytest.mark.parametrize('loss_name', ['elbo', 'ce'])
def test_loss_membership_and_graph(count, objective, stop, loss_name):
  weights = [0] * (count - 1) + [1]
  model = build(f'algo.pass_count_probabilities={weights}',
                f'algo.pass_objective={objective}',
                f'algo.stop_gradient_between_passes={stop}',

                f'algo.loss={loss_name}', device='cuda')
  tokens, valid = batch()
  logits = []
  predictions = []
  loss_inputs = []

  def record_logits(module, inputs, output):
    del module, inputs
    logits.append(output)

  model.backbone.output_layer.register_forward_hook(record_logits)
  forward = model.backbone.forward_carry
  pass_loss = model._pass_loss

  def record_forward(*args, **kwargs):
    result = forward(*args, **kwargs)
    predictions.append(result[0])
    return result

  def record_loss(*args):
    loss_inputs.append(args)
    return pass_loss(*args)

  model.backbone.forward_carry = record_forward
  model._pass_loss = record_loss
  loss, _ = model.nll(tokens, None, None, train_mode=True, valid_tokens=valid)
  expected_indices = list(range(count)) if objective == 'mean' else [count - 1]
  assert len(loss_inputs) == len(expected_indices)
  for inputs, index in zip(loss_inputs, expected_indices):
    assert inputs[0] is predictions[index]
  if loss_name == 'ce':
    expected_loss = sum(-predictions[i].gather(-1, tokens[..., None]).squeeze(-1)
                        * valid for i in expected_indices) / len(expected_indices)
  else:
    expected_loss = sum(pass_loss(*inputs) for inputs in loss_inputs) / len(loss_inputs)
  torch.testing.assert_close(loss, expected_loss)
  assert torch.isfinite(loss).all()
  assert (loss[:, :2] == 0).all()

  final = pass_loss(predictions[-1], *loss_inputs[-1][1:]).sum()
  sources = logits
  for source in sources[:-1]:
    if stop and objective == 'last_only':
      assert not source.requires_grad
    else:
      gradient = torch.autograd.grad(final, source, retain_graph=True,
                                     allow_unused=True)[0]
      if stop:
        assert gradient is None
      else:
        assert gradient is not None and gradient.abs().sum() > 0
  loss.sum().backward()
  assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
  embedding_grad = model.backbone.vocab_embed.embedding.grad
  assert embedding_grad is not None
  assert embedding_grad.abs().sum() > 0
  assert all(torch.isfinite(p.grad).all() for p in model.parameters()
             if p.grad is not None)


@CUDA
def test_detached_mixture_keeps_live_embedding_gradient():
  model = build(device='cuda')
  tokens, _ = batch()
  logits = torch.randn(2, 8, 11, device='cuda', requires_grad=True)
  carry = model.backbone._build_carry(
    logits, None, tokens, carry_representation='expected_embedding', temperature=None,
    detach=True, x0=tokens, valid_tokens=None)
  embedding = model.backbone.vocab_embed.embedding
  probabilities = logits.detach().softmax(-1)
  torch.testing.assert_close(carry, probabilities @ embedding)
  carry.sum().backward()
  assert logits.grad is None
  torch.testing.assert_close(
    embedding.grad, probabilities.sum((0, 1))[:, None].expand_as(embedding),
    rtol=2e-4, atol=2e-4)  # CUDA may enable TF32 matmul for the backward.


def test_all_parameters_trainable_in_optimizer_and_ema():
  model = algo.LAW(config(), Tokenizer())
  parameters = list(model.parameters())
  assert all(p.requires_grad for p in parameters)
  optimizers, _ = model.configure_optimizers()
  optimized = [p for group in optimizers[0].param_groups for p in group['params']]
  assert {id(p) for p in optimized} == {id(p) for p in parameters}
  tracked = list(model._get_parameters())
  assert {id(p) for p in tracked} == {id(p) for p in parameters}
  assert len(parameters) == len(model.ema.shadow_params)
  for parameter, shadow in zip(tracked, model.ema.shadow_params):
    torch.testing.assert_close(parameter, shadow)


class LossModule(torch.nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, tokens, valid):
    return self.model._loss(tokens, valid, train_mode=True).loss


@CUDA
@pytest.mark.parametrize('stop', [True, False])
def test_ddp_parameter_use_at_low_noise(tmp_path, stop):
  torch.distributed.init_process_group(
    'gloo', init_method=f'file://{tmp_path / "ddp"}', rank=0, world_size=1)
  try:
    model = build('algo.pass_count_probabilities=[0,0,1]',
                  'algo.pass_objective=last_only', 'model.mode_embedding=True',

                  f'algo.stop_gradient_between_passes={stop}', device='cuda')
    model._sample_t = lambda n, step: torch.full((n,), .01, device='cuda')
    wrapped = torch.nn.parallel.DistributedDataParallel(
      LossModule(model), device_ids=[0], find_unused_parameters=False)
    tokens, valid = batch()
    for _ in range(2):
      wrapped.zero_grad()
      wrapped(tokens, valid).backward()
      missing = [name for name, p in model.named_parameters()
                 if p.requires_grad and p.grad is None]
      assert not missing, missing
  finally:
    torch.distributed.destroy_process_group()


@CUDA
@pytest.mark.parametrize('loss_name', ['elbo', 'ce'])
def test_amp_single_pass_oracle_and_short_training(loss_name):
  model = build(f'algo.loss={loss_name}', 'algo.pass_count_probabilities=[1]',
                device='cuda')
  tokens, valid = batch()

  def single_pass_oracle():
    t = model._sample_t(tokens.shape[0], None)
    sigmas, alpha, dalpha, _ = model._pass_schedule(t, 1)
    xt = model.q_xt(tokens, alpha, use_pure_noise=False, valid_tokens=valid)
    context = samplers.CarryContext(
      carry=None, carry_representation='expected_embedding',
      injection_mode='replace', looped=False)
    with torch.amp.autocast('cuda', dtype=torch.float32, cache_enabled=False):
      logits = model.backbone(None, xt, sigmas[0], context)
      log_p = logits.log_softmax(-1)
      if loss_name == 'ce':
        loss = -log_p.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
      else:
        loss = model.nll_per_token(
          log_x_theta=log_p, xt=xt, x0=tokens,
          alpha_t=alpha, dalpha_t=dalpha)
    return (loss * valid).sum() / valid.sum()

  torch.manual_seed(123)
  with torch.autocast('cuda', dtype=torch.bfloat16):
    actual = model._loss(tokens, valid, train_mode=True).loss
  torch.manual_seed(123)
  with torch.autocast('cuda', dtype=torch.bfloat16):
    expected = single_pass_oracle()
  torch.testing.assert_close(actual, expected, rtol=0, atol=0)

  model = build(f'algo.loss={loss_name}', 'algo.pass_count_probabilities=[0,0,1]',
                'algo.pass_objective=last_only',
                'algo.carry_regularization.temperature.enabled=True',
                'algo.carry_regularization.gaussian_std=0.01',
                'algo.carry_regularization.rotation_fraction=0.1', device='cuda')
  observations = []
  build_carry = model.backbone._build_carry
  regularize = model.backbone._regularize_law_carry
  pass_loss = model._pass_loss
  matmul = torch.matmul

  def record_normalization(module, inputs, output):
    del module, inputs
    assert output.dtype == torch.float32

  def record_loss(log_p, *args):
    assert log_p.dtype == torch.float32
    return pass_loss(log_p, *args)

  def record_matmul(probabilities, embedding):
    assert probabilities.dtype == embedding.dtype == torch.float32
    return matmul(probabilities, embedding)

  def record_carry(logits, *args, **kwargs):
    assert logits.dtype == torch.bfloat16  # DiT's internal autocast
    assert torch.get_autocast_dtype('cuda') == torch.float32
    result = build_carry(logits, *args, **kwargs)
    observations.append(result.dtype)
    assert result.requires_grad  # outside warm-up prediction no_grad
    return result

  def record_regularizer(carry, *args):
    assert carry.dtype == torch.float32  # live float32 embedding mixture
    assert torch.get_autocast_dtype('cuda') == torch.float32
    result = regularize(carry, *args)
    assert result.dtype == torch.float32
    return result

  model.backbone._build_carry = record_carry
  model.backbone._regularize_law_carry = record_regularizer
  model.backbone.output_layer.norm_final.register_forward_hook(record_normalization)
  model._pass_loss = record_loss
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
  for _ in range(3):
    optimizer.zero_grad()
    with torch.autocast('cuda', dtype=torch.bfloat16):
      with patch('torch.matmul', record_matmul):
        loss = model._loss(tokens, valid, train_mode=True).loss
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert model.backbone.vocab_embed.embedding.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()
               if p.grad is not None)
    optimizer.step()
  assert observations == [torch.float32] * 6


@CUDA
@pytest.mark.parametrize('loss_name', ['elbo', 'ce'])
def test_validation_objective_and_sampling(loss_name):
  model = build(f'algo.loss={loss_name}', device='cuda')
  model._eval_mode()
  tokens, valid = batch()
  logs = {}
  model.log = lambda name, value, **kwargs: logs.update({name: value})
  predictions = []
  forward = model.backbone.forward_carry

  def record_forward(*args, **kwargs):
    result = forward(*args, **kwargs)
    predictions.append(result[0])
    return result

  model.backbone.forward_carry = record_forward
  torch.manual_seed(33)
  loss = model.validation_step({'input_ids': tokens, 'attention_mask': valid}, 0)
  assert len(predictions) == 1
  if loss_name == 'ce':
    token_ce = -predictions[0].gather(-1, tokens[..., None]).squeeze(-1)
    torch.testing.assert_close(loss, (token_ce * valid).sum() / valid.sum())
  torch.manual_seed(33)
  expected = model._loss(tokens, valid).loss
  model.on_validation_epoch_end()
  torch.testing.assert_close(logs[f'val/{loss_name}'], expected.to(torch.float64))
  torch.testing.assert_close(loss, expected)
  samples, metadata = model.generate_samples(num_samples=2, num_steps=3)
  assert samples.shape == tokens.shape and metadata['nfe'] == 3
  assert ((samples >= 0) & (samples < len(Tokenizer()))).all()
  assert isinstance(model.sampler, samplers.LAWSampler)


def test_validation_token_weighting():
  model = build('algo.loss=ce')
  model._eval_mode()
  logs = {}
  model.log = lambda name, value, **kwargs: logs.update({name: value})
  model.metrics.update_valid(torch.tensor(12.), 0., torch.tensor(2))
  model.metrics.update_valid(torch.tensor(20.), 0., torch.tensor(5))
  model.on_validation_epoch_end()
  torch.testing.assert_close(logs['val/ce'], torch.tensor(32 / 7, dtype=torch.float64))


@pytest.mark.parametrize('count', [1, 2, 3])
@pytest.mark.parametrize('resample', [False, True])
def test_temperatures_match_scalar_reference(count, resample):
  model = build('algo.carry_regularization.temperature.enabled=True',
                'algo.carry_regularization.temperature.range_scale=4',
                f'algo.carry_regularization.temperature.resample_per_pass={resample}')
  n = 256
  draw_count = count if resample else 1
  torch.manual_seed(73)
  expected = []
  for _ in range(draw_count):
    low = .25 + .75 * torch.rand(n)
    high = 1 + 3 * torch.rand(n)
    pick = torch.rand(n) < .5
    expected.append(torch.where(pick, low, high).view(n, 1, 1))
  if not resample:
    expected *= count
  rng_state = torch.get_rng_state()
  torch.manual_seed(73)
  with patch('torch.rand', wraps=torch.rand) as draw:
    temperatures = model._carry_temperatures(256, count, 'cpu', True)
  draw.assert_called_once_with(draw_count, 3, n, device='cpu')
  assert temperatures.shape == (count, n, 1, 1)
  torch.testing.assert_close(temperatures, torch.stack(expected), rtol=0, atol=0)
  torch.testing.assert_close(torch.get_rng_state(), rng_state, rtol=0, atol=0)
  assert .25 <= temperatures.min() and temperatures.max() <= 4
  assert (temperatures < 1).any() and (temperatures > 1).any()
  if count > 1:
    assert torch.equal(temperatures[0], temperatures[1]) == (not resample)
    if not resample:
      assert temperatures.stride(0) == 0


@pytest.mark.parametrize('enabled, train_mode', [(False, True), (True, False),
                                               (False, False)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_disabled_temperatures(enabled, train_mode, dtype):
  model = build(f'algo.carry_regularization.temperature.enabled={enabled}')
  logits = torch.randn(2, 8, 11, dtype=dtype, requires_grad=True)
  with patch('torch.rand', side_effect=AssertionError('unexpected draw')):
    temperature = model._carry_temperatures(2, 3, 'cpu', train_mode)
    result = apply_temperature(logits, temperature)
  assert type(temperature) is float and temperature == 1.0
  assert result is logits


@CUDA
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('train_mode', [False, True])
@pytest.mark.parametrize('resample', [False, True])
def test_forward_carry_temperatures(enabled, train_mode, resample):
  model = build(f'algo.carry_regularization.temperature.enabled={enabled}',
                f'algo.carry_regularization.temperature.resample_per_pass={resample}',
                'algo.carry_regularization.gaussian_std=0.01',
                'algo.carry_regularization.rotation_fraction=0.1',
                'algo.pass_count_probabilities=[0,0,1]', device='cuda')
  tokens, valid = batch()
  with patch.object(model.backbone, 'forward_carry',
                    wraps=model.backbone.forward_carry) as forward, \
       patch.object(model.backbone, '_regularize_law_carry',
                    wraps=model.backbone._regularize_law_carry) as regularize:
    loss, _ = model.nll(tokens, None, None, train_mode=train_mode, valid_tokens=valid)
  assert torch.isfinite(loss).all()
  assert forward.call_count == (3 if train_mode else 1)
  assert regularize.call_count == (2 if train_mode else 0)
  for call in forward.call_args_list:
    assert 'reg' not in call.kwargs
    assert call.kwargs['carry_regularization'] is (
      model._carry_regularization if train_mode else None)
    temperature = call.kwargs['carry_temperature']
    if enabled and train_mode:
      assert torch.is_tensor(temperature) and temperature.shape == (2, 1, 1)
    else:
      assert type(temperature) is float and temperature == 1.0
  if enabled and train_mode:
    temperatures = [call.kwargs['carry_temperature'] for call in forward.call_args_list]
    assert torch.equal(temperatures[0], temperatures[1]) == (not resample)


def test_regularization():
  model = build()
  assert [field.name for field in fields(model._carry_regularization)] == [
    'gaussian_std', 'rotation_fraction']
  carry = torch.randn(2, 8, 32, requires_grad=True)
  regularize = model.backbone._regularize_law_carry
  with patch('torch.randn_like', side_effect=AssertionError('unexpected draw')):
    assert regularize(carry, model._carry_regularization) is carry
  rotated = regularize(carry, CarryRegularization(0, .3))
  torch.testing.assert_close(rotated.norm(dim=-1), carry.norm(dim=-1))
  rotated.sum().backward()
  assert torch.isfinite(carry.grad).all() and carry.grad.abs().sum() > 0
  torch.manual_seed(73)
  expected = carry + .2 * torch.randn_like(carry)
  expected_rng = torch.get_rng_state()
  torch.manual_seed(73)
  noisy = regularize(carry, CarryRegularization(.2, 0))
  torch.testing.assert_close(noisy, expected, rtol=0, atol=0)
  assert torch.equal(torch.get_rng_state(), expected_rng)


@CUDA
def test_lightning_train_save_load_validate(tmp_path):
  model = build('algo.loss=ce', 'algo.pass_objective=last_only',
                'algo.pass_count_probabilities=[0,0,1]',
                'algo.stop_gradient_between_passes=False',
                'loader.batch_size=2', 'loader.num_workers=1')
  tokens, valid = batch('cpu')
  data = [{'input_ids': tokens[i], 'attention_mask': valid[i]} for i in range(2)]
  loader = torch.utils.data.DataLoader(data * 2, batch_size=2)
  trainer = L.Trainer(
    accelerator='gpu', devices=1, precision='bf16-mixed', max_steps=2,
    limit_val_batches=1, num_sanity_val_steps=0, logger=False,
    enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
    default_root_dir=tmp_path)
  trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
  assert trainer.global_step == 2
  assert trainer.callback_metrics['trainer/pass_count'] == 3
  assert torch.isfinite(trainer.callback_metrics['val/ce'])
  checkpoint_path = tmp_path / 'law.ckpt'
  trainer.save_checkpoint(checkpoint_path)
  checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
  restored = algo.LAW(model.config, Tokenizer())
  restored.on_load_checkpoint(checkpoint)
  restored.load_state_dict(checkpoint['state_dict'])
  for name, value in restored.state_dict().items():
    torch.testing.assert_close(value, model.state_dict()[name].cpu())
  assert restored.ema.num_updates == 2
  restored.cuda().eval()
  tokens, valid = batch()
  for use_ema in (False, True):
    if use_ema:
      restored._eval_mode()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
      assert torch.isfinite(restored._loss(tokens, valid).loss)
    if use_ema:
      restored._train_mode()
  checkpoint['ema']['shadow_params'].pop(0)
  with pytest.raises(ValueError, match='checkpoint EMA'):
    restored.on_load_checkpoint(checkpoint)
