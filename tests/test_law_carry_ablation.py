from unittest.mock import patch

import pytest
import torch

import algo
import samplers
from models.dit_loop import CarryRegularization
from test_law import CUDA, LossModule, Tokenizer, batch, build, config


def ablation(*overrides, device='cpu'):
  torch.manual_seed(17)
  cfg = config('algo=law_carry_ablation', 'sampler=law_carry_ablation', *overrides)
  model = algo.LAWCarryAblation(cfg, Tokenizer()).to(device).eval()
  torch.nn.init.normal_(model.backbone.output_layer.linear.weight, std=.1)
  if model.backbone.uses_hidden_carry:
    torch.nn.init.ones_(model.backbone.latent_ln.weight)
  return model


def test_public_law_has_no_selectors():
  model = build()
  assert all(name not in model.config.algo for name in (
    'state_representation', 'carry_representation', 'injection_mode'))
  assert not hasattr(model, '_state') and not hasattr(model, '_injection')
  assert not hasattr(model.backbone, 'latent_ln')
  assert 'per' not in model.config.algo.carry_regularization.temperature
  assert algo.LAW.__bases__ == (algo.LAW_BASE,)
  assert algo.LAWCarryAblation.__bases__ == (algo.LAW_BASE,)
  assert algo.LAW.nll is algo.LAWCarryAblation.nll


@pytest.mark.parametrize('representation', ['expected_embedding', 'hidden_representation'])
@pytest.mark.parametrize('injection', ['replace', 'add'])
def test_four_cells_and_rerunnable_variants(representation, injection):
  model = ablation(f'algo.carry_representation={representation}',
                    f'algo.injection_mode={injection}',
                    'algo.loss=ce', 'algo.pass_objective=last_only',
                    'algo.pass_count_probabilities=[1,2,3]', 'algo.dt=0.1',
                    'algo.stop_gradient_between_passes=False',
                    'model.mode_embedding=True')
  assert isinstance(model.sampler, samplers.LAWCarryAblationSampler)
  assert hasattr(model.backbone, 'latent_ln') == (representation == 'hidden_representation')
  assert model.config.algo.pass_count_probabilities == [1, 2, 3]
  torch.testing.assert_close(model._pass_count_probabilities,
                             torch.tensor([1/6, 2/6, 3/6], dtype=torch.float64))


@pytest.mark.parametrize('override', [
  'algo.carry_representation=hidden', 'algo.injection_mode=sum'])
def test_ablation_rejects_invalid_choices(override):
  with pytest.raises(ValueError):
    ablation(override)


@pytest.mark.parametrize('representation', ['expected_embedding', 'hidden_representation'])
@pytest.mark.parametrize('injection', ['replace', 'add'])
def test_injection_arithmetic(representation, injection):
  model = ablation(f'algo.carry_representation={representation}',
                    f'algo.injection_mode={injection}')
  tokens, _ = batch('cpu')
  carry = torch.randn(2, 8, 32)
  backbone = model.backbone
  context = samplers.CarryContext(carry=carry, carry_representation=representation,
                                  injection_mode=injection)
  base = backbone.vocab_embed(tokens)
  expected = carry
  if representation == 'hidden_representation':
    torch.nn.init.constant_(backbone.latent_ln.bias, .3)
    expected = backbone.latent_ln(carry)
  if injection == 'add':
    expected = base + expected
  torch.testing.assert_close(backbone._inject_carry(tokens, context), expected)
  context.carry = None
  expected = base
  if representation == 'hidden_representation':
    expected = base + backbone.latent_ln(torch.zeros_like(base))
  torch.testing.assert_close(backbone._inject_carry(tokens, context), expected)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('gaussian, rotation', [(0, 0), (.2, 0), (0, .3), (.2, .3)])
def test_regularization_reference_parity(dtype, gaussian, rotation):
  model = build()
  carry = torch.randn(2, 8, 32, dtype=dtype, requires_grad=True)
  torch.manual_seed(73)
  expected = carry
  if gaussian > 0:
    expected = expected + gaussian * torch.randn_like(expected)
  if rotation > 0:
    x = expected.float()
    norm = x.norm(dim=-1, keepdim=True)
    x_hat = x / norm.clamp_min(1e-8)
    random = torch.randn_like(x)
    random_hat = random / random.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    omega = (x_hat * random_hat).sum(-1, keepdim=True).clamp(
      -1 + 1e-7, 1 - 1e-7).arccos()
    sin_omega = omega.sin()
    a = ((1 - rotation) * omega).sin() / sin_omega
    b = (rotation * omega).sin() / sin_omega
    expected = (norm * (a * x_hat + b * random_hat)).to(dtype)
  expected_rng = torch.get_rng_state()
  expected_grad = torch.autograd.grad(expected.sum(), carry)[0]
  torch.manual_seed(73)
  actual = model.backbone._regularize_law_carry(
    carry, CarryRegularization(gaussian, rotation))
  actual_grad = torch.autograd.grad(actual.sum(), carry)[0]
  assert actual.dtype == dtype and torch.equal(expected_rng, torch.get_rng_state())
  torch.testing.assert_close(actual, expected, rtol=0, atol=0)
  torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
  assert torch.isfinite(actual_grad).all()


@pytest.mark.parametrize('representation', ['expected_embedding', 'hidden_representation'])
@pytest.mark.parametrize('stop', [False, True])
def test_carry_arithmetic_gradients_and_projection(representation, stop):
  model = ablation(f'algo.carry_representation={representation}')
  tokens, valid = batch('cpu')
  valid[:, 5] = 0
  logits = torch.randn(2, 8, 11, requires_grad=True)
  hidden = torch.randn(2, 8, 32, requires_grad=True)
  embedding = model.backbone.vocab_embed.matrix
  regularization = CarryRegularization(.02, .1)
  probabilities = (logits / 1.7).softmax(-1)
  if stop:
    probabilities = probabilities.detach()
  expected = probabilities @ embedding if representation == 'expected_embedding' else hidden
  torch.manual_seed(73)
  expected = model.backbone._regularize_law_carry(expected, regularization)
  expected = torch.where(valid[..., None].bool(), expected, embedding[tokens])
  if stop and representation == 'hidden_representation':
    expected = expected.detach() + model.backbone.vocab_embed(tokens) * 0.0
  expected_rng = torch.get_rng_state()
  torch.manual_seed(73)
  actual = model.backbone._build_carry(
    logits, {'latent': hidden}, tokens, carry_representation=representation,
    temperature=1.7, detach=stop, x0=tokens,
    valid_tokens=valid, carry_regularization=regularization)
  torch.testing.assert_close(actual, expected, rtol=0, atol=0)
  assert torch.equal(torch.get_rng_state(), expected_rng)
  actual[valid.bool()].sum().backward()
  source = logits if representation == 'expected_embedding' else hidden
  if stop:
    assert source.grad is None
  else:
    assert source.grad.abs().sum() > 0
  if representation == 'expected_embedding':
    assert model.backbone.vocab_embed.embedding.grad.abs().sum() > 0
  elif stop:
    assert model.backbone.vocab_embed.embedding.grad is not None
    assert model.backbone.vocab_embed.embedding.grad.abs().sum() == 0


@CUDA
@pytest.mark.parametrize('representation', ['expected_embedding', 'hidden_representation'])
@pytest.mark.parametrize('injection', ['replace', 'add'])
@pytest.mark.parametrize('stop', [False, True])
def test_ablation_loss_and_graph(representation, injection, stop):
  model = ablation(f'algo.carry_representation={representation}',
                    f'algo.injection_mode={injection}',
                    f'algo.stop_gradient_between_passes={stop}',
                    'algo.pass_objective=last_only',
                    'algo.pass_count_probabilities=[0,0,1]', device='cuda')
  tokens, valid = batch()
  outputs = []
  model.backbone.output_layer.register_forward_pre_hook(
    lambda module, inputs: outputs.append(inputs[0]))
  with patch.object(model.backbone, 'forward_law_carry_ablation',
                    wraps=model.backbone.forward_law_carry_ablation) as forward:
    loss, _ = model.nll(tokens, None, None, train_mode=True, valid_tokens=valid)
  assert forward.call_count == 3
  assert forward.call_args.kwargs['return_carry'] is False
  assert outputs[0].requires_grad == (not stop)
  if not stop:
    gradient = torch.autograd.grad(loss.sum(), outputs[0], retain_graph=True)[0]
    assert gradient.abs().sum() > 0
  loss.sum().backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all()
             for p in model.parameters() if p.requires_grad)


@CUDA
@pytest.mark.parametrize('injection', ['replace', 'add'])
def test_hidden_ddp_parameter_use(tmp_path, injection):
  torch.distributed.init_process_group(
    'gloo', init_method=f'file://{tmp_path / "ddp"}', rank=0, world_size=1)
  try:
    model = ablation('algo.carry_representation=hidden_representation',
                      f'algo.injection_mode={injection}',
                      'algo.pass_objective=last_only', device='cuda')
    wrapped = torch.nn.parallel.DistributedDataParallel(
      LossModule(model), device_ids=[0], find_unused_parameters=False)
    for _ in range(2):
      wrapped.zero_grad()
      wrapped(*batch()).backward()
      assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
  finally:
    torch.distributed.destroy_process_group()


@CUDA
def test_law_ablation_training_parity():
  canonical = build(device='cuda')
  variant = ablation(device='cuda')
  variant.load_state_dict(canonical.state_dict(), strict=True)
  results = []
  for model in (canonical, variant):
    torch.manual_seed(91)
    tokens, valid = batch()
    loss, times = model.nll(tokens, None, None, train_mode=True, valid_tokens=valid)
    loss.sum().backward()
    results.append((loss, times, torch.cuda.get_rng_state(),
                    [p.grad for p in model.parameters()]))
  for actual, expected in zip(results[0][:3], results[1][:3]):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
  for actual, expected in zip(results[0][3], results[1][3]):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
