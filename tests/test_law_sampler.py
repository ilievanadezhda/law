from unittest.mock import patch

import pytest
import torch
from omegaconf import open_dict

import samplers
from models.dit import DIT
from test_law import CUDA, build, config
from temperature_schedules import get_temperature_schedule, sampling_time


def parse_schedule(spec):
  return get_temperature_schedule(spec)


@pytest.fixture
def model():
  network = build()

  def prediction(backbone, x0, xt, sigma, context, return_aux):
    carry = backbone._inject_carry(xt, context)
    logits = carry @ backbone.vocab_embed.matrix.T
    return (logits, {'latent': carry}) if return_aux else logits

  with patch.object(DIT, 'forward', prediction):
    yield network


def test_dispatch_and_config_surface():
  cfg = config()
  assert isinstance(samplers.get_sampler(cfg), samplers.LAWSampler)
  assert all(key not in cfg.sampler for key in (
    'temperature', 'decoder_mode', 'carry_representation', 'injection_mode'))
  for predictor in ('ancestral', 'law-carry-ablation'):
    cfg.sampler.predictor = predictor
    with pytest.raises(ValueError, match='sampler=law'):
      samplers.get_sampler(cfg)
  cfg = config('algo=lddm_u')
  with pytest.raises(ValueError, match='sampler=law'):
    samplers.get_sampler(cfg)
  cfg.sampler.predictor = 'lddm'
  with open_dict(cfg.sampler):
    cfg.sampler.temperature_schedule = 'constant-1.0'
  assert isinstance(samplers.get_sampler(cfg), samplers.LDDMSampler)

  cfg = config('algo=mdlm', 'sampler=ancestral')
  for predictor in ('loopholing', 'loopholing-atw', 'loophole-ancestral',
                    'loopholing-pc'):
    cfg.sampler.predictor = predictor
    with pytest.raises(ValueError, match='Unknown sampler predictor'):
      samplers.get_sampler(cfg)


@pytest.mark.parametrize('schedule', [
  'constant-2', 'last-2', 'linear-4-1', 'cosine-4-1'])
@pytest.mark.parametrize('steps', [1, 4])
@pytest.mark.parametrize('prefix_lengths', [None, [0, 0], [1, 3], [8, 8]])
@pytest.mark.parametrize('readout', ['ancestral', 'greedy'])
def test_trajectory_and_raw_decode(model, schedule, steps, prefix_lengths, readout):
  sampler = samplers.LAWSampler(steps_policy='full', noise_removal=readout,
                                temperature_schedule=parse_schedule(schedule))
  lengths = None if prefix_lengths is None else torch.tensor(prefix_lengths)
  prefix = None if lengths is None else torch.arange(8).expand(2, -1)
  state = sampler.init_state(model, 2, num_steps=steps,
                                prefix_tokens=prefix, prefix_lengths=lengths)
  original = state.xt.clone()
  forward = model.backbone.forward_law
  predictions = []

  def record(trajectory, sigma, **kwargs):
    assert trajectory.regularization is None
    expected_temperature = parse_schedule(schedule)(
      sampling_time(state.step_idx, steps))
    assert kwargs['carry_temperature'] == expected_temperature
    assert kwargs['temperature'] == expected_temperature
    assert (trajectory.carry is None) == (state.step_idx == 0)
    torch.testing.assert_close(trajectory.xt, original, rtol=0, atol=0)
    result = forward(trajectory, sigma, **kwargs)
    if result[1] is not None and lengths is not None:
      fixed = ~trajectory.valid_tokens.bool()
      clean = model.backbone.vocab_embed.matrix[original]
      torch.testing.assert_close(result[1].carry[fixed], clean[fixed])
    predictions.append(result[0])
    return result

  with patch.object(model.backbone, 'forward_law', record), \
       patch('samplers.sample_posterior', side_effect=AssertionError('posterior')), \
       patch('samplers.compute_posterior', side_effect=AssertionError('posterior')), \
       patch('samplers.sample_categorical', wraps=samplers.sample_categorical) as draw:
    for index in range(steps):
      assert not state.done
      sampler.step(model, state)
      if index < steps - 1:
        torch.testing.assert_close(state.xt, original, rtol=0, atol=0)
        draw.assert_not_called()
    assert state.done and state.nfe == state.step_idx == steps
    assert sampler.step(model, state) is state
  probabilities = predictions[-1][:, state.start_idx:].exp()
  if readout == 'ancestral':
    assert draw.call_count == 1
    torch.testing.assert_close(draw.call_args.args[0], probabilities, rtol=0, atol=0)
  else:
    draw.assert_not_called()
    expected = original.clone()
    expected[:, state.start_idx:] = probabilities.argmax(-1)
    sampler._project_prefix(expected, prefix, lengths)
    torch.testing.assert_close(state.xt, expected)
  if lengths is not None:
    fixed = torch.arange(8)[None, :] < lengths[:, None]
    assert torch.equal(state.xt[fixed], prefix[fixed])


@pytest.mark.parametrize('schedule', ['constant-1', 'last-1', 'last-2'])
@pytest.mark.parametrize('readout', ['greedy', 'ancestral'])
def test_seeded_ablation_replacement_parity(model, schedule, readout):
  current = samplers.LAWSampler(steps_policy='full', noise_removal=readout,
                                temperature_schedule=parse_schedule(schedule))
  reference = samplers.LAWCarryAblationSampler(
    carry_representation='expected_embedding', injection_mode='replace',
    temperature_schedule=parse_schedule(schedule), noise_removal=readout)
  prefix = torch.tensor([[1, 2, 3], [3, 2, 1]])
  lengths = torch.tensor([1, 3])
  carries = []
  build_carry = model.backbone._build_carry

  def record(*args, **kwargs):
    result = build_carry(*args, **kwargs)
    carries.append(result)
    return result

  results = []
  with patch.object(model.backbone, '_build_carry', record), torch.no_grad():
    for sampler in (reference, current):
      torch.manual_seed(91)
      tokens, metadata = samplers.run_sampler(
        sampler, model, 2, num_steps=4, prefix_tokens=prefix, prefix_lengths=lengths)
      results.append((tokens, metadata, torch.get_rng_state()))
  for first, second in zip(carries[:3], carries[3:]):
    torch.testing.assert_close(first, second, rtol=0, atol=0)
  assert results[0][1] == results[1][1] == {'nfe': 4}
  assert torch.equal(results[0][0], results[1][0])
  assert torch.equal(results[0][2], results[1][2])


def test_constant_temperature_changes_carry(model):
  carries = []
  for schedule in ('last-2', 'constant-2'):
    sampler = samplers.LAWSampler(steps_policy='full', noise_removal='greedy',
                                  temperature_schedule=parse_schedule(schedule))
    torch.manual_seed(91)
    state = sampler.init_state(model, 2, num_steps=2)
    sampler.step(model, state)
    carries.append(state.trajectory.carry)
  assert not torch.equal(*carries)


@pytest.mark.parametrize('lengths, expected', [([0, 4], 4), ([4, 6], 2), ([8, 8], 1)])
def test_proportional_steps(model, lengths, expected):
  sampler = samplers.LAWSampler(steps_policy='proportional', noise_removal='greedy',
                                temperature_schedule=parse_schedule('constant-1'))
  _, metadata = samplers.run_sampler(
    sampler, model, 2, num_steps=4, prefix_tokens=torch.zeros(2, 8, dtype=torch.long),
    prefix_lengths=torch.tensor(lengths))
  assert metadata['nfe'] == expected


@pytest.mark.parametrize('prefix_length', [7, 8])
@pytest.mark.parametrize('readout', ['greedy', 'ancestral'])
def test_ablation_proportional_minimum_step(model, prefix_length, readout):
  sampler = samplers.LAWCarryAblationSampler(
    carry_representation='expected_embedding', injection_mode='replace',
    steps_policy='proportional', noise_removal=readout,
    temperature_schedule=parse_schedule('constant-1'))
  prefix = torch.arange(8).expand(2, -1)
  tokens, metadata = samplers.run_sampler(
    sampler, model, 2, num_steps=1, prefix_tokens=prefix,
    prefix_lengths=torch.full((2,), prefix_length))
  assert metadata == {'nfe': 1}
  assert tokens.shape == (2, 8)
  assert torch.equal(tokens[:, :prefix_length], prefix[:, :prefix_length])
  assert ((tokens >= 0) & (tokens < model.vocab_size)).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('readout', ['greedy', 'ancestral'])
def test_raw_readout_dtype_and_rng(model, dtype, readout):
  sampler = samplers.LAWSampler(steps_policy='full', noise_removal=readout,
                                temperature_schedule=parse_schedule('constant-1'))
  state = sampler.init_state(model, 2, num_steps=1)
  log_p = torch.randn(2, 8, 11).log_softmax(-1).to(dtype)
  torch.manual_seed(73)
  expected = (log_p.exp().argmax(-1) if readout == 'greedy'
              else samplers.sample_categorical(log_p.exp()))
  expected_rng = torch.get_rng_state()
  torch.manual_seed(73)
  with patch.object(model.backbone, 'forward_law', return_value=(log_p, None)), \
       patch('samplers.sample_posterior', side_effect=AssertionError('posterior')):
    sampler.step(model, state)
  assert torch.equal(state.xt, expected)
  assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize('injection_mode', ['add', 'replace'])
@pytest.mark.parametrize('carry_representation', ['expected_embedding', 'hidden_representation'])
def test_ablation_sampler_transitions(model, injection_mode, carry_representation):
  sampler = samplers.LAWCarryAblationSampler(
    carry_representation=carry_representation, injection_mode=injection_mode,
    temperature_schedule=parse_schedule('constant-1'))
  if carry_representation == 'hidden_representation':
    model.backbone.latent_ln = torch.nn.LayerNorm(32)
    model.backbone.uses_hidden_carry = True
  with patch('samplers.sample_posterior', wraps=samplers.sample_posterior) as posterior:
    tokens, metadata = samplers.run_sampler(sampler, model, 2, num_steps=4)
  assert tokens.shape == (2, 8) and metadata == {'nfe': 4}
  assert posterior.call_count == (3 if injection_mode == 'add' else 0)


@pytest.mark.parametrize('carry_representation', [
  'expected_embedding', 'hidden_representation'])
def test_ablation_sampler_uses_shared_temperature_schedule(
    model, carry_representation):
  sampler = samplers.LAWCarryAblationSampler(
    carry_representation=carry_representation, injection_mode='replace',
    noise_removal='greedy',
    temperature_schedule=parse_schedule('linear-4-1'))
  if carry_representation == 'hidden_representation':
    model.backbone.latent_ln = torch.nn.LayerNorm(32)
    model.backbone.uses_hidden_carry = True
  forward = model.backbone.forward_carry
  temperatures = []

  def record(*args, **kwargs):
    temperatures.append(
      (kwargs['temperature'], kwargs['carry_temperature']))
    return forward(*args, **kwargs)

  with patch.object(model.backbone, 'forward_carry', record):
    samplers.run_sampler(sampler, model, 2, num_steps=4)
  assert temperatures == [(4., 4.), (3., 3.), (2., 2.), (1., 1.)]


@pytest.mark.parametrize('readout', ['greedy', 'ancestral'])
def test_ablation_posterior_readout(model, readout):
  sampler = samplers.LAWCarryAblationSampler(
    carry_representation='expected_embedding', injection_mode='replace',
    decoder_mode='posterior', noise_removal=readout,
    temperature_schedule=parse_schedule('constant-1'))
  with patch('samplers.compute_posterior', wraps=samplers.compute_posterior) as compute, \
       patch('samplers.sample_posterior', wraps=samplers.sample_posterior) as sample:
    tokens, metadata = samplers.run_sampler(sampler, model, 2, num_steps=1)
  assert tokens.shape == (2, 8) and metadata == {'nfe': 1}
  assert (compute.call_count, sample.call_count) == (
    (1, 0) if readout == 'greedy' else (0, 1))


@CUDA
@pytest.mark.parametrize('readout', ['greedy', 'ancestral'])
def test_cuda_seeded_sampler_parity(readout):
  model = build(device='cuda')
  with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    _assert_cuda_sampler_parity(model, readout)


def _assert_cuda_sampler_parity(model, readout):
  canonical = samplers.LAWSampler(steps_policy='full', noise_removal=readout,
                                  temperature_schedule=parse_schedule('constant-1'))
  reference = samplers.LAWCarryAblationSampler(
    carry_representation='expected_embedding', injection_mode='replace',
    noise_removal=readout,
    temperature_schedule=parse_schedule('constant-1'))
  results = []
  for sampler in (reference, canonical):
    torch.manual_seed(91)
    tokens, metadata = samplers.run_sampler(
      sampler, model, 2, num_steps=3,
      prefix_tokens=torch.tensor([[1, 2, 3], [3, 2, 1]], device=model.device),
      prefix_lengths=torch.tensor([1, 3], device=model.device))
    results.append((tokens, metadata, torch.cuda.get_rng_state()))
  assert torch.equal(results[0][0], results[1][0])
  assert results[0][1] == results[1][1] == {'nfe': 3}
  assert torch.equal(results[0][2], results[1][2])
