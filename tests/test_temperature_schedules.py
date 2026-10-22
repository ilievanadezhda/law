import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

import samplers
from temperature_schedules import get_temperature_schedule, sampling_time


@pytest.mark.parametrize('spec, expected', [
  ('constant-2.5', (2.5, 2.5, 2.5)),
  ('last-20', (1.0, 1.0, 20.0)),
  ('linear-4-1', (4.0, 2.5, 1.0)),
  ('linear-.1-2', (0.1, 1.05, 2.0)),
  ('cosine-4-1', (4.0, 1 + 3 / math.sqrt(2), 1.0)),
  ('cosine-+2.0-.5', (2.0, .5 + 1.5 / math.sqrt(2), .5)),
])
def test_temperature_schedule_values(spec, expected):
  schedule = get_temperature_schedule(spec)
  assert tuple(schedule(t) for t in (1.0, .5, 0.0)) == pytest.approx(expected)


@pytest.mark.parametrize('family', ['linear', 'cosine'])
@pytest.mark.parametrize('bounds', [(3.0, 3.0), (.25, 4.0), (4.0, .25)])
def test_interpolated_schedules_are_finite_positive(family, bounds):
  high, low = bounds
  schedule = get_temperature_schedule(
    f'{family}-{high}-{low}')
  values = [schedule(index / 1000) for index in range(1001)]
  assert all(type(value) is float and math.isfinite(value) and value > 0
             for value in values)
  assert schedule(1.0) == high
  assert schedule(0.0) == low
  if high == low:
    assert values == [high] * len(values)


def test_unknown_temperature_schedule_kind():
  with pytest.raises(
      ValueError, match='unknown'):
    get_temperature_schedule('unknown-1')


def test_last_schedule_changes_only_at_exact_zero():
  schedule = get_temperature_schedule('last-.25')
  assert schedule(math.nextafter(0.0, 1.0)) == 1.0
  assert schedule(0.0) == .25


@pytest.mark.parametrize('spec', ['constant-0', 'linear-0-1', 'cosine-1-0',
                                  'constant-nan', 'last-inf', 'linear-1e999-1'])
def test_nonpositive_or_nonfinite_temperature_endpoints_rejected(spec):
  with pytest.raises(ValueError, match='temperature_schedule'):
    get_temperature_schedule(spec)


@pytest.mark.parametrize('num_steps, expected', [
  (1, [0.0]), (2, [1.0, 0.0]), (4, [1.0, 2 / 3, 1 / 3, 0.0])])
def test_normalized_sampling_time(num_steps, expected):
  actual = [sampling_time(index, num_steps)
            for index in range(num_steps)]
  assert actual == pytest.approx(expected)
  assert actual[-1] == 0.0
  assert all(left > right for left, right in zip(actual, actual[1:]))


def _recording_schedule(spec, calls):
  schedule = get_temperature_schedule(spec)

  def record(t):
    calls.append(t)
    return schedule(t)
  return record


class _DiffusionModel:
  def __init__(self, diffusion_type, carry=False):
    self.diffusion_type = diffusion_type
    self.carry = carry
    self.num_tokens = 2
    self.vocab_size = 3
    self.mask_index = 2
    self.device = torch.device('cpu')
    self.dtype = torch.float32
    self.temperatures = []

  def prior_sample(self, num_samples, num_tokens):
    return torch.zeros(num_samples, num_tokens, dtype=torch.long)

  def noise(self, t):
    return None, torch.full_like(t, .5)

  def _sigma_from_alphat(self, alpha):
    return alpha

  def forward(self, xt, sigma, context, return_aux=False):
    self.temperatures.append(context.temperature)
    log_p = torch.zeros(*xt.shape, self.vocab_size).log_softmax(-1)
    if return_aux:
      return log_p, {'latent': torch.zeros(*xt.shape, 2)}
    return log_p


@pytest.mark.parametrize('sampler_type', [samplers.AncestralSampler, samplers.LDDMSampler])
@pytest.mark.parametrize('diffusion_type', ['absorbing', 'uniform'])
@pytest.mark.parametrize('num_steps, expected', [(1, [1.0]), (4, [4.0, 3.0, 2.0, 1.0])])
def test_diffusion_sampler_temperature_integration(
    sampler_type, diffusion_type, num_steps, expected):
  times = []
  sampler = sampler_type(
    noise_removal='greedy', temperature_schedule=_recording_schedule(
      'linear-4-1', times))
  model = _DiffusionModel(diffusion_type, carry=sampler_type is samplers.LDDMSampler)
  with patch('samplers._decode_ancestral_update', side_effect=lambda *args, **kwargs: args[3]):
    samplers.run_sampler(sampler, model, 1, num_steps=num_steps)
  assert times == pytest.approx(
    [0.0] if num_steps == 1 else [1.0, 2 / 3, 1 / 3, 0.0])
  assert model.temperatures == expected


class _SFMBackbone:
  embed_dim = 2
  sphere_embed = SimpleNamespace(weight=torch.tensor([[1., 0.], [0., 1.], [1., 1.]]))


class _SFMModel:
  num_tokens = 2
  device = torch.device('cpu')
  backbone = _SFMBackbone()

  def __init__(self):
    self.temperatures = []

  def noise(self, t):
    return None, torch.as_tensor(t)

  def _sigma_from_alphat(self, alpha):
    return alpha

  def forward(self, xt, sigma, context):
    self.temperatures.append(context.temperature)
    return torch.zeros(*xt.shape[:2], 3).log_softmax(-1)


@pytest.mark.parametrize('invert_time_convention', [False, True])
@pytest.mark.parametrize('num_steps, expected', [(1, [1.0]), (4, [4.0, 3.0, 2.0, 1.0])])
def test_sfm_temperature_uses_evaluation_order(
    invert_time_convention, num_steps, expected):
  times = []
  sampler = samplers.SFMSampler(
    noise_removal='greedy', velocity='exact', use_float64=False,
    slerp_float64=False, eps=1e-5,
    temperature_schedule=_recording_schedule('linear-4-1', times),
    p_nucleus=1.0, top_k=-1, top_k_velocity=-1,
    invert_time_convention=invert_time_convention)
  model = _SFMModel()
  with patch.object(sampler, '_compute_velocity', side_effect=lambda x, E, log_p: x * 0), \
       patch.object(sampler, '_get_step_size', return_value=0.0):
    samplers.run_sampler(sampler, model, 1, num_steps=num_steps)
  assert times == pytest.approx(
    [0.0] if num_steps == 1 else [1.0, 2 / 3, 1 / 3, 0.0])
  assert model.temperatures == expected


class _FLMModel:
  num_tokens = 2
  vocab_size = 3
  device = torch.device('cpu')
  dtype = torch.float32

  def __init__(self):
    self.model_times = []
    self.contexts = []

  def _alpha_t_to_gamma(self, t):
    return t

  def forward(self, xt, sigma, context):
    self.model_times.append(float(sigma[0]))
    self.contexts.append(context)
    logits = torch.tensor([0., 1., 2.]).expand(*xt.shape[:2], -1)
    return logits.log_softmax(-1) if context is None else logits


@pytest.mark.parametrize('num_steps, expected', [(1, [1.0]), (4, [4.0, 3.0, 2.0, 1.0])])
def test_flm_temperature_is_not_its_increasing_model_clock(num_steps, expected):
  times = []
  sampler = samplers.FLMEulerSampler(
    temperature_schedule=_recording_schedule('linear-4-1', times))
  model = _FLMModel()
  samplers.run_sampler(sampler, model, 1, num_steps=num_steps)
  assert times == pytest.approx(
    [0.0] if num_steps == 1 else [1.0, 2 / 3, 1 / 3, 0.0])
  assert model.model_times == sorted(model.model_times)
  actual = [1.0 if context is None else expected[index]
            for index, context in enumerate(model.contexts)]
  assert actual == expected


def test_canonical_sampler_configs_use_noop_schedule():
  root = Path(__file__).resolve().parents[1]
  for name in ('ancestral', 'sfm', 'flm_euler', 'law',
               'law_carry_ablation', 'lddm'):
    config = OmegaConf.load(root / 'configs' / 'sampler' / f'{name}.yaml')
    assert config.temperature_schedule == 'constant-1.0'
    assert 'temperature' not in config


@pytest.mark.parametrize('predictor, algo_name, sampler_type', [
  ('ancestral', 'mdlm', samplers.AncestralSampler),
  ('lddm', 'lddm_u', samplers.LDDMSampler),
  ('sfm', 'sfm', samplers.SFMSampler),
  ('flm_euler', 'flm', samplers.FLMEulerSampler),
  ('law', 'law', samplers.LAWSampler),
  ('law-carry-ablation', 'law_carry_ablation', samplers.LAWCarryAblationSampler),
])
def test_dispatch_parses_one_shared_schedule(predictor, algo_name, sampler_type):
  sampler_config = SimpleNamespace(
    predictor=predictor, temperature_schedule='cosine-4-1',
    use_float64=False, p_nucleus=1.0, steps_policy='full',
    noise_removal='greedy', velocity='exact', top_k=-1,
    top_k_velocity=-1, decoder_mode='raw')
  algo_config = SimpleNamespace(
    name=algo_name, slerp_precision='float32', eps=1e-5,
    invert_time_convention=False, carry_representation='expected_embedding',
    injection_mode='replace')
  config = SimpleNamespace(sampler=sampler_config, algo=algo_config)
  with patch('samplers.get_temperature_schedule',
             wraps=get_temperature_schedule) as parse:
    sampler = samplers.get_sampler(config)
  assert isinstance(sampler, sampler_type)
  parse.assert_called_once_with('cosine-4-1')
  assert sampler.temperature_schedule(1.0) == 4.0
  assert sampler.temperature_schedule(0.0) == 1.0


@pytest.mark.parametrize('predictor, algo_name, sampler_type', [
  ('ar', 'ar', samplers.ARSampler),
  ('mixing', 'duo-mixing', samplers.MixingDUOSampler),
])
def test_ar_and_mixing_dispatch_remain_scalar_and_unscheduled(
    predictor, algo_name, sampler_type):
  sampler_config = SimpleNamespace(
    predictor=predictor, temperature=.7, use_float64=False,
    use_kv_cache=False, greedy=True, early_stopping=False,
    steps_policy='full', noise_removal='greedy', MC='probs')
  config = SimpleNamespace(
    sampler=sampler_config, algo=SimpleNamespace(name=algo_name))
  with patch('samplers.get_temperature_schedule',
             side_effect=AssertionError('temperature schedule parsed')):
    sampler = samplers.get_sampler(config)
  assert isinstance(sampler, sampler_type)
  if predictor == 'ar':
    assert sampler.temperature == .7
