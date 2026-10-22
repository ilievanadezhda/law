import pytest
import torch

from models.ema import ExponentialMovingAverage


def parameters():
  return [torch.nn.Parameter(torch.tensor([1., 2.])),
          torch.nn.Parameter(torch.tensor([3., 4.]))]


@pytest.mark.parametrize('method', ['update', 'copy_to', 'restore'])
@pytest.mark.parametrize('mismatch', ['short', 'long', 'shape'])
def test_invalid_saved_parameters_fail_before_mutation(method, mismatch):
  params = parameters()
  ema = ExponentialMovingAverage(iter(params), decay=.9)
  ema.store(iter(params))
  saved = ema.collected_params if method == 'restore' else ema.shadow_params
  if mismatch == 'short':
    saved.pop()
  elif mismatch == 'long':
    saved.append(torch.zeros_like(saved[-1]))
  else:
    saved[-1] = torch.zeros(1)  # Broadcasting must not hide a shape mismatch.
  with torch.no_grad():
    for param in params:
      param.add_(10)
  original_params = [p.clone() for p in params]
  original_saved = [p.clone() for p in saved]
  error = 'shape mismatch at index 1' if mismatch == 'shape' else 'count mismatch'
  with pytest.raises(ValueError, match=error):
    getattr(ema, method)(iter(params))
  assert ema.num_updates == 0
  for actual, expected in zip(params + saved, original_params + original_saved):
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('method', ['update', 'copy_to', 'restore'])
@pytest.mark.parametrize('count', [1, 3])
def test_parameter_count_mismatch(method, count):
  params = parameters()
  ema = ExponentialMovingAverage(iter(params), decay=.9)
  ema.store(iter(params))
  supplied = (params + [torch.nn.Parameter(torch.zeros(2))])[:count]
  with pytest.raises(ValueError, match='count mismatch'):
    getattr(ema, method)(iter(supplied))


@pytest.mark.parametrize('mismatch', ['short', 'long', 'shape'])
def test_load_validates_construction_structure(mismatch):
  ema = ExponentialMovingAverage(iter(parameters()), decay=.9)
  shadows = [torch.zeros(2), torch.zeros(2)]
  if mismatch == 'short':
    shadows.pop()
  elif mismatch == 'long':
    shadows.append(torch.zeros(2))
  else:
    shadows[-1] = torch.zeros(1)
  # The construction structure must survive even if current shadows are changed.
  ema.shadow_params = shadows
  state = dict(decay=.5, num_updates=7, shadow_params=shadows)
  error = 'shape mismatch at index 1' if mismatch == 'shape' else 'count mismatch'
  with pytest.raises(ValueError, match=f'checkpoint EMA shadow parameters {error}'):
    ema.load_state_dict(state)
  assert ema.decay == .9
  assert ema.num_updates == 0


@pytest.mark.parametrize('num_updates', [None, 7])
def test_old_style_state_and_update_copy_restore(num_updates):
  params = parameters()
  frozen = torch.nn.Parameter(torch.tensor([9.]), requires_grad=False)
  params.insert(1, frozen)
  ema = ExponentialMovingAverage(iter(params), decay=.9)
  state = dict(decay=.5, num_updates=num_updates,
               shadow_params=[torch.zeros(2), torch.zeros(2)])
  ema.load_state_dict(state)
  assert set(ema.state_dict()) == {'decay', 'num_updates', 'shadow_params'}
  assert ema.decay == .5
  assert ema.num_updates == num_updates
  ema.update(iter(params))
  assert ema.num_updates == (None if num_updates is None else num_updates + 1)
  torch.testing.assert_close(ema.shadow_params[0], torch.tensor([.5, 1.]))
  torch.testing.assert_close(ema.shadow_params[1], torch.tensor([1.5, 2.]))

  original = [p.clone() for p in params]
  ema.store(iter(params))
  ema.copy_to(iter(params))
  torch.testing.assert_close(params[0], ema.shadow_params[0])
  torch.testing.assert_close(params[2], ema.shadow_params[1])
  torch.testing.assert_close(frozen, original[1])
  with torch.no_grad():
    frozen.add_(1)
  ema.restore(iter(params))
  for actual, expected in zip(params, original):
    torch.testing.assert_close(actual, expected)
