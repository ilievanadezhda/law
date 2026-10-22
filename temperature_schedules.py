"""Temperature schedules used by samplers."""

import math


class TemperatureSchedule:
  def __call__(self, t):
    raise NotImplementedError


def _positive_finite(value):
  value = float(value)
  if not math.isfinite(value) or value <= 0:
    raise ValueError('temperature_schedule endpoints must be finite and positive')
  return value


class Constant(TemperatureSchedule):
  def __init__(self, temperature):
    self.temperature = _positive_finite(temperature)

  def __call__(self, t):
    return self.temperature


class Last(TemperatureSchedule):
  def __init__(self, temperature):
    self.temperature = _positive_finite(temperature)

  def __call__(self, t):
    return self.temperature if t == 0.0 else 1.0


class Linear(TemperatureSchedule):
  def __init__(self, high, low):
    self.high, self.low = _positive_finite(high), _positive_finite(low)

  def __call__(self, t):
    return self.low + t * (self.high - self.low)


class Cosine(TemperatureSchedule):
  def __init__(self, high, low):
    self.high, self.low = _positive_finite(high), _positive_finite(low)

  def __call__(self, t):
    return self.low + math.sin(math.pi * t / 2) * (self.high - self.low)


def get_temperature_schedule(spec):
  kind, *values = spec.split("-")
  if kind == 'constant':
    return Constant(*values)
  if kind == 'last':
    return Last(*values)
  if kind == 'linear':
    return Linear(*values)
  if kind == 'cosine':
    return Cosine(*values)
  raise ValueError(kind)


def sampling_time(step_idx, num_steps):
  return 0.0 if num_steps == 1 else 1 - step_idx / (num_steps - 1)
