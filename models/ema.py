import torch


class ExponentialMovingAverage:
  """
  Maintains (exponential) moving average of a set of parameters.
  """

  def __init__(self, parameters, decay, use_num_updates=True):
    """
    Args:
        parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
        decay: The exponential decay.
        use_num_updates: Whether to use number of updates when computing
            averages.
    """
    if decay < 0.0 or decay > 1.0:
      raise ValueError('Decay must be between 0 and 1')
    self.decay = decay
    self.num_updates = 0 if use_num_updates else None
    self.shadow_params = [p.clone().detach()
                          for p in parameters if p.requires_grad]
    self._parameter_shapes = [p.shape for p in self.shadow_params]
    self.collected_params = []

  @staticmethod
  def _validate_parameters(parameters, expected_shapes, name):
    if len(parameters) != len(expected_shapes):
      raise ValueError(f'{name} count mismatch: expected {len(expected_shapes)}, '
                       f'got {len(parameters)}')
    for index, (parameter, shape) in enumerate(zip(parameters, expected_shapes)):
      if parameter.shape != shape:
        raise ValueError(f'{name} shape mismatch at index {index}: '
                         f'expected {tuple(shape)}, got {tuple(parameter.shape)}')

  def move_shadow_params_to_device(self, device):
    self.shadow_params = [i.to(device) for i in self.shadow_params]

  def update(self, parameters):
    """
    Update currently maintained parameters.

    Call this every time the parameters are updated, such as the result of
    the `optimizer.step()` call.

    Args:
        parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
    """
    parameters = [p for p in parameters if p.requires_grad]
    self._validate_parameters(self.shadow_params, [p.shape for p in parameters],
                              'EMA shadow parameters')
    decay = self.decay
    if self.num_updates is not None:
      self.num_updates += 1
      decay = min(decay, (1 + self.num_updates) /
                  (10 + self.num_updates))
    one_minus_decay = 1.0 - decay
    with torch.no_grad():
      for s_param, param in zip(self.shadow_params, parameters):
        s_param.sub_(one_minus_decay * (s_param - param))

  def copy_to(self, parameters):
    """
    Copy current parameters into given collection of parameters.

    Args:
        parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
    """
    parameters = [p for p in parameters if p.requires_grad]
    self._validate_parameters(self.shadow_params, [p.shape for p in parameters],
                              'EMA shadow parameters')
    for s_param, param in zip(self.shadow_params, parameters):
      if param.requires_grad:
        param.data.copy_(s_param.data)

  def store(self, parameters):
    """
    Save the current parameters for restoring later.

    Args:
        parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
    """
    self.collected_params = [param.clone() for param in parameters]

  def restore(self, parameters):
    """
    Restore the parameters stored with the `store` method.
    Useful to validate the model with EMA parameters without affecting the
    original optimization process. Store the parameters before the
    `copy_to` method. After validation (or model saving), use this to
    restore the former parameters.

    Args:
        parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
    """
    parameters = list(parameters)
    self._validate_parameters(self.collected_params, [p.shape for p in parameters],
                              'EMA collected parameters')
    for c_param, param in zip(self.collected_params, parameters):
      param.data.copy_(c_param.data)

  def state_dict(self):
    return dict(decay=self.decay,
                num_updates=self.num_updates,
                shadow_params=self.shadow_params)

  def load_state_dict(self, state_dict):
    self._validate_parameters(state_dict['shadow_params'], self._parameter_shapes,
                              'checkpoint EMA shadow parameters')
    self.decay = state_dict['decay']
    self.num_updates = state_dict['num_updates']
    self.shadow_params = state_dict['shadow_params']
