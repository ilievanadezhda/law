import itertools

from dataclass_patch import dataclass

import hydra.utils
import lightning as L
import torch
import transformers

import dataloader
import metrics
import models
import noise_schedules
import samplers
import utils


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  prior_loss: torch.FloatTensor
  num_tokens: torch.FloatTensor


@dataclass
class TrainingContext:
  kv_cache: bool = False


@dataclass
class FLMTrainingContext(TrainingContext):
  sigma_prime: torch.Tensor | None = None
  skip_softmax: bool = False


@dataclass
class CANDITrainingContext(TrainingContext):
  reveal_mask: torch.Tensor | None = None
  continuous_noise: torch.Tensor | None = None
  is_embed: bool = False
  embedding_cache: torch.Tensor | None = None
  # Original token IDs — needed by _process_model_output in
  # embed mode where xt.argmax(-1) is over the embedding dim,
  # not the vocabulary.
  x0_tokens: torch.Tensor | None = None


class TrainerBase(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer,
    vocab_size=None):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self._last_sample_step = None
    if hasattr(self.config.algo, 'loss_type'):
      self.loss_type = config.algo.loss_type
    self.tokenizer = tokenizer
    if vocab_size is None:
      self.vocab_size = len(self.tokenizer)
    else:
      self.vocab_size = vocab_size
    self.sampler = samplers.get_sampler(self.config)
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.parameterization = self.config.algo.parameterization
    if self.config.model.type == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.model.type == 'dit_loop':
      self.backbone = models.dit_loop.LoopDiT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.model.type == 'flm-dit':
      self.backbone = models.flm_dit.FLMDIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.model.type == 'candi-dit':
      self.backbone = models.candi_dit.ContDIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.model.type == 'sphere-dit':
      self.backbone = models.sphere_dit.SphereDiT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.model.type == 'sphere-arch':
      self.backbone = models.sphere_arch.SphereArch(
        self.config, vocab_size=self.vocab_size)
    else:
      raise ValueError(self.config.model.type)

    self.diffusion_type = config.algo.diffusion_type
    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length
    self.softplus = torch.nn.Softplus()
    self.p_nucleus = self.config.sampler.p_nucleus
    self.noise = noise_schedules.get_noise(config)

    self.metrics = metrics.Metrics(
      gen_ppl_eval_model_name_or_path=\
        self.config.eval.gen_ppl_eval_model_name_or_path,
      eval_ppl_batch_size=\
        self.config.eval.perplexity_batch_size)

    self._prepare_ema()
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.algo.time_conditioning
    if config.neg_infinity_mode == 'large-finite':
      self.neg_infinity = -1000000.0
    elif config.neg_infinity_mode == 'true-inf':
      self.neg_infinity = - float('inf')
    else:
      if config.neg_infinity_mode.startswith('value-'):
        self.neg_infinity = float(
          config.neg_infinity_mode.split('-')[1]) * -1
      else:
        raise ValueError(config.neg_infinity_mode)
    self.fast_forward_epochs = None
    self.fast_forward_batches = None

  def _process_sigma(self, sigma, context=None):
    raise NotImplementedError

  def _process_model_output(
      self,
      model_output,
      xt,
      sigma,
      context=None):
    raise NotImplementedError

  def q_xt(self, x, alpha_t, use_pure_noise,
           valid_tokens=None):
    raise NotImplementedError

  def _process_model_input(self, x0, valid_tokens):
    raise NotImplementedError

  def nll(self, input_tokens, output_tokens, context,
          current_accumulation_step=None, train_mode=False,
          valid_tokens=None):
    raise NotImplementedError

  @property
  def ctx_cached_len(self) -> int:
    return self.backbone.ctx_cached_len

  def reset_kv_cache(self):
    self.backbone.reset_kv_cache()

  def _prepare_ema(self):
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        self._get_parameters(),
        decay=self.config.training.ema)
    else:
      self.ema = None

  def _validate_configuration(self):
    if self.config.algo.parameterization == 'ar':
      assert not self.config.algo.time_conditioning
      assert self.config.prior.type == 'none'

    # if self.parameterization in {'score', 'mean'}:
    #   assert self.time_conditioning
    if self.T > 0:
      assert self.parameterization != 'score'

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.metrics.to(*args, **kwargs)
    return self

  def _get_parameters(self):
    return itertools.chain(self.backbone.parameters(),
                           self.noise.parameters())

  def _eval_mode(self):
    if self.ema:
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.backbone.eval()
    self.noise.eval()

  def _train_mode(self):
    if self.ema:
      self.ema.restore(self._get_parameters())
    self.backbone.train()
    self.noise.train()

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    # When `compile_backbone` is enabled, `self.backbone` is an
    # `OptimizedModule`, so the auto-collected state_dict has keys like
    # `backbone._orig_mod.<param>`. Strip the infix so the on-disk format
    # is identical regardless of whether compile was on at save time —
    # compiled and uncompiled checkpoints stay interchangeable.
    sd = checkpoint.get('state_dict')
    if sd and any('._orig_mod.' in k for k in sd):
      checkpoint['state_dict'] = {
        k.replace('._orig_mod.', '.'): v for k, v in sd.items()}
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed']
    # is 1 iteration behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps,
    # not the number of local steps, so we don't multiply with
    # self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_fit_start(self):
    super().on_fit_start()
    flag = self.config.compile_backbone
    if not flag:
      return
    self.backbone = torch.compile(self.backbone, mode="default")

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(self._get_parameters())

  def forward(self, *, x0=None, xt=None, sigma=None, context=None,
              return_aux=False):
    if sigma is not None:
      sigma = self._process_sigma(sigma, context=context)
    aux = None
    with torch.amp.autocast('cuda', dtype=torch.float32):
      if return_aux:
        model_output, aux = self.backbone(
          x0, xt, sigma, context, return_aux=True)
      else:
        model_output = self.backbone(x0, xt, sigma, context)

    if (context is not None
        and 'temperature' in context
        and context.temperature != 1.0):
      model_output = model_output / context.temperature

    processed = self._process_model_output(
      model_output=model_output,
      xt=xt,
      sigma=sigma,
      context=context)
    if return_aux:
      return processed, aux
    return processed

  def on_train_epoch_start(self):
    self.metrics.reset()
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  def training_step(self, batch, batch_idx):
    current_accumulation_step = (
      batch_idx % self.trainer.accumulate_grad_batches)
    losses = self._loss(batch['input_ids'],
                        batch['attention_mask'],
                        current_accumulation_step=current_accumulation_step,
                        train_mode=True)
    self.metrics.update_train(losses.nlls, losses.prior_loss,
                              losses.num_tokens)
    self.log(name='trainer/loss',
             value=losses.loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return losses.loss

  def on_train_epoch_end(self):
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k, value=v.compute(), on_step=False,
               on_epoch=True, sync_dist=True)

  def on_validation_epoch_start(self):
    self.metrics.reset()
    self._eval_mode()
    assert self.metrics.valid_nlls.nll.mean_value == 0
    assert self.metrics.valid_nlls.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    del batch_idx
    losses = self._loss(batch['input_ids'],
                        batch['attention_mask'])
    self.metrics.update_valid(losses.nlls, losses.prior_loss,
                              losses.num_tokens)
    return losses.loss

  def _should_generate_samples(self):
    if not self.config.eval.generate_samples:
      return False
    if (self.trainer.sanity_checking
        and not self.config.eval.compute_perplexity_on_sanity):
      return False
    every = getattr(self.config.eval,
                    'generate_samples_every_n_steps', 0) or 0
    if every and self._last_sample_step is not None \
       and self.global_step - self._last_sample_step < every - 10:
      return False
    self._last_sample_step = self.global_step
    return True

  def on_validation_epoch_end(self):
    for k, v in self.metrics.valid_nlls.items():
      self.log(name=k,  value=v.compute(), on_step=False,
               on_epoch=True, sync_dist=True)
    if self._should_generate_samples():
      samples, text_samples = None, None
      want_corr = getattr(self.config.eval, 'compute_hidden_corr', False)
      corr_batches = []
      for _ in range(
        self.config.sampler.num_sample_batches):
        if want_corr:
          with utils.capture_last_hidden(self.backbone) as box:
            samples, _ = self.generate_samples(
              num_samples=self.config.loader.eval_batch_size)
          corr_batches.append(utils.token_correlation(box['hidden']))
        else:
          samples, _ = self.generate_samples(
            num_samples=self.config.loader.eval_batch_size)

        self.metrics.record_entropy(samples)
        # Decode the samples to be re-tokenized by eval model
        text_samples = self.tokenizer.batch_decode(samples)
        if self.config.eval.compute_generative_perplexity:
          self.metrics.record_generative_perplexity(
            text_samples, self.num_tokens, self.device)
      corr_stats = utils.average_stats(corr_batches)
      if corr_stats is not None:
        for key in ('mean_corr', 'mean_abs_corr', 'mean_cos',
                    'hidden_norm_mean'):
          self.log(name=f'val/hidden_{key}', value=corr_stats[key],
                   on_step=False, on_epoch=True, sync_dist=True)
      if samples is not None:
        self.log('val/sample_entropy',
                 self.metrics.sample_entropy.compute(),
                 on_epoch=True, on_step=False, sync_dist=True)
      if text_samples is not None:
        if self.trainer.global_rank == 0 and hasattr(
          self.trainer.logger, 'log_table'):
          # Log the last generated samples
          text_samples = text_samples[
            : self.config.sampler.num_sample_log]
          self.trainer.logger.log_table(
            key=f'samples@global_step{self.global_step}',
            columns=['Generated Samples'],
            data=[[s] for s in text_samples])
        if self.config.eval.compute_generative_perplexity:
          self.log('val/gen_ppl',
                   self.metrics.gen_ppl.compute(),
                   on_epoch=True,
                   on_step=False,
                   sync_dist=True)
    self._train_mode()

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
      (p for p in self._get_parameters() if p.requires_grad),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {'scheduler': scheduler,
                      'interval': 'step',
                      'monitor': 'val/loss',
                      'name': 'trainer/lr'}
    return [optimizer], [scheduler_dict]

  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None,
                       eps=1e-5, prefix_tokens=None,
                       prefix_lengths=None):
    """Generate samples from the model."""
    return samplers.run_sampler(
      self.sampler, self, num_samples,
      num_steps=num_steps, eps=eps, prefix_tokens=prefix_tokens,
      prefix_lengths=prefix_lengths)

  def restore_model_and_sample(self, num_steps, eps=1e-5,
                               prefix_tokens=None,
                               prefix_lengths=None):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    self._eval_mode()
    samples, metadata = self.generate_samples(
      num_samples=self.config.loader.eval_batch_size,
      num_steps=num_steps,
      eps=eps,
      prefix_tokens=prefix_tokens,
      prefix_lengths=prefix_lengths)
    self._train_mode()
    return samples, metadata

  def _loss(self, x0, valid_tokens,
            context=None,
            current_accumulation_step=None,
            train_mode=False):
    (input_tokens, output_tokens,
     valid_tokens) = self._process_model_input(
       x0, valid_tokens)
    if context is None:
      context = TrainingContext()
    loss, t = self.nll(input_tokens, output_tokens,
                        context,
                        current_accumulation_step, train_mode,
                        valid_tokens=valid_tokens)
    assert loss.ndim == 2
    
    masked_loss = loss * valid_tokens  # (B, L)
    per_sample_loss_sum = masked_loss.sum(-1)  # (B,)
    per_sample_num_tokens = valid_tokens.sum(-1)  # (B,)

    per_sample_loss = (per_sample_loss_sum
      / per_sample_num_tokens.clamp(min=1))  # (B,)
    total_loss_sum = per_sample_loss_sum.sum()  # scalar
    total_num_tokens = per_sample_num_tokens.sum()  # scalar
    mean_token_loss = (total_loss_sum
      / total_num_tokens.clamp(min=1)) # scalar

    if (train_mode and t is not None
        and self.config.noise.adaptive):
      t_all = self.all_gather(t.detach()).reshape(-1)
      loss_all = self.all_gather(
        per_sample_loss.detach()).reshape(-1)
      self.noise.record_time_loss_pair(
        t_all, loss_all, self.global_step)

    return Loss(loss=mean_token_loss,
                nlls=total_loss_sum,
                prior_loss=0.0,
                num_tokens=total_num_tokens)


class Diffusion(TrainerBase):
  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.config.sampler.noise_removal in {
      'none', 'ancestral', 'greedy'}
    if self.config.sampler.noise_removal == 'greedy':
      assert self.config.sampler.predictor != 'analytic'
      assert self.parameterization in {'mean', 'subs'}

  def _process_model_input(self, x0, valid_tokens):
    return x0, None, valid_tokens

  def _process_sigma(self, sigma, context=None):
    del context
    if sigma.ndim == 1:
      sigma = sigma.unsqueeze(-1)
    assert sigma.ndim == 2
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0:
      sigma = sigma.unsqueeze(0)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def _sample_t(self, n, accum_step):
    if accum_step is not None:
      # During training
      batch_dim = n
      n = self.config.loader.global_batch_size
    _eps_t = torch.rand(n, device=self.device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=self.device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if accum_step is not None:
      t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
      t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
      t = t.chunk(self.trainer.accumulate_grad_batches)[
        accum_step]
      # corner case for the last datapoint
      t = t[:batch_dim]
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)
    return t

  def _sigma_from_alphat(self, alpha_t):
    return -torch.log(alpha_t)

  def nll_per_token(self, model_output, xt, x0, alpha_t,
                    dalpha_t, low_var=False, context=None,
                    train_mode=False):
    raise NotImplementedError

  def _use_pure_noise(self, train_mode, context=None):
    del train_mode, context
    return False

  def nll(self, x0, output_tokens, context,
          current_accumulation_step=None, train_mode=False,
          valid_tokens=None):
    del output_tokens
    t = self._sample_t(x0.shape[0],
                       current_accumulation_step)
    assert t.shape[0] == x0.shape[0]
    use_pure_noise = self._use_pure_noise(
      train_mode=train_mode, context=context)
    if use_pure_noise:
      t = torch.ones_like(t)

    dalpha_t, alpha_t = self.noise(t)
    alpha_t = alpha_t.unsqueeze(-1)
    dalpha_t = dalpha_t.unsqueeze(-1)
    assert alpha_t.ndim == 2
    assert dalpha_t.ndim == 2
    sigma = self._sigma_from_alphat(alpha_t)

    xt = self.q_xt(x0, alpha_t, use_pure_noise=use_pure_noise,
                   valid_tokens=valid_tokens)
    log_x_theta = self.forward(
      x0=x0, xt=xt, sigma=sigma, context=context)
    utils.print_nans(log_x_theta, 'model_output')
    loss = self.nll_per_token(
      log_x_theta=log_x_theta,
      xt=xt,
      x0=x0,
      alpha_t=alpha_t,
      dalpha_t=dalpha_t,
      low_var=train_mode and self.loss_type == 'low_var',
      context=context,
      train_mode=train_mode)
    return loss, t


class AbsorbingState(Diffusion):
  def __init__(self, config, tokenizer):
    # NOTE: Ideally, we should do
    # vocab_size = len(tokenizer), so that we account
    # for the special tokens added in dataloader.py.
    # But we use tokenizer.vocab_size so as to be
    # consistent with the prior checkpoints.
    vocab_size = tokenizer.vocab_size
    if (not hasattr(tokenizer, 'mask_token')
        or tokenizer.mask_token is None):
      self.mask_index = vocab_size
      vocab_size += 1
    else:
      self.mask_index = tokenizer.mask_token_id
    super().__init__(config, tokenizer,
                     vocab_size=vocab_size)
    self.save_hyperparameters()

  def _validate_configuration(self):
    super()._validate_configuration()
    #if self.parameterization in {'score', 'mean'}:
    #  assert self.time_conditioning
    assert not (self.parameterization == 'mean'
                and self.T == 0)
    if self.T > 0:
      assert self.parameterization in {'mean', 'subs'}

  def q_xt(self, x, alpha_t, use_pure_noise,
           valid_tokens=None):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      alpha_t: float torch.Tensor with shape (batch_size, 1).
      valid_tokens: optional bool/int torch.Tensor with shape
          (batch_size, diffusion_model_input_length). If provided,
          only positions where valid_tokens == 1 are noised;
          prompt positions (valid_tokens == 0) stay clean.
    """
    if use_pure_noise:
      xt = torch.full_like(x, self.mask_index)
    else:
      move_indices = torch.rand(
        * x.shape, device=x.device) < 1 - alpha_t
      xt = torch.where(move_indices, self.mask_index, x)
    if valid_tokens is not None:
      xt = torch.where(valid_tokens.bool(), xt, x)
    return xt

  def prior_sample(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64, device=self.device)


class UniformState(Diffusion):
  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    self.save_hyperparameters()

  def _validate_configuration(self):
    super()._validate_configuration()
    # assert self.time_conditioning
    assert self.parameterization == 'mean'
    if self.config.algo.name not in {'distillation', 'duo-mixing'}:
      assert self.T == 0

  def q_xt(self, x, alpha_t, use_pure_noise,
           valid_tokens=None):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      alpha_t: float torch.Tensor with shape
        (batch_size, 1).
      valid_tokens: optional bool/int torch.Tensor with shape
          (batch_size, diffusion_model_input_length). If provided,
          only positions where valid_tokens == 1 are noised;
          prompt positions (valid_tokens == 0) stay clean.
    """
    uniform_tensor = torch.randint(
      0, self.vocab_size, x.shape, device=x.device)
    if use_pure_noise:
      xt = uniform_tensor
    else:
      move_indices = torch.rand(
        *x.shape, device=x.device) < 1 - alpha_t
      xt = torch.where(move_indices, uniform_tensor, x)
    if valid_tokens is not None:
      xt = torch.where(valid_tokens.bool(), xt, x)
    return xt

  def prior_sample(self, *batch_dims):
    return torch.randint(
      0, self.vocab_size, batch_dims, dtype=torch.int64,
      device=self.device)
