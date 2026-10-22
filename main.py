import gc
import glob
import json
import os
import time

import fsspec
import hydra
import lightning as L
import matplotlib.pyplot as plt
from lightning.fabric import Fabric
import numpy as np
import omegaconf
from pathlib import Path
import rich.syntax
import rich.tree
import torch
from tqdm import tqdm, trange

import algo
from callbacks.throughput import ThroughputCallback
import dataloader
import sandbox_gsm8k
import utils


omegaconf.OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
  'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)

# Log to wandb offline by default; set WANDB_MODE=online to sync.
os.environ.setdefault('WANDB_MODE', 'offline')


def _load_from_checkpoint(diffusion_model, config, tokenizer):
  # if 'hf' in config.algo.backbone:
  #   return diffusion_model(
  #     config, tokenizer=tokenizer).to('cuda')
  
  return diffusion_model.load_from_checkpoint(
    config.eval.checkpoint_path,
    tokenizer=tokenizer,
    config=config,
    strict=config.eval.strict_loading)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=64):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    # Read from the dataset directly to avoid spawning DataLoader workers;
    # fall back to iteration for streaming datasets.
    dataset = dl.dataset
    try:
      example = dataset[0]
    except (TypeError, KeyError, NotImplementedError):
      example = next(iter(dataset))
    input_ids = example['input_ids']
    print('Example input_ids.shape',
          getattr(input_ids, 'shape', f'(len={len(input_ids)},)'))
    first = input_ids[:k]
    last = input_ids[-k:]
    try:
      print(f'First {k} tokens:', tokenizer.decode(first))
      print('ids:', first)
      print(f'Last {k} tokens:', tokenizer.decode(last))
      print('ids:', last)
    except:
      print('First tokens:', first)
      print('Last tokens:', last)


@torch.no_grad
def _generate_samples(diffusion_model, config, logger,
                      tokenizer):
  logger.info('Starting Sample Eval.')
  fabric = Fabric(accelerator=config.trainer.accelerator, 
                  devices=config.trainer.devices, 
                  num_nodes=config.trainer.num_nodes)
  fabric.launch()
  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  model.metrics.gen_ppl.reset()
  model.metrics.sample_entropy.reset()
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  num_devices = config.trainer.num_nodes * config.trainer.devices
  if config.sampler.num_sample_batches % num_devices != 0:
    raise ValueError(
      f'Num. batches ({config.sampler.num_sample_batches}) '
      f'not divided by num. devices ({num_devices})')

  num_batch_per_device = (config.sampler.num_sample_batches
                          // num_devices)
  total_num_samples = (config.sampler.num_sample_batches
                      * config.loader.eval_batch_size)

  fname_vars = dict(
    predictor=config.sampler.predictor,
    noise_removal=config.sampler.noise_removal,
    use_float64=config.sampler.use_float64,
    p_nucleus=config.sampler.p_nucleus,
    num_samples=total_num_samples,
    num_sampling_steps=config.sampler.steps,
    ckpt_hash=utils.short_hash(config.eval.checkpoint_path))
  samples_fname = utils.vars_to_fname(**fname_vars) + ".json"

  model.to(fabric.device)
  if fabric.global_rank == 0:
    logger.info(f'Sampling {total_num_samples} samples.')
  all_text_samples = []
  all_np_samples = []
  total_nfe = 0
  corr_batches = []
  want_corr = config.eval.compute_hidden_corr
  for _ in trange(num_batch_per_device, desc=f'Sampling '
                          f'({config.sampler.steps} steps)',
                          disable=fabric.global_rank != 0):
    if want_corr:
      # The hook keeps the last pass, i.e. the state the tokens are drawn from.
      with utils.capture_last_hidden(model.backbone) as box:
        samples, metadata = model.restore_model_and_sample(
          num_steps=config.sampler.steps)
      corr_batches.append(utils.token_correlation(box['hidden']))
    else:
      samples, metadata = model.restore_model_and_sample(
        num_steps=config.sampler.steps)
    total_nfe += metadata['nfe']
    samples = fabric.all_gather(samples)
    if fabric.world_size > 1:
      samples = samples.flatten(0, 1)

    np_samples = samples.cpu().numpy()
    text_samples = model.tokenizer.batch_decode(samples)
    model.metrics.record_entropy(samples)
    all_text_samples.extend(text_samples)
    all_np_samples.append(np_samples)

  fabric.barrier()
  if fabric.global_rank == 0:
    logger.info("Evaluating generative perplexity...")
    all_np_samples = np.concatenate(all_np_samples)
    # Evaluate with retokenize and first chunk only (orig)
    model.metrics.record_generative_perplexity(
      all_text_samples, 
      config.model.length,
      retokenize=True,
      first_chunk_only=True,
      device=model.device)
    gen_ppl_first_chunk_retok = model.metrics.gen_ppl.compute().item()
    model.metrics.gen_ppl.reset()
    gen_ppl_all_no_retok = None
    entropy = model.metrics.sample_entropy.compute().item()

    avg_nfe = total_nfe / max(num_batch_per_device, 1)
    logger.info('Generative perplexity (retokenize, first '
               f'chunk only): {gen_ppl_first_chunk_retok:.5f}')
    logger.info(f'Sample entropy: {entropy:.5f}')
    logger.info(f'Average NFE per batch: {avg_nfe:.1f}')
    hidden_corr_stats = utils.average_stats(corr_batches)
    if hidden_corr_stats is not None:
      logger.info('Hidden-state token correlation: mean '
                  f'{hidden_corr_stats["mean_corr"]:.5f} (abs '
                  f'{hidden_corr_stats["mean_abs_corr"]:.5f}, cos '
                  f'{hidden_corr_stats["mean_cos"]:.5f})')
    if config.eval.results_json_path is not None:
      samples_path = config.eval.results_json_path
    else:
      samples_path = os.path.join(os.getcwd(), 'samples',
                                  samples_fname)
    os.makedirs(os.path.dirname(samples_path), exist_ok=True)

    save_dict = dict(
      text=all_text_samples,
      np_tokens_b64=utils.np_to_base64(all_np_samples),
      gen_ppl_first_chunk_retok=gen_ppl_first_chunk_retok,
      gen_ppl_all_no_retok=gen_ppl_all_no_retok,
      entropy=entropy,
      hidden_corr=hidden_corr_stats,
      ckpt_name=config.eval.checkpoint_path,
      config=omegaconf.OmegaConf.to_container(config,
                                              resolve=True),
      avg_nfe=avg_nfe)
    with fsspec.open(samples_path, 'w') as f:
      json.dump(save_dict, f, indent=4)
    print('Samples saved at:', samples_path)
  fabric.barrier()


def _eval_ppl(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Perplexity Eval.')

  model = _load_from_checkpoint(
    diffusion_model=diffusion_model,
    config=config,
    tokenizer=tokenizer)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      ** config.wandb)
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))
  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  _, valid_ds = dataloader.get_dataloaders(
    config, tokenizer, skip_train=True, valid_seed=config.seed)
  results = trainer.validate(model, valid_ds)
  if config.eval.results_json_path is not None:
    save_path = Path(config.eval.results_json_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
      json.dump(results[0], f)
      print(f'Saved results to `{save_path}`')


def _train(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Training.')
  wandb_logger = None
  if config.get('wandb', None) is not None:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      **config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None):
    ckpt_path = config.checkpointing.resume_ckpt_path
    # Optionally load checkpoint last-v1 etc. Pick the one with highest global step
    if ckpt_path.endswith('last.ckpt'):
      ckpt_path = ckpt_path.replace('last.ckpt', 'last*')
      matching_ckpts = glob.glob(ckpt_path)
      if len(matching_ckpts) == 0:
        ckpt_path = None
      elif len(matching_ckpts) == 1:
        ckpt_path = matching_ckpts[0]
      else:
        # If there are multiple last.ckpt checkpoints, pick 
        #  the one with latest global_step.
        latest_path = None
        latest_step = 0
        for path in matching_ckpts:
          ckpt = torch.load(path, map_location='cpu', weights_only=False)
          if ckpt['global_step'] > latest_step:
            latest_step = ckpt['global_step']
            latest_path = path
        ckpt_path = latest_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  callbacks.append(ThroughputCallback(
    log_every_n_batches=config.trainer.log_every_n_steps))

  # Datasets are built here without a separate Fabric (a second process group
  # would conflict with the Trainer's DDP). For multi-GPU runs, build the
  # dataset cache on a single GPU first.
  train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
  _print_batch(train_ds, valid_ds, tokenizer)

  if config.training.finetune_path != '':
    assert utils.fsspec_exists(config.training.finetune_path)
    # Honor eval.strict_loading here too: finetuning from a checkpoint trained
    # by a *different* implementation can have non-matching state_dict keys.
    # Default stays strict (eval.strict_loading=true); set it false to relax.
    model = diffusion_model.load_from_checkpoint(
      config.training.finetune_path,
      tokenizer=tokenizer,
      config=config,
      strict=config.eval.strict_loading)
    if config.training.finetune_from_ema:
      # load_from_checkpoint set the params to the ONLINE weights; the pretrained
      # EMA was loaded into model.ema (on_load_checkpoint). Copy that EMA over the
      # params so finetuning starts from the EMA weights, then rebuild the EMA so
      # its shadow is reseeded from those weights and num_updates resets to 0
      # (fresh decay warmup -- otherwise it inherits ~pretraining-step count and
      # skips warmup). Relies on the EMA shadow matching the param set, which the
      # shared EMA count/shape validation guarantees.
      assert model.ema is not None, \
        'training.finetune_from_ema needs training.ema>0 and a checkpoint with an EMA'
      model.ema.copy_to(model._get_parameters())
      model._prepare_ema()
  else:
    model = diffusion_model(config, 
                            tokenizer=valid_ds.tokenizer)

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


def _load_generated_gsm8k(config, logger):
  logger.info(f'Loading GSM8K predictions from {config.gsm8k.input_file}')
  with open(config.gsm8k.input_file) as f:
    loaded = json.load(f)
  # Accept either a bare list of records or a previously-saved results.json
  # (which wraps the list under 'records').
  if isinstance(loaded, dict) and 'records' in loaded:
    return loaded['records']
  return loaded


def _pad_prefix_batch(id_lists, device):
  """Pad a list of 1D tensors into a batch.

  Returns:
    padded: (B, max_len) int64 tensor, zero-padded.
    lengths: (B,) int64 tensor of actual lengths.
  """
  lengths = torch.tensor(
    [len(ids) for ids in id_lists], device=device)
  max_len = int(lengths.max())
  padded = torch.zeros(
    len(id_lists), max_len, dtype=torch.long, device=device)
  for i, ids in enumerate(id_lists):
    padded[i, :len(ids)] = ids.to(device)
  return padded, lengths


def _sample_gsm8k(diffusion_model, config, logger, tokenizer, fabric):
  model = _load_from_checkpoint(diffusion_model, config, tokenizer)
  model.to(fabric.device)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  model._eval_mode()

  # Load dataset on rank 0 first, then broadcast to others
  with fabric.rank_zero_first():
    dataset = dataloader.get_dataset(config, tokenizer, mode='valid')

  # Interleaved split: rank r gets indices r, r+W, r+2W, ...
  rank = fabric.global_rank
  world_size = fabric.world_size
  local_indices = list(range(rank, len(dataset), world_size))
  batch_size = config.loader.eval_batch_size

  os.makedirs(config.gsm8k.output_dir, exist_ok=True)
  rank_json_path = os.path.join(config.gsm8k.output_dir,
                                f'rank_{rank:04d}.json')
  records = []
  total_nfe = 0
  num_batches = 0
  for batch_start in trange(
      0, len(local_indices), batch_size,
      desc=f'Rank {rank} sampling',
      disable=rank != 0):
    batch_indices = local_indices[
      batch_start:batch_start + batch_size]
    B = len(batch_indices)

    all_ids = [torch.tensor(dataset[idx]['input_ids'])
               for idx in batch_indices]
    padded, lengths = _pad_prefix_batch(
      all_ids, fabric.device)

    samples, metadata = model.generate_samples(
      num_samples=B,
      num_steps=config.sampler.steps,
      prefix_tokens=padded,
      prefix_lengths=lengths)
    total_nfe += metadata['nfe']
    num_batches += 1

    for i, idx in enumerate(batch_indices):
      prompt_len = int(lengths[i])
      generated_ids = samples[i, prompt_len:].cpu().tolist()

      response = tokenizer.decode(
        generated_ids, skip_special_tokens=True)
      records.append({
        'prompt': dataset[idx]['prompt'],
        'response_ground_truth':
          dataset[idx]['response_ground_truth'],
        'response': response,
      })
    # Cleanup memory for the next sampling batch
    del samples, metadata
    gc.collect()

  avg_nfe = total_nfe / max(num_batches, 1)
  logger.info(f'Rank {rank}: average NFE per batch: {avg_nfe:.1f}')

  with open(rank_json_path, 'w') as f:
    json.dump(records, f, indent=2)
  logger.info(f'Rank {rank}: saved {len(records)} records to {rank_json_path}')

  fabric.barrier()

  if fabric.global_rank != 0:
    return None

  all_records = []
  for r in range(world_size):
    path = os.path.join(config.gsm8k.output_dir, f'rank_{r:04d}.json')
    with open(path) as f:
      all_records.extend(json.load(f))
  return all_records


@torch.no_grad()
def _gsm8k_eval(diffusion_model, config, logger, tokenizer):
  logger.info('Starting GSM8K eval.')
  fabric = Fabric(
    accelerator=config.trainer.accelerator,
    devices=config.trainer.devices,
    num_nodes=config.trainer.num_nodes)
  fabric.launch()

  seed = config.seed + fabric.global_rank
  L.seed_everything(seed)

  if config.gsm8k.input_file is not None:
    if fabric.global_rank == 0:
      samples = _load_generated_gsm8k(config, logger)
    else:
      samples = None
  else:
    samples = _sample_gsm8k(
      diffusion_model, config, logger, tokenizer, fabric)
  if fabric.global_rank == 0:
    logger.info(f'Total records evaluated: {len(samples)}')
    per_sample_correct = np.array([
      int(sandbox_gsm8k.evaluate_samples(
        rec['response'],
        rec['response_ground_truth'],
        timeout_s=config.gsm8k.timeout))
      for rec in samples], dtype=np.int32)
    correct = int(per_sample_correct.sum())
    # Optionally: bootstrapped 95% CI (test-set bootstrap, percentile method).
    # When bootstrap_size > 1 the reported `accuracy` is the bootstrap mean
    # (per Raschka's recommendation) with 95% CI from the 2.5/97.5 percentiles
    # of the bootstrap distribution. `accuracy_se` is the std of the bootstrap
    # means (= bootstrap-estimated standard error of the accuracy).
    if config.gsm8k.bootstrap_size > 1:
      rng = np.random.default_rng(config.seed)
      idx = rng.integers(0, len(samples),
                         size=(config.gsm8k.bootstrap_size, len(samples)))
      boot_acc = per_sample_correct[idx].mean(axis=1)
      acc = float(boot_acc.mean())
      acc_se = float(boot_acc.std(ddof=1))
      acc_ci_lo = float(np.percentile(boot_acc, 2.5))
      acc_ci_hi = float(np.percentile(boot_acc, 97.5))
      logger.info(
        f'GSM8K accuracy: {acc * 100:.2f}% '
        f'[{acc_ci_lo * 100:.2f}%, {acc_ci_hi * 100:.2f}%] (95% CI) '
        f'({correct}/{len(samples)}, bootstrap N={config.gsm8k.bootstrap_size})')
    else:
      acc = correct / len(samples)
      acc_se = 0.0
      acc_ci_lo = acc
      acc_ci_hi = acc
      logger.info(f'GSM8K accuracy: {acc * 100:.2f}% ({correct}/{len(samples)})')

    os.makedirs(config.gsm8k.output_dir, exist_ok=True)
    merged_path = os.path.join(config.gsm8k.output_dir, 'results.json')
    results = {'accuracy': acc,
               'accuracy_se': acc_se,
               'accuracy_ci_lo': acc_ci_lo,
               'accuracy_ci_hi': acc_ci_hi,
               'num_correct': correct,
               'num_total': len(samples),
               'records': samples}
    if config.gsm8k.input_file is not None:
      results['input_file'] = config.gsm8k.input_file
    with open(merged_path, 'w') as f:
      json.dump(results, f, indent=2)
    logger.info(f'Results saved to {merged_path}')

  fabric.barrier()


@torch.no_grad()
def _sudoku_eval(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Sudoku eval.')
  fabric = Fabric(
    accelerator=config.trainer.accelerator,
    devices=config.trainer.devices,
    num_nodes=config.trainer.num_nodes)
  fabric.launch()

  seed = config.seed + fabric.global_rank
  L.seed_everything(seed)

  model = _load_from_checkpoint(diffusion_model, config, tokenizer)
  model.to(fabric.device)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  model._eval_mode()

  with fabric.rank_zero_first():
    dataset = dataloader.get_dataset(
      config, tokenizer, mode='valid')

  prompt_len = tokenizer.prompt_len
  rank = fabric.global_rank
  world_size = fabric.world_size
  local_indices = list(range(rank, len(dataset), world_size))
  batch_size = config.sudoku.batch_size
  records = []

  for batch_start in trange(
      0, len(local_indices), batch_size,
      desc='Sudoku eval', disable=rank != 0):
    batch_indices = local_indices[
      batch_start:batch_start + batch_size]
    B = len(batch_indices)

    # All sudoku examples have the same length, no padding needed
    batch = torch.stack(
      [dataset[idx]['input_ids'] for idx in batch_indices])
    prefix = batch[:, :prompt_len].to(fabric.device)
    prefix_lengths = torch.full(
      (B,), prompt_len, device=fabric.device)

    samples, _ = model.generate_samples(
      num_samples=B,
      num_steps=config.sampler.steps,
      prefix_tokens=prefix,
      prefix_lengths=prefix_lengths)

    gt = batch[:, prompt_len:]
    generated = samples[:, prompt_len:].cpu()
    correct = (generated == gt).all(dim=1)

    for i, idx in enumerate(batch_indices):
      records.append({
        'generated': tokenizer.decode(generated[i]),
        'ground_truth': tokenizer.decode(gt[i]),
        'correct': correct[i].item(),
      })

    del samples
    gc.collect()

  # Save per-rank results and aggregate on rank 0
  os.makedirs(config.sudoku.output_dir, exist_ok=True)
  rank_path = os.path.join(
    config.sudoku.output_dir, f'rank_{rank:04d}.json')
  with open(rank_path, 'w') as f:
    json.dump(records, f, indent=2)

  fabric.barrier()

  if fabric.global_rank == 0:
    all_records = []
    for r in range(world_size):
      path = os.path.join(
        config.sudoku.output_dir, f'rank_{r:04d}.json')
      with open(path) as f:
        all_records.extend(json.load(f))
    num_correct = sum(r['correct'] for r in all_records)
    total = len(all_records)
    logger.info(
      f'Sudoku accuracy: {num_correct}/{total} '
      f'({num_correct / total * 100:.2f}%)')
    merged_path = os.path.join(
      config.sudoku.output_dir, 'results.json')
    results = {
      'accuracy': num_correct / total,
      'num_correct': num_correct,
      'num_total': total,
      'records': all_records,
    }
    with open(merged_path, 'w') as f:
      json.dump(results, f, indent=2)
    logger.info(f'Results saved to {merged_path}')

  fabric.barrier()


@torch.no_grad()
def _maze_eval(diffusion_model, config, logger, tokenizer):
  logger.info('Starting Maze eval.')
  fabric = Fabric(
    accelerator=config.trainer.accelerator,
    devices=config.trainer.devices,
    num_nodes=config.trainer.num_nodes)
  fabric.launch()

  seed = config.seed + fabric.global_rank
  L.seed_everything(seed)

  model = _load_from_checkpoint(diffusion_model, config, tokenizer)
  model.to(fabric.device)
  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None
  model._eval_mode()

  eval_split = config.maze.get('eval_split', 'valid')
  assert eval_split in ('valid', 'train'), (
    f'maze.eval_split must be "valid" or "train", got {eval_split!r}')
  logger.info(f'Maze eval split: {eval_split}.')
  with fabric.rank_zero_first():
    dataset = dataloader.get_dataset(
      config, tokenizer, mode=eval_split)

  prompt_len = tokenizer.prompt_len
  # The two halves differ ONLY at the interior shortest-path cells, which
  # is a small fraction of the grid (e.g. ~100-140 of 900 cells at 30x30) --
  # so ~85% of the solution half is a verbatim copy of the prompt, and
  # full-span exact match is dominated by copying. Score the path cells
  # separately so the headline number isn't inflated by it.
  path_id = tokenizer.path_id
  rank = fabric.global_rank
  world_size = fabric.world_size
  num_examples = len(dataset)
  max_examples = config.maze.get('max_examples', None)
  if max_examples is not None:
    num_examples = min(num_examples, int(max_examples))
    logger.info(
      f'Capping maze eval at {num_examples} examples.')
  local_indices = list(range(rank, num_examples, world_size))
  batch_size = config.maze.batch_size
  records = []

  for batch_start in trange(
      0, len(local_indices), batch_size,
      desc='Maze eval', disable=rank != 0):
    batch_indices = local_indices[
      batch_start:batch_start + batch_size]
    B = len(batch_indices)

    # All maze examples have the same length, no padding needed.
    batch = torch.stack(
      [dataset[idx]['input_ids'] for idx in batch_indices])
    prefix = batch[:, :prompt_len].to(fabric.device)
    prefix_lengths = torch.full(
      (B,), prompt_len, device=fabric.device)

    samples, _ = model.generate_samples(
      num_samples=B,
      num_steps=config.sampler.steps,
      prefix_tokens=prefix,
      prefix_lengths=prefix_lengths)

    # The prompt half is conditioning: the sampler must hold it fixed at
    # every step. If it drifts, the model is solving a corrupted maze and
    # every downstream number is meaningless -- so check, don't assume.
    prompt_out = samples[:, :prompt_len].cpu()
    prompt_in = batch[:, :prompt_len]
    prompt_intact = (prompt_out == prompt_in).all(dim=1)
    prompt_cell_match = (prompt_out == prompt_in).float().mean(dim=1)

    gt = batch[:, prompt_len:]
    generated = samples[:, prompt_len:].cpu()
    correct = (generated == gt).all(dim=1)
    # Path cells only: ignore the walls/open cells copied from the prompt.
    is_path = gt == path_id
    path_correct = ((generated == gt) | ~is_path).all(dim=1)

    for i, idx in enumerate(batch_indices):
      generated_str = tokenizer.decode(generated[i])
      gt_str = tokenizer.decode(gt[i])
      # Validity (any shortest path accepted), not exact match to the one
      # labelled path: Maze-Hard mazes have many shortest paths.
      valid_path, valid_solution = utils.maze_solution_validity(
        generated_str, gt_str)
      records.append({
        'index': idx,
        'generated': generated_str,
        'ground_truth': gt_str,
        'correct': correct[i].item(),
        'path_correct': path_correct[i].item(),
        'valid_solution': valid_solution,
        'valid_path': valid_path,
        'prompt_intact': prompt_intact[i].item(),
        'prompt_cell_match': prompt_cell_match[i].item(),
      })

    del samples
    gc.collect()

  # Save per-rank results and aggregate on rank 0
  os.makedirs(config.maze.output_dir, exist_ok=True)
  rank_path = os.path.join(
    config.maze.output_dir, f'rank_{rank:04d}.json')
  with open(rank_path, 'w') as f:
    json.dump(records, f, indent=2)

  fabric.barrier()

  if fabric.global_rank == 0:
    all_records = []
    for r in range(world_size):
      path = os.path.join(
        config.maze.output_dir, f'rank_{r:04d}.json')
      with open(path) as f:
        all_records.extend(json.load(f))
    num_correct = sum(r['correct'] for r in all_records)
    num_path_correct = sum(r['path_correct'] for r in all_records)
    num_valid_solution = sum(r['valid_solution'] for r in all_records)
    num_valid_path = sum(r['valid_path'] for r in all_records)
    total = len(all_records)
    num_prompt_intact = sum(r['prompt_intact'] for r in all_records)
    mean_prompt_match = sum(
      r['prompt_cell_match'] for r in all_records) / total
    if num_prompt_intact != total:
      logger.error(
        f'PROMPT CORRUPTED: only {num_prompt_intact}/{total} samples kept '
        f'the conditioning prefix intact (mean cell match '
        f'{mean_prompt_match * 100:.2f}%). The sampler is not holding the '
        f'prefix fixed, so the accuracies below are not meaningful.')
    else:
      logger.info(
        f'Prompt prefix intact in all {total}/{total} samples.')
    logger.info(
      f'Maze accuracy: {num_correct}/{total} '
      f'({num_correct / total * 100:.2f}%)')
    logger.info(
      f'Maze path-cell accuracy: {num_path_correct}/{total} '
      f'({num_path_correct / total * 100:.2f}%)')
    logger.info(
      f'Maze valid-solution accuracy (any shortest path, maze copied '
      f'exactly): {num_valid_solution}/{total} '
      f'({num_valid_solution / total * 100:.2f}%)')
    logger.info(
      f'Maze valid-path accuracy (any shortest path, walls elsewhere '
      f'ignored): {num_valid_path}/{total} '
      f'({num_valid_path / total * 100:.2f}%)')
    merged_path = os.path.join(
      config.maze.output_dir, 'results.json')
    results = {
      'eval_split': eval_split,
      'accuracy': num_correct / total,
      'path_accuracy': num_path_correct / total,
      'valid_solution_accuracy': num_valid_solution / total,
      'valid_path_accuracy': num_valid_path / total,
      'prompt_intact_frac': num_prompt_intact / total,
      'prompt_cell_match': mean_prompt_match,
      'num_correct': num_correct,
      'num_path_correct': num_path_correct,
      'num_valid_solution': num_valid_solution,
      'num_valid_path': num_valid_path,
      'num_total': total,
      'records': all_records,
    }
    with open(merged_path, 'w') as f:
      json.dump(results, f, indent=2)
    logger.info(f'Results saved to {merged_path}')

  fabric.barrier()


class FakeTokenizer:
  def __init__(self, vocab_length):
    self.vocab_size = vocab_length
    self.bos_token_id = 0
    self.mask_id = vocab_length - 1
  
  def __len__(self):
    return self.vocab_size


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
  """Main entry point for training."""
  if config.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
  if config.cudnn_benchmark:
    torch.backends.cudnn.benchmark = True
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)
  if config.algo.name == 'ar':
    diffusion_model = algo.AR
  elif config.algo.name == 'mdlm':
    diffusion_model = algo.MDLM
  elif config.algo.name == 'duo_base':
    diffusion_model = algo.DUO_BASE
  elif config.algo.name == 'duo-mixing':
    diffusion_model = algo.MixingDUO
  elif config.algo.name == 'lddm_m':
    diffusion_model = algo.LDDM_M
  elif config.algo.name == 'lddm_u':
    diffusion_model = algo.LDDM_U
  elif config.algo.name == 'law':
    diffusion_model = algo.LAW
  elif config.algo.name == 'law_carry_ablation':
    diffusion_model = algo.LAWCarryAblation
  elif config.algo.name == 'candi':
    diffusion_model = algo.CANDI
  elif config.algo.name == 'flm':
    diffusion_model = algo.FLM
  elif config.algo.name == 'sfm':
    diffusion_model = algo.SFM
  else:
    raise ValueError(
      f'Invalid algorithm name: {config.algo.name}')
  kwargs = {'diffusion_model': diffusion_model,
            'config': config,
            'tokenizer': tokenizer,
            'logger': logger}
  
  if hasattr(config.model, 'vocab_size'):
    kwargs['tokenizer'] = FakeTokenizer(config.model.vocab_size)
  
  if config.mode == 'sample_eval':
    _generate_samples(**kwargs)
  elif config.mode == 'ppl_eval':
    _eval_ppl(**kwargs)
  elif config.mode == 'gsm8k_eval':
    _gsm8k_eval(**kwargs)
  elif config.mode == 'sudoku_eval':
    _sudoku_eval(**kwargs)
  elif config.mode == 'maze_eval':
    _maze_eval(**kwargs)
  elif config.mode == 'train':
    _train(**kwargs)
  else:
    raise ValueError(config.mode)


if __name__ == '__main__':
  main()
