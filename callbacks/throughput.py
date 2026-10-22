import time
from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


class ThroughputCallback(Callback):
  """
  Simple callback that measures average time per batch over n batches.
  """
  
  def __init__(self, log_every_n_batches: int = 100):
    super().__init__()
    self.log_every_n_batches = log_every_n_batches
    self.start_time: Optional[float] = None
    self.batch_count: int = 0
    
  def on_train_batch_start(self, trainer: pl.Trainer,
    pl_module: pl.LightningModule, batch: Any, batch_idx: int):
    """Start timing when we begin the rolling window."""
    if self.batch_count % self.log_every_n_batches == 0:
      self.start_time = time.perf_counter()
      
  def on_train_batch_end(self, trainer: pl.Trainer,
    pl_module: pl.LightningModule, outputs: Any, batch: Any,
    batch_idx: int):
    """Track batch completion and log average time per batch every N batches."""
    self.batch_count += 1
    
    if self.batch_count % self.log_every_n_batches == 0 \
      and self.start_time is not None:
      batch_shape = batch['input_ids'].shape
      total_time = time.perf_counter() - self.start_time

      # We log throughput based on 3 batch notions:
      # - loader batch: what dataloader provides (always batch['input_ids'].shape[0])
      # - encoder batch: what the encoder actually processes (may be smaller if algo slices)
      # - decoder/loss batch: what the loss is computed over (encoder batch * decoder_copies_per_clean)
      b_loader = batch_shape[0]
      seq_len = batch_shape[1]

      algo_cfg = pl_module.config.algo
      copies = algo_cfg.get("decoder_copies_per_clean", 1)
      enc_bs_cfg = algo_cfg.get("encoder_batch_size", None)

      b_encoder = b_loader
      if enc_bs_cfg is not None:
        b_encoder = int(enc_bs_cfg)
      b_decoder = b_encoder * copies

      # Loader throughput (what dataloader provides)
      loader_seq_per_s = self.log_every_n_batches * b_loader / total_time
      loader_tok_per_s = self.log_every_n_batches * b_loader * seq_len / total_time

      # Encoder throughput (what encoder processes)
      encoder_seq_per_s = self.log_every_n_batches * b_encoder / total_time
      encoder_tok_per_s = self.log_every_n_batches * b_encoder * seq_len / total_time

      # Decoder/loss throughput (what loss is computed over)
      decoder_seq_per_s = self.log_every_n_batches * b_decoder / total_time
      decoder_tok_per_s = self.log_every_n_batches * b_decoder * seq_len / total_time

      # Raw step speed (independent of batch size)
      steps_per_second = self.log_every_n_batches / total_time
      seconds_per_step = total_time / self.log_every_n_batches

      pl_module.log_dict({
        "throughput/loader_seq_per_device_per_second": loader_seq_per_s,
        "throughput/loader_token_per_device_per_second": loader_tok_per_s,
        "throughput/encoder_seq_per_device_per_second": encoder_seq_per_s,
        "throughput/encoder_token_per_device_per_second": encoder_tok_per_s,
        "throughput/decoder_seq_per_device_per_second": decoder_seq_per_s,
        "throughput/decoder_token_per_device_per_second": decoder_tok_per_s,
        "throughput/steps_per_second": steps_per_second,
        "throughput/seconds_per_step": seconds_per_step,
        "throughput/decoder_copies_per_clean": float(copies),
        }, on_step=True, on_epoch=False, sync_dist=False)