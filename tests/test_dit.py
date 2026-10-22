import pytest
import torch

from models.dit import DIT
from samplers import ARContext
from test_law import CUDA, Tokenizer, batch, config


@CUDA
@pytest.mark.parametrize('softcap', [0., 5.])
def test_ordinary_dit_forward(softcap):
  torch.manual_seed(17)
  model = DIT(config('model.type=dit', 'algo.adaLN=False',
                     f'+model.softcap={softcap}'), Tokenizer.vocab_size).cuda().eval()
  torch.nn.init.normal_(model.output_layer.linear.weight, std=.1)
  tokens, _ = batch()
  logits = model(None, tokens, torch.ones(2, device='cuda'))
  assert logits.shape == (2, 8, Tokenizer.vocab_size)
  assert torch.isfinite(logits).all()
  logits.float().square().mean().backward()
  gradient = model.blocks[0].attn_qkv.weight.grad
  assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@CUDA
def test_causal_dit_cache_matches_full_forward():
  torch.manual_seed(17)
  model = DIT(config('model.type=dit', 'algo.adaLN=False',
                     'algo.causal_attention=True'), Tokenizer.vocab_size).cuda().eval()
  torch.nn.init.normal_(model.output_layer.linear.weight, std=.1)
  tokens, _ = batch()
  sigma = torch.ones(2, device='cuda')
  with torch.no_grad():
    expected = model(None, tokens, sigma)
    model.reset_kv_cache()
    actual = torch.cat([
      model(None, tokens[:, i:i + 1], sigma, context=ARContext(kv_cache=True))
      for i in range(tokens.shape[1])], dim=1)
  torch.testing.assert_close(actual, expected, atol=.01, rtol=.01)
  assert model.ctx_cached_len == tokens.shape[1]
  model.reset_kv_cache()
  assert model.ctx_cached_len == 0
  assert all(block.k_cache is None and block.v_cache is None for block in model.blocks)
