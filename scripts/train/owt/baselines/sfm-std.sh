#!/bin/bash

# Trained with the 𝕊-FLM recipe (1M-step schedule). As in the paper, we use the
# checkpoint at step 400k; the learning rate is constant after warmup.

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/owt/baselines/sfm-std}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-4}"

cd "${REPO_ROOT}"

python -u -m main \
    data=openwebtext-split \
    data.cache_dir="${CACHE_DIR}" \
    model=small-sphere-dit \
    model.init=ngpt \
    algo=sfm \
    algo.renormalize_weights=False \
    algo.invert_time_convention=false \
    noise=log-linear-adaptive \
    noise.alpha_max=0.121 \
    noise.adaptive_plot_profile=true \
    noise.adaptive_refit_every=50 \
    noise.adaptive_buffer_size=25600 \
    noise.adaptive_ema=0.9 \
    noise.adaptive_uniform_mix=1e-3 \
    loader.global_batch_size=512 \
    loader.batch_size=32 \
    loader.eval_batch_size=32 \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.max_steps=1_000_000 \
    trainer.val_check_interval=50_000 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
    checkpointing.resume_from_ckpt=True \
    checkpointing.resume_ckpt_path="${OUTPUT_DIR}/checkpoints/last.ckpt" \
    hydra.run.dir="${OUTPUT_DIR}"
