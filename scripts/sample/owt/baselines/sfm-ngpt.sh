#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
VELOCITY="${VELOCITY:-exact}"        # exact / sample
TOPK_VELOCITY="${TOPK_VELOCITY:--1}"  # -1 = full vocabulary
BUF_SIZE=$((50 * 512))
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoints/owt/baselines/sfm-ngpt-400000.ckpt}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/owt/baselines/sfm-ngpt}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-1024}"  # the paper uses 64 and 1024
TEMPERATURE="${TEMPERATURE:-1.0}"

cd "${REPO_ROOT}"

python -u -m main \
    mode=sample_eval \
    data=openwebtext-split \
    data.cache_dir="${CACHE_DIR}" \
    model=small-sphere-arch \
    model.length=1024 \
    model.normalize_input_embed=False \
    algo=sfm \
    algo.renormalize_weights=True \
    algo.invert_time_convention=false \
    noise=log-linear-adaptive \
    noise.alpha_max=0.121 \
    noise.adaptive_refit_every=50 \
    noise.adaptive_buffer_size="${BUF_SIZE}" \
    noise.adaptive_ema=0.9 \
    noise.adaptive_uniform_mix=1e-3 \
    sampler=sfm \
    sampler.noise_removal=greedy \
    sampler.velocity="${VELOCITY}" \
    sampler.top_k_velocity="${TOPK_VELOCITY}" \
    sampler.steps="${STEPS}" \
    sampler.temperature_schedule=constant-"${TEMPERATURE}" \
    sampler.num_sample_batches=32 \
    loader.eval_batch_size=16 \
    loader.num_workers=8 \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.strict_loading=false \
    eval.compute_generative_perplexity=True \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    hydra.run.dir="${OUTPUT_DIR}"
