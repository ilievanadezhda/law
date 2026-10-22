#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoints/owt/baselines/duo-400000.ckpt}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/owt/baselines/duo}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-1024}"  # the paper uses 64 and 1024
TEMPERATURE="${TEMPERATURE:-1.0}"

cd "${REPO_ROOT}"

python -u -m main \
    mode=sample_eval \
    data=openwebtext-split \
    data.cache_dir="${CACHE_DIR}" \
    model=small \
    model.length=1024 \
    algo=duo-base \
    sampler=ancestral \
    sampler.steps="${STEPS}" \
    sampler.temperature_schedule=constant-"${TEMPERATURE}" \
    sampler.num_sample_batches=32 \
    loader.eval_batch_size=16 \
    loader.num_workers=8 \
    eval.checkpoint_path="${CKPT_PATH}" \
    eval.compute_generative_perplexity=True \
    eval.compute_hidden_corr=True \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    hydra.run.dir="${OUTPUT_DIR}"
