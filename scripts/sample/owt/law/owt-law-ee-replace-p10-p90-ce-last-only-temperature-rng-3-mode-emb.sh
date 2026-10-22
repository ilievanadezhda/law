#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoints/owt/law/owt_law_ee_replace_p10_p90_ce_last_only_temperature_rng_3_mode_emb/18-308000.ckpt}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/owt/law/owt-law-ee-replace-p10-p90-ce-last-only-temperature-rng-3-mode-emb}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-1024}"  # the paper uses 64 and 1024
TEMPERATURE="${TEMPERATURE:-1.0}"

cd "${REPO_ROOT}"

python -u -m main \
    mode=sample_eval \
    data=openwebtext-split \
    data.cache_dir="${CACHE_DIR}" \
    model=small-loop \
    model.length=1024 \
    model.block=prenorm \
    model.mode_embedding=True \
    algo=law \
    algo.loss=ce \
    'algo.pass_count_probabilities=[0.1,0.9]' \
    algo.pass_objective=last_only \
    algo.stop_gradient_between_passes=True \
    algo.dt=0.0009765625 \
    algo.carry_regularization.temperature.enabled=True \
    algo.carry_regularization.temperature.range_scale=3 \
    algo.carry_regularization.temperature.resample_per_pass=False \
    sampler=law \
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
