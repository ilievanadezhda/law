#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoints/owt/lddm/owt_law_ee_add_p10_p90_elbo_last_only_temperature_rng_3/19-308000.ckpt}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/owt/lddm/owt-law-ee-add-p10-p90-elbo-last-only-temperature-rng-3}"
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
    model.mode_embedding=False \
    algo=law_carry_ablation \
    algo.carry_representation=expected_embedding \
    algo.injection_mode=add \
    algo.loss=elbo \
    'algo.pass_count_probabilities=[0.1,0.9]' \
    algo.pass_objective=last_only \
    algo.stop_gradient_between_passes=True \
    algo.advance_time_between_passes=False \
    algo.carry_regularization.temperature.enabled=True \
    algo.carry_regularization.temperature.range_scale=3 \
    algo.carry_regularization.temperature.resample_per_pass=False \
    sampler=law_carry_ablation \
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
