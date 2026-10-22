#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/outputs/maze/maze-law-ee-replace-p25-p75-elbo-last-only/checkpoints/last.ckpt}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/maze/maze-law-ee-replace-p25-p75-elbo-last-only}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-800}"  # the paper sweeps 1, 2, 4, ..., 512, 800
TEMPERATURE="${TEMPERATURE:-1.0}"

cd "${REPO_ROOT}"

python -u -m main \
    mode=maze_eval \
    data=maze \
    data.cache_dir="${CACHE_DIR}" \
    model=maze-loop \
    model.length=800 \
    algo=law \
    algo.loss=elbo \
    'algo.pass_count_probabilities=[0.25,0.75]' \
    algo.pass_objective=last_only \
    algo.stop_gradient_between_passes=True \
    algo.advance_time_between_passes=True \
    algo.dt=0.00125 \
    sampler=law \
    eval.checkpoint_path="${CKPT_PATH}" \
    maze.eval_split=valid \
    maze.max_examples=null \
    maze.batch_size=16 \
    maze.output_dir="${OUTPUT_DIR}" \
    sampler.steps="${STEPS}" \
    sampler.temperature_schedule=constant-"${TEMPERATURE}" \
    loader.eval_batch_size=16 \
    loader.num_workers=4 \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    hydra.run.dir="${OUTPUT_DIR}"
