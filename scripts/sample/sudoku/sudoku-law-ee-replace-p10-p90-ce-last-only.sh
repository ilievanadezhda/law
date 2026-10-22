#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DIFFICULTY="${DIFFICULTY:-hard}"  # easy / medium / hard
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/outputs/sudoku/sudoku-law-ee-replace-p10-p90-ce-last-only-${DIFFICULTY}/checkpoints/last.ckpt}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/sudoku/sudoku-law-ee-replace-p10-p90-ce-last-only-${DIFFICULTY}}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-180}"  # the paper sweeps 1, 2, 4, ..., 128, 180
TEMPERATURE="${TEMPERATURE:-1.0}"

cd "${REPO_ROOT}"

python -u -m main \
    mode=sudoku_eval \
    data=sudoku \
    data.cache_dir="${CACHE_DIR}" \
    data.difficulty="${DIFFICULTY}" \
    data.sudoku_num_workers=8 \
    model=tiny-loop \
    algo=law \
    algo.loss=ce \
    'algo.pass_count_probabilities=[0.1,0.9]' \
    algo.pass_objective=last_only \
    algo.stop_gradient_between_passes=True \
    algo.advance_time_between_passes=True \
    algo.dt=0.005555555555555556 \
    sampler=law \
    eval.checkpoint_path="${CKPT_PATH}" \
    sudoku.batch_size=64 \
    sudoku.output_dir="${OUTPUT_DIR}" \
    sampler.steps="${STEPS}" \
    sampler.temperature_schedule=constant-"${TEMPERATURE}" \
    loader.eval_batch_size=64 \
    loader.num_workers=4 \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    hydra.run.dir="${OUTPUT_DIR}"
