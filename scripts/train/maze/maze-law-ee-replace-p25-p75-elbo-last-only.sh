#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/maze/maze-law-ee-replace-p25-p75-elbo-last-only}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"

cd "${REPO_ROOT}"

python -u -m main \
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
    loader.global_batch_size=256 \
    loader.batch_size=256 \
    loader.eval_batch_size=256 \
    loader.num_workers=8 \
    eval.generate_samples=False \
    seed=1 \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.max_steps=80000 \
    trainer.val_check_interval=1000000 \
    trainer.limit_val_batches=0 \
    trainer.num_sanity_val_steps=0 \
    'callbacks=[checkpoint_every_n_steps,checkpoint_monitor,learning_rate_monitor,checkpoint_at_max_steps]' \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=4000 \
    checkpointing.resume_from_ckpt=True \
    checkpointing.resume_ckpt_path="${OUTPUT_DIR}/checkpoints/last.ckpt" \
    hydra.run.dir="${OUTPUT_DIR}"
