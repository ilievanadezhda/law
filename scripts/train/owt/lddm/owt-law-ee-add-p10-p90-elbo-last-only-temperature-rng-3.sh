#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/owt/lddm/owt-law-ee-add-p10-p90-elbo-last-only-temperature-rng-3}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-4}"

cd "${REPO_ROOT}"

python -u -m main \
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
    loader.global_batch_size=512 \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    loader.num_workers=8 \
    eval.generate_samples=True \
    eval.generate_samples_every_n_steps=5000 \
    eval.compute_generative_perplexity=True \
    eval.compute_hidden_corr=True \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.max_steps=308000 \
    trainer.val_check_interval=5000 \
    trainer.num_sanity_val_steps=0 \
    'callbacks=[checkpoint_every_n_steps,checkpoint_monitor,learning_rate_monitor,checkpoint_at_max_steps]' \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
    checkpointing.resume_from_ckpt=True \
    checkpointing.resume_ckpt_path="${OUTPUT_DIR}/checkpoints/last.ckpt" \
    hydra.run.dir="${OUTPUT_DIR}"
