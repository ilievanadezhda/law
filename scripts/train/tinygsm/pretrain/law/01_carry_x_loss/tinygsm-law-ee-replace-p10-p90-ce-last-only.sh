#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../../../../.." && pwd)"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/tinygsm/pretrain/law/01_carry_x_loss/tinygsm-law-ee-replace-p10-p90-ce-last-only}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-4}"

cd "${REPO_ROOT}"

python -u -m main \
    data=tiny-gsm \
    data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.cache_dir="${CACHE_DIR}" \
    data.wrap=False \
    data.train_on_prompt=False \
    data.train_on_pad=True \
    data.filter_too_long=True \
    model=small-loop \
    model.length=512 \
    algo=law \
    algo.loss=ce \
    'algo.pass_count_probabilities=[0.1,0.9]' \
    algo.pass_objective=last_only \
    algo.stop_gradient_between_passes=True \
    sampler=law \
    loader.global_batch_size=512 \
    loader.batch_size=32 \
    loader.eval_batch_size=16 \
    loader.num_workers=8 \
    eval.generate_samples=False \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    trainer.max_steps=192000 \
    trainer.val_check_interval=1000000 \
    trainer.limit_val_batches=0 \
    trainer.num_sanity_val_steps=0 \
    'callbacks=[checkpoint_every_n_steps,checkpoint_monitor,learning_rate_monitor,checkpoint_at_max_steps]' \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
    checkpointing.resume_from_ckpt=True \
    checkpointing.resume_ckpt_path="${OUTPUT_DIR}/checkpoints/last.ckpt" \
    hydra.run.dir="${OUTPUT_DIR}"
