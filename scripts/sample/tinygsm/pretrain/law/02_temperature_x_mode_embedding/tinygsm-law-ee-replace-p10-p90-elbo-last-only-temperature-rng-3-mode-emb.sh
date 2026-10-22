#!/bin/bash

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REPO_ROOT="$(cd "$(dirname "$0")/../../../../../.." && pwd)"
CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoints/tinygsm/pretrain/law/02_temperature_x_mode_embedding/tinygsm_law_ee_replace_p10_p90_elbo_last_only_temperature_rng_3_mode_emb/9-192000.ckpt}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/data_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_runs/tinygsm/pretrain/law/02_temperature_x_mode_embedding/tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-mode-emb}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-1}"
STEPS="${STEPS:-512}"  # the paper sweeps 1, 2, 4, ..., 1024
TEMPERATURE="${TEMPERATURE:-1.0}"

cd "${REPO_ROOT}"

python -u -m main \
    mode=gsm8k_eval \
    eval.checkpoint_path="${CKPT_PATH}" \
    data=gsm8k-test \
    data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    data.data_path="${REPO_ROOT}/data/gsm8k_test.json" \
    data.cache_dir="${CACHE_DIR}" \
    model=small-loop \
    model.length=512 \
    model.mode_embedding=True \
    algo=law \
    sampler=law \
    sampler.steps="${STEPS}" \
    sampler.temperature_schedule=constant-"${TEMPERATURE}" \
    gsm8k.bootstrap_size=1000 \
    gsm8k.output_dir="${OUTPUT_DIR}" \
    loader.eval_batch_size=128 \
    trainer.num_nodes="${NUM_NODES}" \
    trainer.devices="${DEVICES}" \
    hydra.run.dir="${OUTPUT_DIR}"
