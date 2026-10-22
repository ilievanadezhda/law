# ⚖️ Loopholing All the Way: Language Modeling with Pure Self-Conditioning

By [Adrien Deschenaux](https://people.epfl.ch/adrien.deschenaux?lang=en)\*, [Nadezhda Ilieva](https://ilievanadezhda.github.io)\*, [Justin Deschenaux](https://jdeschena.com)\*, [Manuel Madeira](https://manuelmlmadeira.github.io), [Dorina Thanou](https://people.epfl.ch/dorina.thanou), [Aurelien Lucchi](https://alucchi.github.io), [Antonio Orvieto](http://orvi.altervista.org/), [Caglar Gulcehre](https://www.caglar.ai) and [Pascal Frossard](https://people.epfl.ch/pascal.frossard).

EPFL · University of Basel · Microsoft AI · ELLIS Institute Tübingen, MPI for Intelligent Systems, Tübingen AI Center

\*Equal contribution.

[![arXiv](https://img.shields.io/badge/arXiv-2610.30068-red.svg)](https://arxiv.org/abs/2610.30068)
[![HuggingFace](https://img.shields.io/badge/🤗-Huggingface-blue)](https://huggingface.co/nadjushenka12/law)

**Abstract**: Recent Diffusion Language Models (DLMs) use Self-Conditioning (SC), or Loopholing, to improve their performance. While SC is traditionally seen as an auxiliary heuristic, its empirical success raises the question: **can a continuous generative model be based purely on Self-Conditioning?** We answer affirmatively with LAW ("Loopholing All the Way"), which propagates the *Expected Embedding* under the denoiser's prediction across steps, decoding the tokens once, at the end. On Sudoku, LAW outperforms previous Discrete and Continuous Diffusion Models, including Loopholing Discrete Diffusion. On Maze, LAW is the most accurate from 2 to 64 NFE. After training on TinyGSM and sampling at temperature 1 with 1024 NFE, LAW solves 42.2% of GSM8K problems against 34.5% for uniform Loopholing Discrete Diffusion. Additionally, fine-tuning a pre-trained Uniform Diffusion Model (Duo) with LAW increases its accuracy from 17.2% to 48.1%. Finally, on OpenWebText, LAW pushes the Pareto frontier and approaches Autoregressive Models (ARMs) with 64 NFE.

This repository contains the training and evaluation code for LAW, together with LDDM and the baselines we compare against (AR, MDLM, Duo, FLM, 𝕊-FLM, CANDI). We release checkpoints for two settings:

- **OpenWebText (OWT)**: general language modeling, GPT-2 tokenizer.
- **TinyGSM**: math reasoning evaluated on GSM8K, SmolLM-135M tokenizer. Both training from scratch and fine-tuning a Duo checkpoint with LAW.

We also provide training and evaluation scripts (no released checkpoints) for **Maze** (20×20) and **Sudoku** (easy / medium / hard).

[Getting started](#getting-started) · [Checkpoints](#checkpoints) · [Training](#training) · [Sampling & evaluation](#sampling--evaluation) · [Citation](#citation)

# Getting started

Create a fresh environment and install the Python dependencies:

```bash
conda create -n law python=3.12
conda activate law
pip install -r requirements.txt
```

`requirements.txt` intentionally does **not** include `torch`, `numpy` or `flash-attn`. We work inside the NGC PyTorch container (`nvcr.io/nvidia/pytorch:25.02-py3`), which already ships torch and flash-attention with matching CUDA / cuDNN / NCCL builds; these are difficult to align, and pip-installing them on top of the container would replace its builds. If you are not using the container, install them yourself (we use `torch==2.7.0`, `numpy==1.26.4`) **before** running `pip install -r requirements.txt`, for example with a prebuilt flash-attn wheel:

```bash
pip install torch==2.7.0 numpy==1.26.4
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

# Checkpoints

All checkpoints are in the Hugging Face repo [`nadjushenka12/law`](https://huggingface.co/nadjushenka12/law): LAW, LDDM and the OWT baselines (AR, MDLM, Duo, FLM, 𝕊-FLM), trained on OpenWebText and TinyGSM. LAW and LDDM are trained with the same FLOPs as the baselines (400k steps on OWT, 250k steps on TinyGSM), so they train for fewer steps; each folder of the Hugging Face repo has a README explaining how the step count was matched. The TinyGSM baselines are in [`jdeschena/s-flm`](https://huggingface.co/jdeschena/s-flm).

Each file is a full PyTorch Lightning checkpoint (weights, EMA weights, optimizer state and training config). Download them into `./checkpoints` so that the sampling scripts find them at their default path:

```bash
# A single checkpoint:
hf download nadjushenka12/law \
    owt/law/owt_law_ee_replace_p10_p90_ce_last_only_temperature_rng_3_mode_emb/18-308000.ckpt \
    --local-dir ./checkpoints

# All TinyGSM checkpoints:
hf download nadjushenka12/law --include 'tinygsm/**' --local-dir ./checkpoints
```

# Training

All training scripts are in `scripts/train/`, organized like the Hugging Face repo, with one script per released checkpoint. Each script is self-contained and exposes a few environment variables: `OUTPUT_DIR`, `CACHE_DIR`, `NUM_NODES`, `DEVICES`. The global batch size is fixed in each script, and gradient accumulation adapts to the number of GPUs. The LAW and LDDM scripts resume automatically from `OUTPUT_DIR/checkpoints/last.ckpt`. Runs log to wandb offline; set `WANDB_MODE=online` to sync.

We originally trained on 4 nodes × 4 GPUs (OWT), 2 nodes × 4 GPUs (TinyGSM from scratch), 1 node × 4 GPUs (TinyGSM fine-tuning) and a single GPU (Maze, Sudoku).

Example:

```bash
NUM_NODES=1 DEVICES=8 \
OUTPUT_DIR=./outputs/owt_law \
bash scripts/train/owt/law/owt-law-ee-replace-p10-p90-ce-last-only-temperature-rng-3-mode-emb.sh
```

### Main runs

| Setting | Run | Scripts |
|---|---|---|
| OWT | `owt-law-ee-replace-p10-p90-ce-last-only-temperature-rng-3-mode-emb` | [train](scripts/train/owt/law/owt-law-ee-replace-p10-p90-ce-last-only-temperature-rng-3-mode-emb.sh) · [sample](scripts/sample/owt/law/owt-law-ee-replace-p10-p90-ce-last-only-temperature-rng-3-mode-emb.sh) |
| TinyGSM, from scratch | `tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3` | [train](scripts/train/tinygsm/pretrain/law/02_temperature_x_mode_embedding/tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3.sh) · [sample](scripts/sample/tinygsm/pretrain/law/02_temperature_x_mode_embedding/tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3.sh) |
| TinyGSM, fine-tuning Duo | `finetune-tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-bs-512` | [train](scripts/train/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-bs-512.sh) · [sample](scripts/sample/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-bs-512.sh) |
| Maze | `maze-law-ee-replace-p25-p75-elbo-last-only-temperature-rng-2` | [train](scripts/train/maze/maze-law-ee-replace-p25-p75-elbo-last-only-temperature-rng-2.sh) · [sample](scripts/sample/maze/maze-law-ee-replace-p25-p75-elbo-last-only-temperature-rng-2.sh) |
| Sudoku | `sudoku-law-ee-replace-p10-p90-ce-last-only-temperature-rng-2` | [train](scripts/train/sudoku/sudoku-law-ee-replace-p10-p90-ce-last-only-temperature-rng-2.sh) · [sample](scripts/sample/sudoku/sudoku-law-ee-replace-p10-p90-ce-last-only-temperature-rng-2.sh) |

### All runs

Every checkpoint on Hugging Face has a matching script. The folders mirror the Hugging Face repo:

```
scripts/train/                                                     (scripts/sample/ has the same layout)
  owt/law/                                                         LAW: loss, temperature range, mode embedding
  owt/lddm/                                                        LDDM-M, LDDM-U and LDDM with an expected-embedding carry
  owt/baselines/                                                   AR, MDLM, Duo, FLM, 𝕊-FLM
  tinygsm/pretrain/law/01_carry_x_loss/
  tinygsm/pretrain/law/02_temperature_x_mode_embedding/
  tinygsm/pretrain/law/03_pass_count_x_probability/
  tinygsm/pretrain/law/04_backprop_through_loop/
  tinygsm/pretrain/lddm/
  tinygsm/finetune/law/                                            fine-tuning Duo with LAW, batch sizes 64 to 512
  maze/  sudoku/                                                   LAW with and without carry regularization
```

[`scripts/README.md`](scripts/README.md) lists every run with a short description.

Run names describe the setup: `ee-replace` is the expected-embedding carry (`hidden-replace` carries the hidden state, `ee-add` adds the carry to the input as in LDDM); `p10-p90` means 1 pass with probability 0.1 and 2 passes with probability 0.9 (`3-pass-p10-p0-p90`: 1 or 3 passes); `ce` / `elbo` is the loss; `last-only` puts the loss on the last pass only; `temperature-rng-3` is the carry temperature regularization with range 3; `mode-emb` adds the mode embedding; `backprop2` backpropagates through both passes.

### Fine-tuning Duo with LAW (TinyGSM)

The fine-tuning runs start from Duo trained for 250k steps on TinyGSM and train for 50k more steps with LAW. The scripts read the Duo checkpoint from `DONOR_CKPT`, which defaults to the Hugging Face file `tinygsm/finetune/duo_original/duo-250000.ckpt`:

```bash
hf download nadjushenka12/law tinygsm/finetune/duo_original/duo-250000.ckpt --local-dir ./checkpoints
bash scripts/train/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-bs-512.sh
```

That checkpoint already contains the state LAW needs to resume (`pass_count_generator_state`). To start from another Duo checkpoint, add it first with `python scripts/tools/add_pass_count_generator_state.py duo.ckpt duo-law.ckpt`.

### Maze and Sudoku

The Maze dataset (HRM's Maze-Hard recipe at 20×20, 1000 train / 1000 test mazes) is generated on the first run and cached in `CACHE_DIR`. Sudoku scripts accept a `DIFFICULTY` environment variable (`easy` / `medium` / `hard`, default `hard`).

# Sampling & evaluation

The sampling scripts are in `scripts/sample/` and mirror `scripts/train/` one to one. They expose `CKPT_PATH`, `OUTPUT_DIR`, `CACHE_DIR`, `NUM_NODES`, `DEVICES`, `STEPS` (number of sampling steps) and `TEMPERATURE` (default 1.0).

- **OWT**: generates unconditional samples and computes their generative perplexity (`STEPS` defaults to 1024).
- **TinyGSM**: solves the GSM8K test set and writes per-example results to `OUTPUT_DIR` (`STEPS` defaults to 512).
- **Maze / Sudoku**: solves the validation puzzles and reports accuracy (`STEPS` defaults to 800 and 180). By default, they read `outputs/<setting>/<run>/checkpoints/last.ckpt`, written by the matching training script.

For OWT and TinyGSM, `CKPT_PATH` defaults to the final checkpoint of the run under `./checkpoints`, as downloaded from Hugging Face.

Example: evaluate the fine-tuned TinyGSM model with 64 sampling steps:

```bash
hf download nadjushenka12/law \
    tinygsm/finetune/law/finetune_tinygsm_law_ee_replace_p10_p90_elbo_last_only_temperature_rng_3_bs_512/2-50000.ckpt \
    --local-dir ./checkpoints
STEPS=64 bash scripts/sample/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-bs-512.sh
```

The 𝕊-FLM sampling scripts expose two extra variables: `VELOCITY` (`exact`, the default, or `sample`) and `TOPK_VELOCITY` (`-1`, the default, for the full vocabulary, or a positive integer to restrict the velocity to the top-k tokens).

# Acknowledgements

This codebase builds on a number of excellent open-source projects:

- [**𝕊-FLM**](https://github.com/jdeschena/s-flm): our training and evaluation code descends from theirs, as does the 𝕊-FLM baseline.
- [**Duo**](https://github.com/s-sahoo/duo) and [**MDLM**](https://github.com/kuleshov-group/mdlm): discrete diffusion baselines and training scaffolding.
- [**Loopholing Discrete Diffusion (LDDM)**](https://arxiv.org/abs/2510.19304): the LDDM baseline.
- [**FLM**](https://github.com/david3684/flm): Flow Language Model baseline.
- [**PUMA**](https://github.com/JaeyeonKim01/PUMA): reference for the TinyGSM data preparation.
- [**PRISM**](https://github.com/JaeyeonKim01/PRISM): reference for the Sudoku data preparation.
- [**CANDI**](https://github.com/patrickpynadath1/candi-diffusion): continuous-and-discrete diffusion baseline.
- [**HRM**](https://github.com/sapientinc/HRM): the Maze-Hard generation recipe.

# License

This codebase is distributed under the Apache-2.0 license.

# Citation
```
@misc{deschenaux2026loopholingwaylanguagemodeling,
      title={Loopholing All the Way: Language Modeling with pure Self-Conditioning}, 
      author={Adrien Deschenaux and Nadezhda Ilieva and Justin Deschenaux and Manuel Madeira and Dorina Thanou and Aurélien Lucchi and Antonio Orvieto and Caglar Gulcehre and Pascal Frossard},
      year={2026},
      eprint={2610.30068},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.30068}, 
}
```
