# Scripts

One training script (`train/`) and one sampling script (`sample/`) per run, with the same path in both folders. For OWT and TinyGSM the folders mirror the Hugging Face repo [`nadjushenka12/law`](https://huggingface.co/nadjushenka12/law), with one script per released checkpoint. Run the scripts from anywhere; they `cd` to the repository root. See the [main README](../README.md) for the environment variables they read.

Run names describe the setup: `ee-replace` is the expected-embedding carry (`hidden-replace` carries the hidden state, `ee-add` adds the carry to the input as in LDDM); `p10-p90` means 1 pass with probability 0.1 and 2 passes with probability 0.9 (`3-pass-p10-p0-p90`: 1 or 3 passes); `ce` / `elbo` is the loss; `last-only` puts the loss on the last pass only; `temperature-rng-3` is the carry temperature regularization with range 3; `mode-emb` adds the mode embedding; `backprop2` backpropagates through both passes.

## OpenWebText

LAW:

| Run | Description | Scripts |
|---|---|---|
| `owt-law-ee-replace-p10-p90-elbo-last-only` | ELBO loss | [train](train/owt/law/owt-law-ee-replace-p10-p90-elbo-last-only.sh) · [sample](sample/owt/law/owt-law-ee-replace-p10-p90-elbo-last-only.sh) |
| `owt-law-ee-replace-p10-p90-ce-last-only` | CE loss | [train](train/owt/law/owt-law-ee-replace-p10-p90-ce-last-only.sh) · [sample](sample/owt/law/owt-law-ee-replace-p10-p90-ce-last-only.sh) |
| `owt-law-ee-replace-p10-p90-ce-last-only-temperature-rng-4` | CE loss, temperature range 4 | [train](train/owt/law/owt-law-ee-replace-p10-p90-ce-last-only-temperature-rng-4.sh) · [sample](sample/owt/law/owt-law-ee-replace-p10-p90-ce-last-only-temperature-rng-4.sh) |
| `owt-law-ee-replace-p10-p90-ce-last-only-temperature-rng-3-mode-emb` | **Main result.** CE loss, temperature range 3, mode embedding | [train](train/owt/law/owt-law-ee-replace-p10-p90-ce-last-only-temperature-rng-3-mode-emb.sh) · [sample](sample/owt/law/owt-law-ee-replace-p10-p90-ce-last-only-temperature-rng-3-mode-emb.sh) |


LDDM:

| Run | Description | Scripts |
|---|---|---|
| `owt-lddm-m` | LDDM-M (masked) | [train](train/owt/lddm/owt-lddm-m.sh) · [sample](sample/owt/lddm/owt-lddm-m.sh) |
| `owt-lddm-u` | LDDM-U (uniform) | [train](train/owt/lddm/owt-lddm-u.sh) · [sample](sample/owt/lddm/owt-lddm-u.sh) |
| `owt-law-ee-add-p10-p90-elbo-last-only` | LDDM-U with an expected-embedding carry | [train](train/owt/lddm/owt-law-ee-add-p10-p90-elbo-last-only.sh) · [sample](sample/owt/lddm/owt-law-ee-add-p10-p90-elbo-last-only.sh) |
| `owt-law-ee-add-p10-p90-elbo-last-only-temperature-rng-3` | LDDM-U with an expected-embedding carry, temperature range 3 | [train](train/owt/lddm/owt-law-ee-add-p10-p90-elbo-last-only-temperature-rng-3.sh) · [sample](sample/owt/lddm/owt-law-ee-add-p10-p90-elbo-last-only-temperature-rng-3.sh) |


Baselines. They follow the 𝕊-FLM recipe (1M-step schedule); as in the paper, we use the checkpoint at step 400k. The learning rate is constant after warmup.

| Run | Description | Scripts |
|---|---|---|
| `ar` | Autoregressive | [train](train/owt/baselines/ar.sh) · [sample](sample/owt/baselines/ar.sh) |
| `mdlm` | MDLM | [train](train/owt/baselines/mdlm.sh) · [sample](sample/owt/baselines/mdlm.sh) |
| `duo` | Duo | [train](train/owt/baselines/duo.sh) · [sample](sample/owt/baselines/duo.sh) |
| `flm` | FLM | [train](train/owt/baselines/flm.sh) · [sample](sample/owt/baselines/flm.sh) |
| `sfm-std` | 𝕊-FLM, standard DiT backbone | [train](train/owt/baselines/sfm-std.sh) · [sample](sample/owt/baselines/sfm-std.sh) |
| `sfm-ngpt` | 𝕊-FLM, 𝕊-arch backbone | [train](train/owt/baselines/sfm-ngpt.sh) · [sample](sample/owt/baselines/sfm-ngpt.sh) |


## TinyGSM, from scratch

Carry representation × loss:

| Run | Description | Scripts |
|---|---|---|
| `tinygsm-law-ee-replace-p10-p90-ce-last-only` | Expected-embedding carry, CE loss | [train](train/tinygsm/pretrain/law/01_carry_x_loss/tinygsm-law-ee-replace-p10-p90-ce-last-only.sh) · [sample](sample/tinygsm/pretrain/law/01_carry_x_loss/tinygsm-law-ee-replace-p10-p90-ce-last-only.sh) |
| `tinygsm-law-ee-replace-p10-p90-elbo-last-only` | Expected-embedding carry, ELBO loss | [train](train/tinygsm/pretrain/law/01_carry_x_loss/tinygsm-law-ee-replace-p10-p90-elbo-last-only.sh) · [sample](sample/tinygsm/pretrain/law/01_carry_x_loss/tinygsm-law-ee-replace-p10-p90-elbo-last-only.sh) |
| `tinygsm-law-hidden-replace-p10-p90-ce-last-only` | Hidden-state carry, CE loss | [train](train/tinygsm/pretrain/law/01_carry_x_loss/tinygsm-law-hidden-replace-p10-p90-ce-last-only.sh) · [sample](sample/tinygsm/pretrain/law/01_carry_x_loss/tinygsm-law-hidden-replace-p10-p90-ce-last-only.sh) |
| `tinygsm-law-hidden-replace-p10-p90-elbo-last-only` | Hidden-state carry, ELBO loss | [train](train/tinygsm/pretrain/law/01_carry_x_loss/tinygsm-law-hidden-replace-p10-p90-elbo-last-only.sh) · [sample](sample/tinygsm/pretrain/law/01_carry_x_loss/tinygsm-law-hidden-replace-p10-p90-elbo-last-only.sh) |


Temperature range × mode embedding:

| Run | Description | Scripts |
|---|---|---|
| `tinygsm-law-ee-replace-p10-p90-elbo-last-only-mode-emb` | Mode embedding | [train](train/tinygsm/pretrain/law/02_temperature_x_mode_embedding/tinygsm-law-ee-replace-p10-p90-elbo-last-only-mode-emb.sh) · [sample](sample/tinygsm/pretrain/law/02_temperature_x_mode_embedding/tinygsm-law-ee-replace-p10-p90-elbo-last-only-mode-emb.sh) |
| `tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3` | **Main result.** Temperature range 3 | [train](train/tinygsm/pretrain/law/02_temperature_x_mode_embedding/tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3.sh) · [sample](sample/tinygsm/pretrain/law/02_temperature_x_mode_embedding/tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3.sh) |
| `tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-mode-emb` | Temperature range 3, mode embedding | [train](train/tinygsm/pretrain/law/02_temperature_x_mode_embedding/tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-mode-emb.sh) · [sample](sample/tinygsm/pretrain/law/02_temperature_x_mode_embedding/tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-mode-emb.sh) |
| `tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-4` | Temperature range 4 | [train](train/tinygsm/pretrain/law/02_temperature_x_mode_embedding/tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-4.sh) · [sample](sample/tinygsm/pretrain/law/02_temperature_x_mode_embedding/tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-4.sh) |
| `tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-4-mode-emb` | Temperature range 4, mode embedding | [train](train/tinygsm/pretrain/law/02_temperature_x_mode_embedding/tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-4-mode-emb.sh) · [sample](sample/tinygsm/pretrain/law/02_temperature_x_mode_embedding/tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-4-mode-emb.sh) |


Pass count × multi-pass probability:

| Run | Description | Scripts |
|---|---|---|
| `tinygsm-law-ee-replace-p25-p75-elbo-last-only-temperature-rng-3` | 2 passes w.p. 0.75 (200k steps) | [train](train/tinygsm/pretrain/law/03_pass_count_x_probability/tinygsm-law-ee-replace-p25-p75-elbo-last-only-temperature-rng-3.sh) · [sample](sample/tinygsm/pretrain/law/03_pass_count_x_probability/tinygsm-law-ee-replace-p25-p75-elbo-last-only-temperature-rng-3.sh) |
| `tinygsm-law-ee-replace-3-pass-p10-p0-p90-elbo-last-only-temperature-rng-3` | 3 passes w.p. 0.9 (156k steps) | [train](train/tinygsm/pretrain/law/03_pass_count_x_probability/tinygsm-law-ee-replace-3-pass-p10-p0-p90-elbo-last-only-temperature-rng-3.sh) · [sample](sample/tinygsm/pretrain/law/03_pass_count_x_probability/tinygsm-law-ee-replace-3-pass-p10-p0-p90-elbo-last-only-temperature-rng-3.sh) |
| `tinygsm-law-ee-replace-3-pass-p25-p0-p75-elbo-last-only-temperature-rng-3` | 3 passes w.p. 0.75 (167k steps) | [train](train/tinygsm/pretrain/law/03_pass_count_x_probability/tinygsm-law-ee-replace-3-pass-p25-p0-p75-elbo-last-only-temperature-rng-3.sh) · [sample](sample/tinygsm/pretrain/law/03_pass_count_x_probability/tinygsm-law-ee-replace-3-pass-p25-p0-p75-elbo-last-only-temperature-rng-3.sh) |
| `tinygsm-law-ee-replace-4-pass-p10-p0-p0-p90-elbo-last-only-temperature-rng-3` | 4 passes w.p. 0.9 (132k steps) | [train](train/tinygsm/pretrain/law/03_pass_count_x_probability/tinygsm-law-ee-replace-4-pass-p10-p0-p0-p90-elbo-last-only-temperature-rng-3.sh) · [sample](sample/tinygsm/pretrain/law/03_pass_count_x_probability/tinygsm-law-ee-replace-4-pass-p10-p0-p0-p90-elbo-last-only-temperature-rng-3.sh) |
| `tinygsm-law-ee-replace-4-pass-p25-p0-p0-p75-elbo-last-only-temperature-rng-3` | 4 passes w.p. 0.75 (143k steps) | [train](train/tinygsm/pretrain/law/03_pass_count_x_probability/tinygsm-law-ee-replace-4-pass-p25-p0-p0-p75-elbo-last-only-temperature-rng-3.sh) · [sample](sample/tinygsm/pretrain/law/03_pass_count_x_probability/tinygsm-law-ee-replace-4-pass-p25-p0-p0-p75-elbo-last-only-temperature-rng-3.sh) |


Backpropagation through the loop:

| Run | Description | Scripts |
|---|---|---|
| `tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-backprop2` | 2 passes w.p. 0.9 (132k steps) | [train](train/tinygsm/pretrain/law/04_backprop_through_loop/tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-backprop2.sh) · [sample](sample/tinygsm/pretrain/law/04_backprop_through_loop/tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-backprop2.sh) |
| `tinygsm-law-ee-replace-p0-p100-elbo-last-only-temperature-rng-3-backprop2` | Always 2 passes (125k steps) | [train](train/tinygsm/pretrain/law/04_backprop_through_loop/tinygsm-law-ee-replace-p0-p100-elbo-last-only-temperature-rng-3-backprop2.sh) · [sample](sample/tinygsm/pretrain/law/04_backprop_through_loop/tinygsm-law-ee-replace-p0-p100-elbo-last-only-temperature-rng-3-backprop2.sh) |


LDDM with an expected-embedding carry:

| Run | Description | Scripts |
|---|---|---|
| `tinygsm-law-ee-add-p10-p90-elbo-last-only` | LDDM-U with an expected-embedding carry | [train](train/tinygsm/pretrain/lddm/tinygsm-law-ee-add-p10-p90-elbo-last-only.sh) · [sample](sample/tinygsm/pretrain/lddm/tinygsm-law-ee-add-p10-p90-elbo-last-only.sh) |
| `tinygsm-law-ee-add-p10-p90-elbo-last-only-temperature-rng-3` | LDDM-U with an expected-embedding carry, temperature range 3 | [train](train/tinygsm/pretrain/lddm/tinygsm-law-ee-add-p10-p90-elbo-last-only-temperature-rng-3.sh) · [sample](sample/tinygsm/pretrain/lddm/tinygsm-law-ee-add-p10-p90-elbo-last-only-temperature-rng-3.sh) |


## TinyGSM, fine-tuning Duo with LAW

These runs start from Duo trained for 250k steps on TinyGSM and train for 50k more steps with LAW. See [Fine-tuning Duo with LAW](../README.md#fine-tuning-duo-with-law-tinygsm) for the Duo checkpoint.

| Run | Description | Scripts |
|---|---|---|
| `finetune-tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-bs-64` | 2 passes w.p. 0.9, batch size 64 | [train](train/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-bs-64.sh) · [sample](sample/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-bs-64.sh) |
| `finetune-tinygsm-law-ee-replace-4-pass-p25-p0-p0-p75-elbo-last-only-temperature-rng-3-bs-64` | 4 passes w.p. 0.75, batch size 64 | [train](train/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-4-pass-p25-p0-p0-p75-elbo-last-only-temperature-rng-3-bs-64.sh) · [sample](sample/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-4-pass-p25-p0-p0-p75-elbo-last-only-temperature-rng-3-bs-64.sh) |
| `finetune-tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-bs-128` | 2 passes w.p. 0.9, batch size 128 | [train](train/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-bs-128.sh) · [sample](sample/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-bs-128.sh) |
| `finetune-tinygsm-law-ee-replace-4-pass-p25-p0-p0-p75-elbo-last-only-temperature-rng-3-bs-128` | 4 passes w.p. 0.75, batch size 128 | [train](train/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-4-pass-p25-p0-p0-p75-elbo-last-only-temperature-rng-3-bs-128.sh) · [sample](sample/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-4-pass-p25-p0-p0-p75-elbo-last-only-temperature-rng-3-bs-128.sh) |
| `finetune-tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-bs-256` | 2 passes w.p. 0.9, batch size 256 | [train](train/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-bs-256.sh) · [sample](sample/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-bs-256.sh) |
| `finetune-tinygsm-law-ee-replace-4-pass-p25-p0-p0-p75-elbo-last-only-temperature-rng-3-bs-256` | 4 passes w.p. 0.75, batch size 256 | [train](train/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-4-pass-p25-p0-p0-p75-elbo-last-only-temperature-rng-3-bs-256.sh) · [sample](sample/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-4-pass-p25-p0-p0-p75-elbo-last-only-temperature-rng-3-bs-256.sh) |
| `finetune-tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-bs-512` | **Main result.** 2 passes w.p. 0.9, batch size 512 | [train](train/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-bs-512.sh) · [sample](sample/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-p10-p90-elbo-last-only-temperature-rng-3-bs-512.sh) |
| `finetune-tinygsm-law-ee-replace-4-pass-p25-p0-p0-p75-elbo-last-only-temperature-rng-3-bs-512` | 4 passes w.p. 0.75, batch size 512 | [train](train/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-4-pass-p25-p0-p0-p75-elbo-last-only-temperature-rng-3-bs-512.sh) · [sample](sample/tinygsm/finetune/law/finetune-tinygsm-law-ee-replace-4-pass-p25-p0-p0-p75-elbo-last-only-temperature-rng-3-bs-512.sh) |


## Maze

| Run | Description | Scripts |
|---|---|---|
| `maze-law-ee-replace-p25-p75-elbo-last-only` | NELBO loss, 2 passes w.p. 0.75 (80,000 steps) | [train](train/maze/maze-law-ee-replace-p25-p75-elbo-last-only.sh) · [sample](sample/maze/maze-law-ee-replace-p25-p75-elbo-last-only.sh) |
| `maze-law-ee-replace-p25-p75-elbo-last-only-temperature-rng-2` | **Main result.** NELBO loss, 2 passes w.p. 0.75, temperature range 2 (80,000 steps) | [train](train/maze/maze-law-ee-replace-p25-p75-elbo-last-only-temperature-rng-2.sh) · [sample](sample/maze/maze-law-ee-replace-p25-p75-elbo-last-only-temperature-rng-2.sh) |


## Sudoku

Set the difficulty with `DIFFICULTY` (`easy` / `medium` / `hard`, default `hard`).

| Run | Description | Scripts |
|---|---|---|
| `sudoku-law-ee-replace-p10-p90-elbo-last-only` | NELBO loss, 2 passes w.p. 0.9 (15,385 steps) | [train](train/sudoku/sudoku-law-ee-replace-p10-p90-elbo-last-only.sh) · [sample](sample/sudoku/sudoku-law-ee-replace-p10-p90-elbo-last-only.sh) |
| `sudoku-law-ee-replace-p10-p90-ce-last-only` | CE loss, 2 passes w.p. 0.9 (15,385 steps) | [train](train/sudoku/sudoku-law-ee-replace-p10-p90-ce-last-only.sh) · [sample](sample/sudoku/sudoku-law-ee-replace-p10-p90-ce-last-only.sh) |
| `sudoku-law-ee-replace-p25-p75-ce-last-only` | CE loss, 2 passes w.p. 0.75 (16,000 steps) | [train](train/sudoku/sudoku-law-ee-replace-p25-p75-ce-last-only.sh) · [sample](sample/sudoku/sudoku-law-ee-replace-p25-p75-ce-last-only.sh) |
| `sudoku-law-ee-replace-p10-p90-ce-last-only-temperature-rng-2` | **Main result.** CE loss, 2 passes w.p. 0.9, temperature range 2 (15,385 steps) | [train](train/sudoku/sudoku-law-ee-replace-p10-p90-ce-last-only-temperature-rng-2.sh) · [sample](sample/sudoku/sudoku-law-ee-replace-p10-p90-ce-last-only-temperature-rng-2.sh) |
