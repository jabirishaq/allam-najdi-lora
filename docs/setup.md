# Setup Guide

> **Plain-English summary:** this page walks you through everything you need
> to click and type, in order, to get from "nothing installed" to "training
> is running in Google Colab" — including the exact web addresses to visit
> and the fixes for the three mistakes we actually made while building this.

This guide assumes no prior Colab or Hugging Face experience. If you've
already got a Hugging Face account and a Colab habit, skim the exact-value
boxes and skip the rest.

## Step 1: Hugging Face account + license accept

1. Create a free account at [huggingface.co/join](https://huggingface.co/join)
   if you don't have one.
2. Go to the base model page:
   [huggingface.co/humain-ai/ALLaM-7B-Instruct-preview](https://huggingface.co/humain-ai/ALLaM-7B-Instruct-preview)
3. If the page shows a license/access-request gate, click **"Agree and
   access repository"** (exact label may vary slightly by model card) and
   accept the terms. Access is typically granted immediately for this model.
4. Create an access token: go to
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   → **"Create new token"** → choose the **"Read"** role → name it (e.g.
   `allam-najdi-lora`) → **"Create token"**.
5. Copy the token (starts with `hf_...`). You'll paste it into the notebook
   in Step 5 — don't commit it to any file in this repo.

## Step 2: Google Colab setup

1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Sign in with the same Google account you'll use for Drive (Step 4 needs
   this to match).
3. Open this repo's notebook one of two ways:
   - **From GitHub:** in Colab, go to **File → Open notebook → GitHub** tab,
     paste this repository's URL, and select
     `notebooks/full_training_colab.ipynb`.
   - **Upload directly:** **File → Upload notebook**, and select the
     `.ipynb` file from a local clone of this repo.

## Step 3: GPU selection (T4 free tier)

1. In the open notebook, go to **Runtime → Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU**.
3. Under **Runtime shape**, leave it on the free-tier default (not
   "High-RAM" — not required for this project and may consume paid compute
   units unnecessarily).
4. Click **Save**.
5. Confirm you got a T4 by running `!nvidia-smi` in a scratch cell after
   connecting — the output should list `Tesla T4`. Colab's free tier does
   not always guarantee T4 availability; if you're assigned a different GPU
   or none at all, wait and reconnect, or try again later (usage quotas
   reset on a rolling basis).

## Step 4: Drive mount

1. Run Cell 2 of the notebook (`drive.mount('/content/drive')`).
2. A popup/link will ask you to authorize Colab to access your Google
   Drive. Choose the Google account matching Step 2, and click **Allow**.
3. Confirm the mount succeeded — the cell should print the project
   directory path (`/content/drive/MyDrive/allam-najdi-lora`) without
   errors.

## Step 5: Run notebook cells in order

Run the notebook's 9 code cells top to bottom, in order — later cells
depend on state (model, tokenizer, dataset) created by earlier ones:

1. Install dependencies
2. Mount Google Drive
3. Hugging Face authentication (paste your token from Step 1 when prompted)
4. Download and prepare the dataset
5. Load ALLaM-7B in 4-bit (QLoRA)
6. Attach the LoRA adapter
7. Train (takes roughly 30–45 minutes on a T4 for 3 epochs over the
   filtered dataset)
8. Save the adapter to Drive
9. Evaluate base vs. fine-tuned side by side

## Common errors and fixes

These are the three errors we actually hit while building this project, in
the order you're likely to encounter them.

### 1. `OSError: You are trying to access a gated repo` when loading the model

**Cause:** either the Hugging Face token wasn't picked up, or the license
for `humain-ai/ALLaM-7B-Instruct-preview` wasn't accepted on the model page.

**Fix:**
- Re-check Step 1.3 — make sure you clicked through the license accept on
  the model page itself, not just created a token.
- Re-run the authentication cell (Cell 3) and make sure `login()` completes
  without error before moving on.
- Confirm the token has at least **Read** access (Step 1.4).

### 2. `CUDA out of memory` during training (Cell 7)

**Cause:** the T4's 15GB VRAM is fully consumed — usually because a
previous cell's model/session wasn't cleared before re-running, or the
runtime silently downgraded from T4 to a smaller GPU.

**Fix:**
- **Runtime → Restart session**, then re-run cells 1–3 and 5–7 (you can
  skip re-downloading the dataset in Cell 4 if `train.jsonl`/`eval.jsonl`
  already exist on Drive).
- Re-confirm you're on a T4 via `!nvidia-smi` (Step 3.5).
- If it persists, lower `per_device_train_batch_size` in Cell 7's
  `SFTConfig` from 2 to 1 (gradient accumulation will compensate for the
  smaller per-step batch).

### 3. `RuntimeError: expected scalar type Half but found BFloat16` (or similar dtype mismatch) during training

**Cause:** a T4 GPU does not have native bf16 tensor cores. Mixing a
`bnb_4bit_compute_dtype=torch.bfloat16` quantization config with
`fp16=True` training args is fine (compute dtype for de-quantized matmuls
is independent of the trainer's mixed-precision mode), but if you change
the trainer to `bf16=True` instead of `fp16=True` on a T4, you'll hit dtype
mismatches or silently degraded performance.

**Fix:** on a T4, always keep `fp16=True` (not `bf16=True`) in the
`SFTConfig` in Cell 7, exactly as shipped in this notebook. Leave the
`bnb_4bit_compute_dtype=torch.bfloat16` in the quantization config as-is —
that setting is fine on T4 because it's only used transiently to de-quantize
4-bit weights for each matmul, not as the trainer's overall precision mode.
