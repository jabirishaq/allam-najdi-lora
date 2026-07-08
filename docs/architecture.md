# Architecture

> **Plain-English summary:** we didn't retrain Saudi Arabia's AI model from
> scratch — that would need far more than $1 and a free laptop GPU. Instead
> we (1) shrank the model's memory footprint using a compression trick
> called QLoRA, and (2) trained a tiny "patch" called a LoRA adapter that
> sits on top of the frozen model and nudges its behavior toward Najdi
> dialect. The patch file is under 100MB, versus roughly 14GB for the full
> model.

## What is ALLaM-7B?

ALLaM-7B is a 7-billion-parameter large language model developed by Saudi
Arabia's **National Center for AI (NCAI)**, under the **Saudi Data and AI
Authority (SDAIA)**, in partnership with King Fahd University of Petroleum
& Minerals (KFUPM). It is trained from scratch (not fine-tuned from an
existing English-first model) on a mixed Arabic/English pretraining corpus
reported at roughly **5.2 trillion tokens**, with deliberate emphasis on
Arabic linguistic structure and cultural context rather than treating
Arabic as a secondary language bolted onto an English-centric model. This
project uses the public instruction-tuned preview checkpoint,
`humain-ai/ALLaM-7B-Instruct-preview`, as a frozen base — see the "one rule"
in [`CLAUDE.md`](../CLAUDE.md).

At 7B parameters, in native 16-bit precision, the model's weights alone
require:

```
7,000,000,000 parameters × 2 bytes/parameter ≈ 14 GB
```

That's before accounting for activations, KV cache, or (during training)
gradients and optimizer states — well beyond what a free-tier Colab T4's
15GB of VRAM can hold alongside anything else.

## What QLoRA does to memory

**QLoRA** (Quantized Low-Rank Adaptation, Dettmers et al., 2023) combines
two ideas used in this project:

1. **4-bit quantization of the frozen base model.** Instead of storing each
   weight as a 16-bit float, weights are quantized to 4-bit **NF4**
   (NormalFloat4, a data type tuned to how neural network weights are
   actually distributed) via `bitsandbytes`. This alone cuts the base
   model's memory footprint roughly 4x:

   ```
   14 GB (fp16) → ~3.5-4 GB (4-bit NF4)
   ```

   Double quantization (`bnb_4bit_use_double_quant=True`) further quantizes
   the quantization constants themselves, saving a bit more on top.

2. **Training only a small adapter on top of the frozen, quantized base**
   (see LoRA section below), rather than updating the base weights at all.
   Because the base model's weights never receive gradients, there's no
   need to keep them in a higher-precision, gradient-friendly format — they
   can stay quantized for the entire training run.

The combined effect is what makes training a 7B model feasible on a 15GB
T4 at all: base model ≈ 4GB, leaving several GB of headroom for
activations, the LoRA adapter's own gradients/optimizer state, and a small
batch.

## What LoRA adapts

**LoRA** (Low-Rank Adaptation, Hu et al., 2021) freezes the base model
entirely and instead injects small, trainable low-rank matrices alongside
specific existing weight matrices. During the forward pass, each targeted
matrix's output is the frozen base computation *plus* a small correction
computed by the adapter; only that correction is trained.

This project targets four projection matrices inside each transformer
layer's self-attention block:

- **`q_proj`** — query projection
- **`k_proj`** — key projection
- **`v_proj`** — value projection
- **`o_proj`** — output projection

These four are the standard LoRA target set for decoder-only causal LMs.
Adapting attention (rather than, say, only the MLP/feed-forward layers) is
where a *register* or *style* shift — like nudging generation toward Najdi
dialect — tends to show up most efficiently per trainable parameter,
because attention governs how the model weighs and combines context, which
is closely tied to the conversational register it produces.

## Why r=8 was chosen for T4 constraints

LoRA's rank, `r`, controls the size of the low-rank correction matrices
(and therefore how much adaptation capacity the adapter has, and how much
extra memory/compute it costs). Two constraints pushed this project to
`r=8`:

- **Memory.** On a T4 already spending most of its 15GB on the 4-bit base
  model plus activations, every doubling of `r` meaningfully increases
  adapter parameter count, gradient memory, and optimizer state. `r=8` was
  the largest rank that left comfortable headroom for a workable batch size
  (`per_device_train_batch_size=2` with gradient accumulation — see
  `scripts/train_lora.py`).
- **Task fit.** LoRA literature finds that low ranks (4-8) are typically
  sufficient for adaptations that shift *style, register, or tone* rather
  than teach the model new facts or reasoning capabilities. Dialect
  adaptation — teaching the model to respond in Najdi phrasing rather than
  MSA, using facts it largely already has — fits squarely into the
  low-rank-sufficient category, so `r=8` was not just a memory-forced
  compromise but also a reasonable capacity choice for this specific task.

`lora_alpha=16` sets the adapter's scaling factor to `alpha/r = 2`, a
commonly used ratio that keeps the adapter's contribution to each forward
pass meaningful without overwhelming the frozen base model's own signal
early in training.

## Why the adapter file is only 30-80MB, not 14GB

The adapter only stores the small low-rank matrices for the 4 targeted
projection types, across every transformer layer — nothing else. For a
matrix of shape `(d_out, d_in)`, LoRA replaces a full `d_out × d_in`
trainable update with two much smaller matrices of shape `(d_out, r)` and
`(r, d_in)`. With `r=8` and ALLaM-7B's hidden dimensions, the total
adapter parameter count across all targeted matrices and layers comes out
several orders of magnitude smaller than the base model's 7 billion
parameters — hence a ~30-80MB adapter file (depending on exact save
precision and metadata) versus ~14GB for the full base model. This is also
precisely why LoRA adapters are practical to commit to a GitHub repo (as
release assets) or share on the Hugging Face Hub without needing to
redistribute the base model itself.

## Architecture diagram

```
                     ┌───────────────────────────────────────┐
                     │   ALLaM-7B-Instruct-preview (frozen)   │
                     │   loaded in 4-bit NF4 (QLoRA)          │
                     │   ~4 GB VRAM                           │
                     │                                        │
                     │   ┌─────────────────────────────────┐  │
                     │   │  Transformer layer (× N)         │  │
                     │   │                                   │  │
                     │   │   q_proj ──┐                      │  │
                     │   │   k_proj ──┼─ frozen, 4-bit        │  │
                     │   │   v_proj ──┤                      │  │
                     │   │   o_proj ──┘                      │  │
                     │   │      │                             │  │
                     │   │      │  +  LoRA adapter (trainable) │  │
                     │   │      │     ┌───────────────┐       │  │
                     │   │      └────▶│  A (d × r)     │       │  │
                     │   │            │  B (r × d)     │       │  │
                     │   │            │  r = 8, α = 16 │       │  │
                     │   │            └───────────────┘       │  │
                     │   │                                    │  │
                     │   └─────────────────────────────────┘  │
                     └───────────────────────────────────────┘
                                       │
                                       ▼
                     output = frozen_base_output + (α/r) · B(A(x))
                     (only A, B are updated during training;
                      base weights are never modified — see CLAUDE.md)

     Saved artifacts:
       adapter/  (A, B matrices for every targeted layer) ── ~30-80 MB
       [base model weights are NOT re-saved — downloaded fresh from the
        Hub each time, since they're never modified]
```
