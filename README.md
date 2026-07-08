# ALLaM Najdi LoRA
### Teaching Saudi Arabia's sovereign LLM to speak Najdi Arabic — for $1

<!-- TODO: replace <you>/<repo> below with your actual GitHub path once pushed -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<you>/<repo>/blob/main/notebooks/full_training_colab.ipynb)
![Base model](https://img.shields.io/badge/base%20model-ALLaM--7B--Instruct--preview-2f6f4e)
![Method](https://img.shields.io/badge/method-QLoRA%20(r%3D8)-4a6fa5)
![Training cost](https://img.shields.io/badge/training%20cost-~%241-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

---

> **Plain-English summary:** Saudi Arabia built its own Arabic AI model called
> ALLaM, but like most Arabic language models it was trained mostly on
> "textbook" Modern Standard Arabic and struggles with everyday spoken dialect
> — the way people actually talk. This project teaches ALLaM to understand
> and respond in **Najdi Arabic** (the dialect of Riyadh and central Saudi
> Arabia), specifically for government-service conversations like renewing a
> license or asking about a civil affairs appointment. We did this cheaply
> (about $1 of compute) using a technique called LoRA, which trains a small
> "patch" on top of the model instead of retraining the whole thing.

## About this repo

This repository is the full, reproducible record of a small LoRA fine-tune
of ALLaM-7B, targeted at one practical vertical: **government and citizen
services**. Everything needed to reproduce the result end-to-end is
included — not just the writeup.

| | |
|---|---|
| **Base model** | [`humain-ai/ALLaM-7B-Instruct-preview`](https://huggingface.co/humain-ai/ALLaM-7B-Instruct-preview) (frozen — never modified, see [`CLAUDE.md`](CLAUDE.md)) |
| **Dialect** | Najdi Arabic (Riyadh / central Saudi Arabia) |
| **Vertical** | Government services + customer service conversations |
| **Dataset** | [`HeshamHaroon/saudi-dialect-conversations`](https://huggingface.co/datasets/HeshamHaroon/saudi-dialect-conversations) (public, filtered) |
| **Method** | QLoRA (4-bit base) + LoRA adapter (r=8, α=16) on `q/k/v/o_proj` |
| **Hardware** | Free-tier Google Colab, T4 GPU |
| **Total cost** | ~$1 |
| **Adapter size** | ~30–80 MB (vs. ~14 GB for the full base model) |
| **What's inside** | A runnable notebook, standalone CLI scripts, and full docs — see the table of contents below |

**Contents:**
[What is ALLaM?](#what-is-allam) ·
[The dialect gap](#the-dialect-gap-problem) ·
[What's different here](#what-this-project-does-differently) ·
[Cost breakdown](#cost-breakdown) ·
[Setup](#setup-instructions) ·
[Results](#results) ·
[Citation](#citation)

## What is ALLaM?

**ALLaM** (Arabic Large Language Model) is Saudi Arabia's sovereign large
language model, developed by the **National Center for AI (NCAI)**, an arm of
the **Saudi Data and AI Authority (SDAIA)**, in partnership with King
Fahd University of Petroleum & Minerals (KFUPM). ALLaM-7B is trained from
scratch on a mix of Arabic and English text (reported at roughly 5.2 trillion
tokens across pretraining stages) with a specific focus on Arabic linguistic
and cultural fidelity, rather than being an English-first model with Arabic
bolted on afterward. It was announced as part of Saudi Arabia's national AI
strategy and is positioned as sovereign infrastructure for Arabic-language
government and public-sector AI applications.

This project uses the public preview checkpoint,
[`humain-ai/ALLaM-7B-Instruct-preview`](https://huggingface.co/humain-ai/ALLaM-7B-Instruct-preview),
as the frozen base model.

## The dialect gap problem

Modern Standard Arabic (MSA/Fusha) is the formal, written register used in
news, official documents, and most Arabic NLP training corpora. It is
nobody's native spoken language — Arabic speakers grow up speaking one of
many regional dialects (Najdi, Hijazi, Egyptian, Levantine, Gulf, Maghrebi,
etc.) that differ from MSA and from each other in vocabulary, morphology, and
pragmatics. Because most large Arabic LLMs — including general-purpose
Arabic models — are trained overwhelmingly on MSA text, they tend to:

- respond in stiff, overly formal MSA even when addressed in dialect,
- misunderstand dialect-specific vocabulary and constructions,
- fail to produce dialect-appropriate responses for practical, everyday
  interactions such as government service requests.

This gap is documented and quantified by **Barmandah (2025)**,
*"Evaluating Dialectal Robustness of Arabic Large Language Models"*
([arXiv:2508.13525](https://arxiv.org/abs/2508.13525)), which shows that
Arabic LLMs — ALLaM included — measurably degrade in quality and dialectal
faithfulness when evaluated on regional dialect inputs versus MSA inputs.

## What this project does differently

Barmandah (2025) is primarily an **evaluation** paper — it measures the
dialect gap. This project is a small, reproducible **intervention**: it
fine-tunes ALLaM-7B with LoRA on real Najdi dialect conversation data to
narrow that gap for a specific, practically important vertical.

Concretely, this project differs from generic dialect fine-tuning work in
three ways:

1. **Vertical focus: government services.** Rather than fine-tuning on
   generic dialect chit-chat, the training data is filtered specifically to
   `government_services` and `customer_service` conversation categories —
   the use case most relevant to a sovereign Arabic LLM deployed in public
   sector products.
2. **Fully open dataset.** Training data comes entirely from the public
   [`HeshamHaroon/saudi-dialect-conversations`](https://huggingface.co/datasets/HeshamHaroon/saudi-dialect-conversations)
   dataset on Hugging Face — no proprietary or scraped data, so the pipeline
   is reproducible by anyone.
3. **Near-zero cost, fully documented pipeline.** The entire run — data prep,
   training, and evaluation — runs on a free-tier Colab T4 GPU and is
   captured end-to-end in a single notebook plus standalone scripts, so the
   result is reproducible for the cost of an afternoon and about a dollar of
   compute.

## Cost breakdown

| Item | Cost |
|---|---|
| Dataset (`HeshamHaroon/saudi-dialect-conversations`, HF Hub) | $0 |
| Base model (`humain-ai/ALLaM-7B-Instruct-preview`, HF Hub) | $0 |
| Compute (Google Colab, T4 GPU, ~45 min training) | ~$1 (Colab compute units) |
| **Total** | **~$1** |

## Setup instructions

1. **Get access to the base model.** Create a Hugging Face account, accept the
   license for
   [`humain-ai/ALLaM-7B-Instruct-preview`](https://huggingface.co/humain-ai/ALLaM-7B-Instruct-preview),
   and generate a read-scoped access token.
2. **Clone this repository** and, if running locally/standalone, install
   dependencies: `pip install -r requirements.txt`.
3. **Prepare the data** by running `python scripts/prepare_data.py` (or Cell
   4 of the notebook), which downloads and filters
   `HeshamHaroon/saudi-dialect-conversations` into `train.jsonl` / `eval.jsonl`.
4. **Train the adapter** by running `python scripts/train_lora.py` (or Cells
   5–8 of the notebook) on a GPU with at least 15GB VRAM (a free Colab T4
   works). This produces a LoRA adapter directory of ~30-80MB.
5. **Evaluate** by running `python scripts/evaluate.py`, which loads the base
   model and the fine-tuned adapter side by side and compares their outputs
   on held-out Najdi government-services prompts.

See [`docs/setup.md`](docs/setup.md) for the detailed, click-by-click version
of these steps, including exact Colab menu paths and fixes for the three
errors we actually hit.

## Results

Full numbers and analysis are in [`docs/results.md`](docs/results.md).
Summary — training and validation loss over 3 epochs:

| Epoch | Training loss | Validation loss |
|---|---|---|
| 0 | 2.037 | 1.865 |
| 1 | 1.746 | 1.810 |
| 2 | 1.517 | 1.821 |

Sample qualitative comparison (base vs. fine-tuned) on a Najdi
government-services prompt:

| Question (Najdi) | Base ALLaM-7B | Fine-tuned (this project) |
|---|---|---|
| "وش الأوراق اللي أحتاجها عشان أجدد الإقامة؟" | Formal MSA response, generic document list, no dialect register | Shorter, dialect-consistent response using Najdi function words and register appropriate to a citizen-service chat |

Full transcripts for all 3 evaluation questions are in
[`docs/results.md`](docs/results.md).

## Citation

If you use this repository, please cite both the underlying dialectal
robustness research this project builds on and this project itself:

```bibtex
@article{barmandah2025dialectal,
  title   = {Evaluating Dialectal Robustness of Arabic Large Language Models},
  author  = {Barmandah, Ahmad},
  journal = {arXiv preprint arXiv:2508.13525},
  year    = {2025}
}

@misc{allam_najdi_lora_2026,
  title  = {ALLaM Najdi LoRA: Dialect Adaptation for Government Services},
  author = {s445001043@uqu.edu.sa},
  year   = {2026},
  note   = {LoRA fine-tune of humain-ai/ALLaM-7B-Instruct-preview on Najdi Arabic government-services data},
  url    = {https://github.com/<you>/<repo>}  % TODO: fill in once pushed
}
```

For questions about this project, contact **s445001043@uqu.edu.sa**.
