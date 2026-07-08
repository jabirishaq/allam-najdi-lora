# CLAUDE.md

Guidance for Claude Code (or any AI coding agent) working in this repository.

## Project purpose

This repository contains a LoRA fine-tune of **ALLaM-7B-Instruct-preview**
(SDAIA's sovereign Arabic LLM) adapted to the **Najdi Arabic dialect** for
government-services conversational use cases (e.g. Absher, Tawakkalna-style
citizen service chat). The goal is to reduce the "dialect gap" documented in
Barmandah (2025, arXiv:2508.13525) — the tendency of Arabic LLMs trained
predominantly on Modern Standard Arabic (MSA) to respond stiffly or
incorrectly when addressed in regional dialect.

This is a completed research project. The repository exists to make the
training pipeline, data preparation, and evaluation reproducible and citable.

## Stack

- **Python** 3.10+
- **transformers** — model loading, tokenization, generation
- **peft** — LoRA adapter definition, attachment, and merging
- **bitsandbytes** — 4-bit (QLoRA) quantization for T4-constrained memory
- **trl** — `SFTTrainer` for supervised fine-tuning
- **datasets** — HF Datasets loading/filtering pipeline
- **Google Colab** — training environment (free-tier T4 GPU, 15GB VRAM)

## Key files

| File | What it does |
|---|---|
| `notebooks/full_training_colab.ipynb` | End-to-end Colab notebook: install deps, mount Drive, auth, prepare data, load model in 4-bit, attach LoRA, train, save, evaluate. This is the canonical, runnable record of the experiment. |
| `scripts/prepare_data.py` | Standalone script that downloads `HeshamHaroon/saudi-dialect-conversations`, filters to `government_services` + `customer_service` categories, and writes `train.jsonl` / `eval.jsonl`. Runs outside Colab. |
| `scripts/train_lora.py` | Standalone QLoRA training script — loads ALLaM-7B in 4-bit, attaches a LoRA adapter (r=8, alpha=16), trains 3 epochs, saves the adapter. |
| `scripts/evaluate.py` | Loads base ALLaM-7B and the fine-tuned adapter, runs the same 3 Najdi government-services prompts through both, prints a side-by-side comparison. |
| `docs/setup.md` | Step-by-step environment setup (HF account, Colab, GPU selection, Drive mount). |
| `docs/results.md` | Training/validation loss tables and qualitative before/after outputs. |
| `docs/architecture.md` | Explains ALLaM-7B, QLoRA memory savings, which projection matrices LoRA adapts, and why r=8 was chosen. |
| `requirements.txt` | Pinned dependency versions used for the actual training run. |

## The one rule

**Never modify base model weights directly.**

All adaptation happens through the LoRA adapter (`peft`), which is trained,
saved, and distributed as a small (~30-80MB) delta on top of the frozen,
4-bit-quantized ALLaM-7B base. The base model is loaded read-only in every
script and notebook. If a change requires altering base weights instead of
adapter weights, stop and reconsider the approach — full fine-tuning of a 7B
model is out of scope for this project's compute budget (~$1, free-tier T4)
and defeats the purpose of the adapter-based, easily-shareable artifact this
repo produces.
