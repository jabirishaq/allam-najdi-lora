"""
train_lora.py

Standalone QLoRA fine-tuning script for adapting ALLaM-7B-Instruct-preview
to Najdi Arabic government-services conversations.

This mirrors Cells 5-8 of notebooks/full_training_colab.ipynb, but is
written to run outside Colab (e.g. on a local GPU box or a cloud instance)
given `train.jsonl` / `eval.jsonl` produced by prepare_data.py.

Requires a CUDA GPU with >= ~15GB VRAM (this project was trained and
tested on a free-tier Colab T4).

Run:
    export HF_TOKEN=hf_xxx   # must have accepted the ALLaM-7B-Instruct-preview license
    python scripts/train_lora.py --data-dir ./data --output-dir ./adapter
"""

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

BASE_MODEL_ID = "humain-ai/ALLaM-7B-Instruct-preview"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune ALLaM-7B on Najdi dialect data.")
    parser.add_argument("--data-dir", default="./data", help="Directory containing train.jsonl / eval.jsonl.")
    parser.add_argument("--output-dir", default="./adapter", help="Where to save the trained LoRA adapter.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    return parser.parse_args()


def build_quantization_config() -> BitsAndBytesConfig:
    """
    4-bit quantization config (QLoRA). This is what makes it possible to
    fit a 7B-parameter model on a 15GB T4 at all: the frozen base weights
    are stored in 4-bit NF4 instead of 16-bit, cutting the base model's
    memory footprint from ~14GB to ~4GB. Only the small LoRA adapter
    matrices are trained in higher precision (bfloat16 compute dtype
    below), so quality loss from quantizing the frozen base is minimal.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # NormalFloat4: the QLoRA paper's quantization
                                             # scheme, better suited to weight distributions
                                             # than plain int4.
        bnb_4bit_compute_dtype=torch.bfloat16,  # matmuls are de-quantized to bf16 on the fly;
                                                 # T4 supports bf16 compute even though it
                                                 # predates native bf16 tensor cores on newer GPUs.
        bnb_4bit_use_double_quant=True,     # quantizes the quantization constants themselves,
                                             # saving a further ~0.4 bits/parameter -- worth it
                                             # given how VRAM-constrained a T4 is.
    )


def build_lora_config() -> LoraConfig:
    """
    LoRA adapter config.

    r=8: the rank of the low-rank update matrices. Higher r (16, 32, 64)
    captures more adaptation capacity but costs more trainable parameters
    and VRAM for optimizer states. On a T4 with only ~15GB VRAM and a 4-bit
    base model already using a meaningful chunk of that budget, r=8 was
    the largest rank that kept training comfortably within memory with
    room for a reasonable batch size. This is also a dialect-adaptation
    task, not a new-capability task -- LoRA literature (Hu et al., 2021)
    finds low ranks like 4-8 are usually sufficient for style/register
    shifts, as opposed to teaching wholly new knowledge or reasoning
    behavior, which is closer to what we're doing here (nudging generation
    register towards Najdi dialect) than the latter.
    alpha=16: LoRA scaling factor, applied as alpha/r to the adapter output.
    alpha=16 with r=8 gives a scaling factor of 2, a commonly-used ratio
    (alpha = 2*r) that keeps the adapter's contribution meaningful relative
    to the frozen base without overwhelming it early in training.
    """
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,       # light regularization; the fine-tuning set is small
                                  # (a filtered subset of one dataset), so some dropout
                                  # helps avoid overfitting to specific phrasings.
        bias="none",             # standard LoRA practice: don't adapt bias terms,
                                  # keep the adapter strictly low-rank.
        task_type="CAUSAL_LM",
        # Target modules: the four attention projection matrices.
        # q_proj / k_proj / v_proj / o_proj are where LoRA adapts the
        # attention mechanism's query, key, value, and output projections.
        # This is the standard LoRA target set for decoder-only causal LMs:
        # adapting attention (rather than, say, only the MLP layers) is
        # where dialect/register shifts -- which are largely about *how*
        # the model attends to and reweights context -- show up most
        # efficiently per trainable parameter.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def main() -> None:
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. You need a Hugging Face token with access to "
            f"{BASE_MODEL_ID} (accept its license on the model page first)."
        )

    train_path = os.path.join(args.data_dir, "train.jsonl")
    eval_path = os.path.join(args.data_dir, "eval.jsonl")
    dataset = load_dataset(
        "json",
        data_files={"train": train_path, "eval": eval_path},
    )

    print(f"Loading base model in 4-bit: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=hf_token)
    if tokenizer.pad_token is None:
        # Many causal LMs (ALLaM included) ship without a dedicated pad
        # token; reuse EOS as pad since we only pad, never generate it
        # mid-sequence during training.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=build_quantization_config(),
        device_map="auto",
        token=hf_token,
    )

    # Attach the LoRA adapter. From this point on, the base model's
    # weights are frozen (see CLAUDE.md: never modify base model weights
    # directly) -- only the small adapter matrices below are trainable.
    lora_config = build_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def format_example(example: dict) -> str:
        # Simple instruct-style template. Kept plain/minimal rather than
        # ALLaM's full chat template because the source dataset is
        # single-turn prompt/response pairs, not multi-turn chats.
        return f"### السؤال:\n{example['prompt']}\n\n### الرد:\n{example['response']}"

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        # T4-optimized settings below: a T4 has no bf16 tensor cores and
        # only 15GB VRAM, so we train in fp16 (not bf16) and keep the
        # per-device batch tiny, compensating with gradient accumulation
        # to reach a reasonable effective batch size without OOMing.
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,   # effective batch size = 2 * 4 = 8
        fp16=True,
        gradient_checkpointing=True,     # trades compute for memory by not
                                          # caching all activations -- necessary
                                          # headroom on a 15GB card running a
                                          # 7B model plus optimizer states.
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        max_length=512,       # Najdi government-services Q&A pairs in the
                                # source dataset are short; 512 tokens covers
                                # them with margin without wasting VRAM on
                                # padding to a longer sequence length.
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        formatting_func=format_example,
    )

    trainer.train()

    # Save only the adapter (a few tens of MB), not the base model.
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
