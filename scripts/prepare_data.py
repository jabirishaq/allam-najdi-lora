"""
prepare_data.py

Standalone data preparation script for the ALLaM Najdi LoRA project.
Runs outside Colab -- all it needs is `pip install -r requirements.txt`
and a Hugging Face token with read access (only required if the source
dataset is gated; HeshamHaroon/saudi-dialect-conversations is public, so
no token is strictly needed, but we support one via HF_TOKEN for
consistency with the training script).

What this script does:
  1. Downloads the `HeshamHaroon/saudi-dialect-conversations` dataset from
     the Hugging Face Hub.
  2. Filters it down to the two conversation categories relevant to this
     project's vertical: `government_services` and `customer_service`.
     These are the categories closest to real citizen-facing interactions
     (renewing documents, asking about appointments, service complaints),
     which is the use case this fine-tune targets.
  3. Splits the filtered examples into a train set and a small held-out
     eval set.
  4. Writes both splits to disk as JSONL (`train.jsonl`, `eval.jsonl`) in
     a simple {"prompt": ..., "response": ...} schema that `trl`'s
     SFTTrainer can consume directly.

Run:
    python scripts/prepare_data.py --output-dir ./data
"""

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset

# Categories we care about for this project's government-services vertical.
# The source dataset labels each conversation with a `category` field;
# these two are the ones that map to citizen/government interactions.
TARGET_CATEGORIES = {"government_services", "customer_service"}

# Fraction of the filtered data held out for evaluation. Kept small because
# the filtered subset itself is already a small slice of the full dataset.
EVAL_FRACTION = 0.1

# Fixed seed so the train/eval split is reproducible across runs.
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Najdi dialect government-services data.")
    parser.add_argument(
        "--dataset-name",
        default="HeshamHaroon/saudi-dialect-conversations",
        help="Hugging Face Hub dataset id to download.",
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Directory to write train.jsonl and eval.jsonl to.",
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=EVAL_FRACTION,
        help="Fraction of filtered examples to hold out for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for the train/eval split.",
    )
    return parser.parse_args()


def load_and_filter(dataset_name: str) -> list[dict]:
    """Download the source dataset and keep only the categories we care about."""
    # HF token is optional here since the dataset is public, but we read it
    # anyway so this script behaves the same way as train_lora.py / evaluate.py
    # if the dataset (or a private fork of it) ever becomes gated.
    hf_token = os.environ.get("HF_TOKEN")

    print(f"Downloading dataset: {dataset_name}")
    raw = load_dataset(dataset_name, split="train", token=hf_token)

    print(f"Raw dataset size: {len(raw)} examples")
    print(f"Filtering to categories: {sorted(TARGET_CATEGORIES)}")

    filtered = [
        example
        for example in raw
        if example.get("category") in TARGET_CATEGORIES
    ]

    print(f"Filtered dataset size: {len(filtered)} examples")

    if not filtered:
        raise ValueError(
            "No examples matched the target categories. Check that the "
            "dataset's `category` field values match TARGET_CATEGORIES "
            "(the source dataset schema may have changed)."
        )

    return filtered


def to_prompt_response(example: dict) -> dict:
    """
    Normalize a raw dataset row into the {"prompt", "response"} schema
    expected by the training script.

    # النص العربي (الحوار باللهجة النجدية) موجود في حقول dialect_text /
    # response في مجموعة البيانات الأصلية -- هنا نحوّله لتنسيق موحّد
    # يستخدمه SFTTrainer مباشرة.
    """
    return {
        "prompt": example["dialect_text"].strip(),
        "response": example["response"].strip(),
    }


def write_jsonl(examples: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            # ensure_ascii=False so Arabic text is written as readable UTF-8
            # rather than escaped \uXXXX sequences.
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples to {path}")


def main() -> None:
    args = parse_args()

    filtered = load_and_filter(args.dataset_name)
    normalized = [to_prompt_response(ex) for ex in filtered]

    # Shuffle deterministically before splitting so train/eval aren't
    # biased by any ordering in the source dataset (e.g. grouped by category).
    rng = random.Random(args.seed)
    rng.shuffle(normalized)

    eval_size = max(1, int(len(normalized) * args.eval_fraction))
    eval_examples = normalized[:eval_size]
    train_examples = normalized[eval_size:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(train_examples, output_dir / "train.jsonl")
    write_jsonl(eval_examples, output_dir / "eval.jsonl")

    print("Done.")


if __name__ == "__main__":
    main()
