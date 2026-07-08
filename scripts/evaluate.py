"""
evaluate.py

Loads the frozen base ALLaM-7B-Instruct-preview model and the fine-tuned
LoRA adapter from this project, runs the same 3 Najdi government-services
questions through both, and prints a side-by-side comparison so you can
visually inspect what the fine-tune changed.

This is a qualitative check, not a benchmark -- see docs/results.md for
the quantitative training/validation loss numbers and a written analysis
of these same 3 outputs.

Run:
    export HF_TOKEN=hf_xxx
    python scripts/evaluate.py --adapter-dir ./adapter
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL_ID = "humain-ai/ALLaM-7B-Instruct-preview"

# Three held-out Najdi-dialect government-services questions used
# throughout this project as the fixed qualitative eval set.
# أسئلة تجريبية باللهجة النجدية تخص الخدمات الحكومية -- نفس الأسئلة
# مستخدمة في docs/results.md لتوثيق النتائج.
EVAL_QUESTIONS = [
    "وش الأوراق اللي أحتاجها عشان أجدد الإقامة؟",
    "كيف أقدر أحجز موعد في الأحوال المدنية؟",
    "ودي أستفسر عن رسوم تجديد رخصة القيادة، كم تكلفتها؟",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base vs. fine-tuned ALLaM-7B on Najdi prompts.")
    parser.add_argument("--adapter-dir", default="./adapter", help="Path to the trained LoRA adapter.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    return parser.parse_args()


def load_base_model(hf_token: str):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        token=hf_token,
    )
    return model, tokenizer


def generate(model, tokenizer, question: str, max_new_tokens: int) -> str:
    prompt = f"### السؤال:\n{question}\n\n### الرد:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Strip the prompt back off so we only print the generated response.
    response = full_text[len(prompt):].strip()
    return response


def main() -> None:
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. You need a Hugging Face token with access to "
            f"{BASE_MODEL_ID} (accept its license on the model page first)."
        )

    print(f"Loading base model: {BASE_MODEL_ID}")
    base_model, tokenizer = load_base_model(hf_token)

    print(f"Attaching fine-tuned LoRA adapter from: {args.adapter_dir}")
    finetuned_model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    # Base and fine-tuned generation share the same underlying weights in
    # memory except for the adapter delta, so we generate with the adapter
    # disabled for the "base" pass and enabled for the "fine-tuned" pass
    # rather than loading the 7B model twice.
    results = []
    for question in EVAL_QUESTIONS:
        with finetuned_model.disable_adapter():
            base_response = generate(finetuned_model, tokenizer, question, args.max_new_tokens)
        finetuned_response = generate(finetuned_model, tokenizer, question, args.max_new_tokens)
        results.append((question, base_response, finetuned_response))

    print("\n" + "=" * 80)
    print("BASE vs. FINE-TUNED -- Najdi Government-Services Evaluation")
    print("=" * 80)
    for i, (question, base_response, finetuned_response) in enumerate(results, start=1):
        print(f"\n[Question {i}] {question}")
        print("-" * 80)
        print(f"BASE ALLaM-7B:\n{base_response}")
        print("-" * 80)
        print(f"FINE-TUNED (Najdi LoRA):\n{finetuned_response}")
        print("=" * 80)


if __name__ == "__main__":
    main()
