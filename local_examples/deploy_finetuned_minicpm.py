import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


# Current stage: fine-tuned model deployment.
# Load the base model, optionally attach a PEFT adapter, and compare responses
# through the same prompt format used by the fine-tuning script.
MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "model-cache"
    / "OpenBMB"
    / "MiniCPM-2B-sft-fp32"
)
DEFAULT_ADAPTER_DIR = Path(__file__).resolve().parent / "finetune_outputs" / "lora"


def generate(model, tokenizer, prompt: str, device: torch.device) -> str:
    text = f"用户：{prompt}\n助手："
    inputs = tokenizer(text, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=96,
        do_sample=False,
        repetition_penalty=1.02,
        pad_token_id=tokenizer.eos_token_id,
    )
    answer_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def load_base(dtype: torch.dtype, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy and compare MiniCPM adapter")
    parser.add_argument("prompt", nargs="*", default=["RAG", "的全称是什么？"])
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = " ".join(args.prompt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer, base_model = load_base(dtype, device)
    base_answer = generate(base_model, tokenizer, prompt, device)
    print(f"prompt: {prompt}")
    print(f"base_answer: {base_answer}")

    if not args.adapter_dir.exists():
        print(f"adapter_missing: {args.adapter_dir}")
        return
    if PeftModel is None:
        print("adapter_error: peft is not installed")
        return

    adapter_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    adapter_model.eval()
    adapter_answer = generate(adapter_model, tokenizer, prompt, device)
    print(f"adapter_dir: {args.adapter_dir}")
    print(f"adapter_answer: {adapter_answer}")


if __name__ == "__main__":
    main()
