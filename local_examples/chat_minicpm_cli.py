from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Current stage: model deployment usage.
# This CLI script loads the local MiniCPM model once, then supports repeated
# command-line questions without reloading the model for every prompt.
MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "model-cache"
    / "OpenBMB"
    / "MiniCPM-2B-sft-fp32"
)


def load_model():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory does not exist: {MODEL_DIR}")

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"model_dir: {MODEL_DIR}")
    print(f"device: {device_map}")
    print("loading model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    print("model ready. Type exit or quit to stop.")
    return tokenizer, model


def ask(tokenizer, model, prompt: str, history):
    response, history = model.chat(
        tokenizer,
        prompt,
        history=history,
        temperature=0.5,
        top_p=0.8,
        repetition_penalty=1.02,
    )
    return response, history


def main() -> None:
    tokenizer, model = load_model()
    history = []

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        response, _ = ask(tokenizer, model, prompt, history)
        print(f"user: {prompt}")
        print(f"assistant: {response}")
        return

    while True:
        prompt = input("\nuser> ").strip()
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break

        response, history = ask(tokenizer, model, prompt, history)
        print(f"assistant> {response}")


if __name__ == "__main__":
    main()
