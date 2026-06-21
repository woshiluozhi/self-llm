from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Current stage: model deployment usage.
# This minimal script verifies that the downloaded MiniCPM model can run
# local inference through Transformers on this machine.
MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "model-cache"
    / "OpenBMB"
    / "MiniCPM-2B-sft-fp32"
)


def main() -> None:
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory does not exist: {MODEL_DIR}")

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"model_dir: {MODEL_DIR}")
    print(f"device: {device_map}")

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

    prompt = " ".join(sys.argv[1:]) or "山东省最高的山是哪座山？"
    response, _ = model.chat(
        tokenizer,
        prompt,
        temperature=0.5,
        top_p=0.8,
        repetition_penalty=1.02,
    )

    print(f"prompt: {prompt}")
    print(f"response: {response}")


if __name__ == "__main__":
    main()
