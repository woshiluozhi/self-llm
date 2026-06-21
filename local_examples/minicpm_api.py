from datetime import datetime
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


# Current stage: model deployment usage.
# This FastAPI app wraps the already verified local MiniCPM inference flow as
# an HTTP API so browsers, scripts, and future web demos can call the model.
MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "model-cache"
    / "OpenBMB"
    / "MiniCPM-2B-sft-fp32"
)

app = FastAPI(title="MiniCPM Local API")

tokenizer = None
model = None


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    temperature: float = 0.5
    top_p: float = 0.8
    repetition_penalty: float = 1.02


def torch_gc() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@app.on_event("startup")
def load_model() -> None:
    global tokenizer, model

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
    print("model ready.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None and tokenizer is not None,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/chat")
def chat(request: ChatRequest):
    if model is None or tokenizer is None:
        return {"status": 503, "response": "", "error": "model is not loaded"}

    response, _ = model.chat(
        tokenizer,
        request.prompt,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )
    torch_gc()

    return {
        "status": 200,
        "prompt": request.prompt,
        "response": response,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
