import os
from datetime import datetime
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from local_examples.rag_vector_store import (
    DEFAULT_DOCS_DIR,
    DEFAULT_INDEX_PATH,
    build_index,
    format_context,
    index_stats,
    search,
)


# Current stage: application integration.
# This FastAPI app wraps the already verified local MiniCPM inference flow as
# an HTTP API so browsers, scripts, and future web demos can call the model.
MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "model-cache"
    / "OpenBMB"
    / "MiniCPM-2B-sft-fp32"
)
ADAPTER_DIR = os.environ.get("MINICPM_ADAPTER_DIR")
WEB_DEMO_PATH = Path(__file__).with_name("web_demo.html")

app = FastAPI(title="MiniCPM Local API")

tokenizer = None
model = None
sessions: dict[str, list[dict[str, str]]] = {}

WEB_DEMO_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MiniCPM WebDemo</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --line: #d8dee9;
      --accent: #2563eb;
      --accent-dark: #1d4ed8;
      --user: #e8f0ff;
      --assistant: #eef7f1;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }

    main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: 100vh;
      max-width: 960px;
      margin: 0 auto;
      padding: 24px;
      gap: 16px;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 12px;
    }

    h1 {
      margin: 0;
      font-size: 24px;
      font-weight: 650;
      letter-spacing: 0;
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
      white-space: nowrap;
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #9ca3af;
    }

    .dot.ok {
      background: #16a34a;
    }

    #messages {
      min-height: 360px;
      overflow-y: auto;
      padding: 4px 2px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .message {
      width: min(760px, 92%);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      line-height: 1.55;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    .message.user {
      align-self: flex-end;
      background: var(--user);
    }

    .message.assistant {
      align-self: flex-start;
      background: var(--assistant);
    }

    form {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }

    textarea {
      width: 100%;
      min-height: 52px;
      max-height: 160px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      font: inherit;
      color: var(--text);
      background: var(--panel);
    }

    button {
      min-width: 88px;
      border: 0;
      border-radius: 8px;
      padding: 0 18px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
    }

    button:hover {
      background: var(--accent-dark);
    }

    button:disabled {
      cursor: wait;
      opacity: 0.7;
    }

    @media (max-width: 640px) {
      main {
        padding: 16px;
      }

      header {
        align-items: flex-start;
        flex-direction: column;
      }

      form {
        grid-template-columns: 1fr;
      }

      button {
        min-height: 44px;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>MiniCPM WebDemo</h1>
      <span class="status"><span id="status-dot" class="dot"></span><span id="status-text">Checking</span></span>
    </header>
    <section id="messages" aria-live="polite"></section>
    <form id="chat-form">
      <textarea id="prompt" name="prompt" placeholder="输入问题" rows="2"></textarea>
      <button id="send" type="submit">发送</button>
    </form>
  </main>
  <script>
    const messages = document.getElementById("messages");
    const form = document.getElementById("chat-form");
    const promptInput = document.getElementById("prompt");
    const sendButton = document.getElementById("send");
    const statusDot = document.getElementById("status-dot");
    const statusText = document.getElementById("status-text");

    function addMessage(role, text) {
      const item = document.createElement("div");
      item.className = `message ${role}`;
      item.textContent = text;
      messages.appendChild(item);
      messages.scrollTop = messages.scrollHeight;
      return item;
    }

    async function updateHealth() {
      try {
        const res = await fetch("/health");
        const data = await res.json();
        statusDot.className = data.model_loaded ? "dot ok" : "dot";
        statusText.textContent = data.model_loaded ? "Ready" : "Loading";
      } catch {
        statusDot.className = "dot";
        statusText.textContent = "Offline";
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const prompt = promptInput.value.trim();
      if (!prompt) return;

      addMessage("user", prompt);
      promptInput.value = "";
      sendButton.disabled = true;
      const pending = addMessage("assistant", "...");

      try {
        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt }),
        });
        const data = await res.json();
        pending.textContent = data.response || data.error || "No response";
      } catch (error) {
        pending.textContent = `Request failed: ${error}`;
      } finally {
        sendButton.disabled = false;
        promptInput.focus();
      }
    });

    promptInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    updateHealth();
    setInterval(updateHealth, 5000);
  </script>
</body>
</html>
"""


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    session_id: str = "default"
    use_history: bool = True
    temperature: float = 0.5
    top_p: float = 0.8
    max_length: int = 1024
    repetition_penalty: float = 1.02


class ResetRequest(BaseModel):
    session_id: str = "default"


class RagRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=8)
    temperature: float = 0.3
    top_p: float = 0.8
    max_length: int = 1200
    repetition_penalty: float = 1.02


def torch_gc() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def attach_adapter_if_configured(base_model):
    if not ADAPTER_DIR:
        return base_model

    adapter_path = Path(ADAPTER_DIR)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "MINICPM_ADAPTER_DIR was set, but peft is not installed."
        ) from exc

    print(f"loading adapter: {adapter_path}")
    return PeftModel.from_pretrained(base_model, adapter_path)


def call_chat_model(prompt: str, history: list[dict[str, str]], request) -> tuple[str, list]:
    chat_owner = model
    chat_fn = getattr(chat_owner, "chat", None)
    if chat_fn is None and hasattr(chat_owner, "base_model"):
        base_model = getattr(chat_owner.base_model, "model", None)
        chat_fn = getattr(base_model, "chat", None)
    if chat_fn is None:
        raise RuntimeError("Loaded model does not expose a MiniCPM chat method.")

    return chat_fn(
        tokenizer,
        prompt,
        history=history,
        max_length=request.max_length,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )


def ensure_rag_index() -> dict[str, int | bool]:
    stats = index_stats(DEFAULT_INDEX_PATH)
    if not stats["exists"] or stats["chunks"] == 0:
        build_index(DEFAULT_DOCS_DIR, DEFAULT_INDEX_PATH)
        stats = index_stats(DEFAULT_INDEX_PATH)
    return stats


@app.get("/", response_class=HTMLResponse)
def web_demo():
    if WEB_DEMO_PATH.exists():
        return WEB_DEMO_PATH.read_text(encoding="utf-8")
    return WEB_DEMO_HTML


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
    model = attach_adapter_if_configured(model)
    model.eval()
    print("model ready.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None and tokenizer is not None,
        "cuda_available": torch.cuda.is_available(),
        "active_sessions": len(sessions),
        "rag_index": index_stats(DEFAULT_INDEX_PATH),
        "adapter_dir": ADAPTER_DIR,
    }


@app.post("/chat")
def chat(request: ChatRequest):
    if model is None or tokenizer is None:
        return {"status": 503, "response": "", "error": "model is not loaded"}

    history = sessions.setdefault(request.session_id, []) if request.use_history else []
    response, updated_history = call_chat_model(request.prompt, history, request)
    if request.use_history:
        sessions[request.session_id] = updated_history
    torch_gc()

    return {
        "status": 200,
        "session_id": request.session_id,
        "prompt": request.prompt,
        "response": response,
        "history_turns": len(sessions.get(request.session_id, [])) // 2,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.post("/reset")
def reset(request: ResetRequest):
    sessions.pop(request.session_id, None)
    torch_gc()
    return {
        "status": 200,
        "session_id": request.session_id,
        "message": "session reset",
    }


@app.get("/rag/status")
def rag_status():
    return {
        "status": 200,
        "docs_dir": str(DEFAULT_DOCS_DIR),
        "index_path": str(DEFAULT_INDEX_PATH),
        "index": index_stats(DEFAULT_INDEX_PATH),
    }


@app.post("/rag/rebuild")
def rag_rebuild():
    chunks = build_index(DEFAULT_DOCS_DIR, DEFAULT_INDEX_PATH)
    return {
        "status": 200,
        "docs_dir": str(DEFAULT_DOCS_DIR),
        "index_path": str(DEFAULT_INDEX_PATH),
        "indexed_chunks": chunks,
        "index": index_stats(DEFAULT_INDEX_PATH),
    }


@app.post("/rag/query")
def rag_query(request: RagRequest):
    if model is None or tokenizer is None:
        return {"status": 503, "response": "", "error": "model is not loaded"}

    ensure_rag_index()
    chunks = search(request.prompt, DEFAULT_INDEX_PATH, top_k=request.top_k)
    context = format_context(chunks)
    rag_prompt = (
        "请只根据下面的资料回答问题。如果资料不足，就明确说资料不足，"
        "不要编造资料中没有的信息。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{request.prompt}\n"
        "回答："
    )

    response, _ = call_chat_model(rag_prompt, [], request)
    torch_gc()

    return {
        "status": 200,
        "prompt": request.prompt,
        "response": response,
        "sources": [
            {
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "score": round(chunk.score, 4),
                "text": chunk.text,
            }
            for chunk in chunks
        ],
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
