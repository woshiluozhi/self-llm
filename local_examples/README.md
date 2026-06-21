# Local MiniCPM Examples

Current stage: application integration.

These examples document the path completed after environment setup:

1. Run local MiniCPM inference with Transformers.
2. Ask repeated questions from a command-line chat loop.
3. Expose the same local model through a FastAPI service.
4. Call that service from a small Python client.
5. Use the Swagger `/docs` page to call `/chat` manually.
6. Use a browser WebDemo backed by the same `/chat` API.
7. Connect LangChain to the local API.
8. Run a minimal RAG flow over a local Markdown document.

The model weights are intentionally kept outside this repository under
`../model-cache/OpenBMB/MiniCPM-2B-sft-fp32` because they are about 10 GB.

## Mental Model

The same model is reused through several access layers:

```text
MiniCPM weights
  -> Transformers local inference
  -> FastAPI /chat
  -> Python client / WebDemo / LangChain / RAG
```

`minicpm_api.py` owns the GPU model process. Client scripts should call the
API instead of loading another copy of the 10 GB model.

## Single Prompt

Run a single local prompt:

```bat
python local_examples\run_minicpm_chat.py "山东省最高的山是哪座山？"
```

This loads the model, asks one question, prints the answer, then exits.

## Command-Line Chat

Start command-line chat:

```bat
python local_examples\chat_minicpm_cli.py
```

This loads the model once and keeps it in memory while you ask repeated
questions. Type `exit` or `quit` to stop.

## FastAPI Service

Start the API service:

```bat
python -m uvicorn local_examples.minicpm_api:app --host 127.0.0.1 --port 8000
```

Or use the helper script:

```bat
powershell -ExecutionPolicy Bypass -File local_examples\start_minicpm_api.ps1
```

Stop the API service:

```bat
powershell -ExecutionPolicy Bypass -File local_examples\stop_minicpm_api.ps1
```

Inspect port, process, GPU memory, and logs:

```bat
powershell -ExecutionPolicy Bypass -File local_examples\inspect_minicpm_runtime.ps1
```

Useful URLs:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/
```

## Swagger `/docs`

Open `http://127.0.0.1:8000/docs`, expand `POST /chat`, click `Try it out`,
and send a JSON body like this:

```json
{
  "prompt": "请用一句话解释什么是FastAPI",
  "temperature": 0.5,
  "top_p": 0.8,
  "repetition_penalty": 1.02
}
```

The `response` field in the response body is the model answer.

## WebDemo

Open the browser chat page:

```text
http://127.0.0.1:8000/
```

The page sends browser `fetch("/chat")` requests to the same local API.

## Python API Client

Call the API:

```bat
python local_examples\call_minicpm_api.py "请用一句话解释什么是大语言模型"
```

This proves another Python program can use the model through HTTP.

## Frontend API Call

The browser page uses this pattern:

```js
await fetch("/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ prompt: "你好" }),
});
```

That is the same interface a future frontend or other application would use.

## LangChain

Run:

```bat
python local_examples\langchain_minicpm_api.py
```

This wraps the local `/chat` API as a LangChain LLM. LangChain builds the
prompt, then the wrapper sends it to the local model API.

## Minimal RAG

Run:

```bat
python local_examples\rag_minicpm_api.py "应用接入阶段要完成什么？"
```

This reads `local_examples\rag_docs\self_llm_stage.md`, retrieves relevant
chunks, places them into a prompt, and asks MiniCPM to answer from that context.
