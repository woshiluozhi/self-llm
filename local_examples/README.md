# Local MiniCPM Examples

Current stage: fine-tuned model deployment verification.

This folder records the local learning path for MiniCPM after environment
setup. The goal is not only to run one prompt, but to understand how a local
model becomes an application and how fine-tuned weights are loaded back into
that application.

## Progress Map

```text
Environment
  -> local Transformers inference
  -> command-line chat
  -> FastAPI service
  -> Swagger / Python client / WebDemo
  -> LangChain workflow
  -> multi-document RAG vector store
  -> LoRA / P-tuning / full fine-tuning workflow
  -> adapter comparison and API deployment
```

The model weights are intentionally kept outside this repository:

```text
../model-cache/OpenBMB/MiniCPM-2B-sft-fp32
```

The generated RAG SQLite index and fine-tuned adapter outputs are local runtime
artifacts and are ignored by Git.

## File Roles

`run_minicpm_chat.py` loads MiniCPM directly with Transformers and asks one
question.

`chat_minicpm_cli.py` keeps the model in memory and lets you chat in a terminal.

`minicpm_api.py` is the FastAPI server. It loads the model once, provides
`/chat`, `/rag/query`, `/rag/status`, `/rag/rebuild`, `/health`, and serves the
browser WebDemo at `/`.

`web_demo.html` is the browser UI. Chat mode calls `/chat`; RAG mode calls
`/rag/query` and shows retrieved source chunks.

`call_minicpm_api.py` is a small Python HTTP client for `/chat`.

`langchain_minicpm_api.py` wraps the local API as a LangChain LLM.

`langchain_workflow_app.py` demonstrates several LangChain workflows: direct
answering, learning-stage advice, summarization, and RAG.

`rag_vector_store.py` builds a small local vector store using deterministic
hash embeddings and SQLite.

`rag_build_index.py` indexes the Markdown documents under `rag_docs`.

`rag_minicpm_api.py` runs a command-line RAG query through LangChain and the
local MiniCPM API.

`finetune_minicpm.py` provides LoRA, P-tuning, and guarded full fine-tuning
flows.

`deploy_finetuned_minicpm.py` compares the base model and a saved PEFT adapter.

`start_minicpm_api.ps1`, `stop_minicpm_api.ps1`, and
`inspect_minicpm_runtime.ps1` manage the local API process.

## Start The API

```bat
powershell -ExecutionPolicy Bypass -File local_examples\start_minicpm_api.ps1
```

Useful URLs:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
http://127.0.0.1:8000/rag/status
```

Stop the API:

```bat
powershell -ExecutionPolicy Bypass -File local_examples\stop_minicpm_api.ps1
```

Inspect process, port, GPU memory, and logs:

```bat
powershell -ExecutionPolicy Bypass -File local_examples\inspect_minicpm_runtime.ps1
```

## Local Inference

Run one prompt:

```bat
python local_examples\run_minicpm_chat.py "山东省最高的山是哪座山？"
```

Start terminal chat:

```bat
python local_examples\chat_minicpm_cli.py
```

## API Calling

Python client:

```bat
python local_examples\call_minicpm_api.py "请用一句话解释什么是大语言模型"
```

Swagger page:

```text
http://127.0.0.1:8000/docs
```

Example `/chat` JSON:

```json
{
  "prompt": "请用一句话解释什么是 FastAPI",
  "session_id": "default",
  "use_history": true,
  "temperature": 0.5,
  "top_p": 0.8,
  "max_length": 1024
}
```

`use_history=true` means continuous chat memory is enabled. LangChain and RAG
use `use_history=false` so long task prompts do not pollute the chat session.

## WebDemo

Open:

```text
http://127.0.0.1:8000/
```

Chat mode is normal continuous conversation.

RAG knowledge base mode first retrieves local documents, then asks MiniCPM to
answer from the retrieved context. The page shows the source chunks below the
answer.

## LangChain

Minimal wrapper check:

```bat
python local_examples\langchain_minicpm_api.py
```

Workflow examples:

```bat
python local_examples\langchain_workflow_app.py --mode stage "我完成环境配置后下一步应该做什么？"
python local_examples\langchain_workflow_app.py --mode rag "RAG 的全称是什么？"
python local_examples\langchain_workflow_app.py --mode summary "环境、部署、RAG、微调是 self-llm 的主要学习路径。"
```

LangChain is the orchestration layer. It builds prompts, routes workflow modes,
retrieves context for RAG, and calls the local MiniCPM API.

## RAG Knowledge Base

Build or rebuild the vector index:

```bat
python local_examples\rag_build_index.py
```

Run command-line RAG:

```bat
python local_examples\rag_minicpm_api.py "RAG 的全称是什么？"
```

Call RAG through the API:

```text
POST http://127.0.0.1:8000/rag/query
```

Example body:

```json
{
  "prompt": "微调阶段有哪些方法？",
  "top_k": 4,
  "temperature": 0.3,
  "max_length": 1200
}
```

The current local knowledge base has documents for environment setup,
deployment, application integration, and fine-tuning.

## Fine-Tuning

Install requirements:

```bat
pip install -r local_examples\requirements.txt
```

Dry-run LoRA, P-tuning, and full fine-tuning:

```bat
python local_examples\finetune_minicpm.py --method lora --dry-run
python local_examples\finetune_minicpm.py --method ptuning --dry-run
python local_examples\finetune_minicpm.py --method full --dry-run
```

Run a tiny LoRA smoke train:

```bat
python local_examples\finetune_minicpm.py --method lora --max-steps 1 --max-length 160
```

Run a tiny P-tuning smoke train:

```bat
python local_examples\finetune_minicpm.py --method ptuning --max-steps 1 --max-length 160
```

Full fine-tuning updates all model parameters, so the script blocks real full
training unless you pass `--allow-full-train`. Keep this guarded unless you
understand the GPU memory and training time cost.

## Fine-Tuned Deployment

Compare base model and a LoRA adapter:

```bat
python local_examples\deploy_finetuned_minicpm.py "RAG 的全称是什么？" --adapter-dir local_examples\finetune_outputs\lora
```

Start the API with a saved adapter:

```bat
powershell -ExecutionPolicy Bypass -File local_examples\start_minicpm_api.ps1 -AdapterDir local_examples\finetune_outputs\lora
```

Check `/health`; `adapter_dir` should show the loaded adapter path. For the
learning smoke test, the adapter is only trained for one step, so it proves the
engineering path rather than model quality.
