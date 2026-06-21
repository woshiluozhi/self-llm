# Stage Log

Current stage: fine-tuned model deployment verification.

Completed:

- Created the `self-llm` Conda environment.
- Installed GPU-enabled PyTorch and model runtime dependencies.
- Downloaded `OpenBMB/MiniCPM-2B-sft-fp32` to a local model cache outside the
  Git repository.
- Verified local Transformers inference on CUDA.
- Added a command-line chat loop.
- Added a FastAPI `/chat` endpoint.
- Added a Python API client and verified end-to-end HTTP calling.
- Verified manual Swagger `/docs` calling for `POST /chat`.
- Added a browser WebDemo at `/`.
- Added scripts to start, stop, and inspect the local API runtime.
- Added continuous chat session support.
- Added LangChain integration through the local FastAPI endpoint.
- Added a multi-mode LangChain workflow app.
- Added a multi-document RAG knowledge base with local embeddings and SQLite
  vector storage.
- Added RAG API endpoints and WebDemo RAG mode with retrieved source display.
- Added LoRA, P-tuning, and guarded full fine-tuning workflows.
- Ran LoRA dry-run, P-tuning dry-run, and full fine-tuning dry-run.
- Ran one-step LoRA and one-step P-tuning smoke training.
- Added base-vs-adapter comparison.
- Verified API startup with a LoRA adapter through `-AdapterDir`.

Current local evidence:

- `/health` reports model loaded, CUDA available, and RAG index present.
- RAG CLI answers `RAG 的全称是什么？` from the local knowledge base.
- WebDemo RAG mode shows retrieved source chunks.
- LoRA smoke training saved an adapter under
  `local_examples/finetune_outputs/lora`.
- P-tuning smoke training saved an adapter under
  `local_examples/finetune_outputs/ptuning`.
- Generated adapter and SQLite files are local artifacts ignored by Git.

Remaining learning work:

- Add more realistic fine-tuning data.
- Train LoRA for more than a one-step smoke test.
- Evaluate before/after quality with a fixed question set.
- Only attempt real full fine-tuning after confirming GPU memory, time budget,
  and recovery plan.
