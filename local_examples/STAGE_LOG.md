# Stage Log

Current stage: application integration.

Completed:

- Created the `self-llm` Conda environment.
- Installed GPU-enabled PyTorch and model runtime dependencies.
- Downloaded `OpenBMB/MiniCPM-2B-sft-fp32` to a local model cache outside the Git repository.
- Verified local Transformers inference on CUDA.
- Added a command-line chat loop.
- Added a FastAPI `/chat` endpoint.
- Added a Python API client and verified end-to-end HTTP calling.
- Verified manual Swagger `/docs` calling for `POST /chat`.
- Added a browser WebDemo at `/`.
- Added scripts to start, stop, and inspect the local API runtime.
- Added LangChain integration through the local FastAPI endpoint.
- Added a minimal local-document RAG example.

Next suggested stage:

- Move toward task-specific application work, then start microtuning only after the deployment and application flow is comfortable.
