# Stage Log

Current stage: model deployment usage.

Completed:

- Created the `self-llm` Conda environment.
- Installed GPU-enabled PyTorch and model runtime dependencies.
- Downloaded `OpenBMB/MiniCPM-2B-sft-fp32` to a local model cache outside the Git repository.
- Verified local Transformers inference on CUDA.
- Added a command-line chat loop.
- Added a FastAPI `/chat` endpoint.
- Added a Python API client and verified end-to-end HTTP calling.

Next suggested stage:

- Build a simple WebDemo chat page on top of the FastAPI service.
