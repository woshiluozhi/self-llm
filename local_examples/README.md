# Local MiniCPM Examples

Current stage: model deployment usage.

These examples document the path completed after environment setup:

1. Run local MiniCPM inference with Transformers.
2. Ask repeated questions from a command-line chat loop.
3. Expose the same local model through a FastAPI service.
4. Call that service from a small Python client.

The model weights are intentionally kept outside this repository under
`../model-cache/OpenBMB/MiniCPM-2B-sft-fp32` because they are about 10 GB.

## Commands

Run a single local prompt:

```bat
python local_examples\run_minicpm_chat.py "山东省最高的山是哪座山？"
```

Start command-line chat:

```bat
python local_examples\chat_minicpm_cli.py
```

Start the API service:

```bat
python -m uvicorn local_examples.minicpm_api:app --host 127.0.0.1 --port 8000
```

Call the API:

```bat
python local_examples\call_minicpm_api.py "请用一句话解释什么是大语言模型"
```
