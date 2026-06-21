# Model Deployment Stage

模型部署使用阶段负责把模型真正跑起来。这个阶段通常包括下载模型权重、加载 tokenizer、加载 model、执行本地推理、构建命令行聊天、提供 FastAPI 接口和浏览器 WebDemo。

在当前本地流程中，`minicpm_api.py` 是服务端，它加载 MiniCPM 模型并提供 `/chat` 接口。`call_minicpm_api.py` 是 Python 客户端，它通过 HTTP 请求调用本地模型服务。

如果浏览器可以打开 `http://127.0.0.1:8000/` 并正常得到回答，说明模型已经从“能运行”变成了“能被人使用”。
