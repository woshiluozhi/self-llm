# Self-LLM Local Learning Notes

环境配置阶段负责准备 Python、Conda、CUDA、PyTorch、Transformers、模型下载工具等基础依赖。这个阶段的目标是让机器具备运行开源大模型的能力。

模型部署使用阶段负责下载模型权重，加载 tokenizer 和 model，完成本地推理、命令行聊天、FastAPI 接口和 WebDemo。这个阶段的目标是把模型跑起来，并能被程序调用。

应用接入阶段负责把模型接入实际工具，包括 Python 客户端、网页前端、LangChain 链路和 RAG。这个阶段的目标是让模型基于外部输入、文档或业务流程完成任务。

微调阶段负责准备训练数据，运行 LoRA、P-tuning 或全量微调，并对比微调前后的效果。这个阶段的目标是让模型更适合特定领域或特定风格。
