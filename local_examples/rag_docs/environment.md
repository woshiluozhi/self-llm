# Environment Stage

环境配置阶段的核心目标是让机器具备运行开源大模型的能力。需要准备 Conda 环境、Python 版本、pip 镜像源、CUDA 驱动、PyTorch、Transformers、ModelScope 和 Hugging Face 下载工具。

判断环境是否成功的关键证据包括：`conda activate self-llm` 可以进入环境，`torch.cuda.is_available()` 返回 True，`nvidia-smi` 能看到显卡和显存占用。

如果 PyTorch 能识别 CUDA，就说明模型推理可以使用 NVIDIA GPU，而不是只靠 CPU。
