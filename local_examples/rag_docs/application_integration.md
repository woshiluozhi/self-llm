# Application Integration Stage

应用接入阶段负责把模型接入实际工具，包括 Python 客户端、网页前端、LangChain 链路和 RAG 知识库。这个阶段的目标是让模型基于外部输入、文档或业务流程完成任务。

LangChain 的作用是把多个步骤串成流程，例如整理问题、拼接提示词、调用模型、处理输出。LangChain 不是模型本身，而是流程编排工具。

RAG 的全称是 Retrieval-Augmented Generation，中文通常叫检索增强生成。RAG 的流程是先从文档或知识库中检索相关资料，再把资料和问题一起交给模型生成答案。
