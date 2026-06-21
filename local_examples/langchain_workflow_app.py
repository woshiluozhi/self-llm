import argparse
import sys
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_examples.langchain_minicpm_api import MiniCPMApiLLM
from local_examples.rag_vector_store import (
    DEFAULT_DOCS_DIR,
    DEFAULT_INDEX_PATH,
    build_index,
    format_context,
    index_stats,
    search,
)


# Current stage: application integration.
# This script shows LangChain as workflow orchestration: prompts, routing,
# retrieval, and the local MiniCPM API are composed into runnable chains.


def ensure_index() -> None:
    stats = index_stats(DEFAULT_INDEX_PATH)
    if not stats["exists"] or stats["chunks"] == 0:
        build_index(DEFAULT_DOCS_DIR, DEFAULT_INDEX_PATH)


def stage_hint(question: str) -> str:
    text = question.lower()
    if any(word in text for word in ["cuda", "conda", "pytorch", "环境"]):
        return "环境配置"
    if any(word in text for word in ["api", "webdemo", "部署", "推理", "fastapi"]):
        return "模型部署使用"
    if any(word in text for word in ["lora", "p-tuning", "微调", "训练"]):
        return "微调"
    if any(word in text for word in ["rag", "langchain", "知识库", "检索"]):
        return "应用接入"
    return "综合学习"


def build_direct_chain(llm: MiniCPMApiLLM):
    prompt = PromptTemplate.from_template(
        "请用清楚、简短的中文回答这个问题：\n{question}"
    )
    return prompt | llm


def build_stage_chain(llm: MiniCPMApiLLM):
    prepare = RunnableLambda(
        lambda question: {
            "question": question,
            "stage": stage_hint(question),
        }
    )
    prompt = PromptTemplate.from_template(
        "你是 self-llm 项目的学习助手。\n"
        "判断到的问题阶段是：{stage}\n"
        "用户问题：{question}\n\n"
        "请按“现在在学什么、为什么学、下一步做什么”三点回答。"
    )
    return prepare | prompt | llm


def build_summary_chain(llm: MiniCPMApiLLM):
    prompt = PromptTemplate.from_template(
        "请把下面内容总结成 3 条学习要点，每条不超过 30 字：\n{text}"
    )
    return prompt | llm


def build_rag_chain(llm: MiniCPMApiLLM, top_k: int):
    def retrieve(question: str) -> dict[str, str]:
        ensure_index()
        chunks = search(question, DEFAULT_INDEX_PATH, top_k=top_k)
        return {
            "question": question,
            "context": format_context(chunks),
            "sources": ", ".join(
                f"{chunk.source}#{chunk.chunk_index}" for chunk in chunks
            ),
        }

    prompt = PromptTemplate.from_template(
        "请只根据资料回答问题。如果资料不足，就说资料不足。\n\n"
        "资料来源：{sources}\n"
        "资料内容：\n{context}\n\n"
        "问题：{question}\n"
        "回答："
    )
    return RunnableLambda(retrieve) | prompt | llm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangChain workflow app for MiniCPM")
    parser.add_argument(
        "question",
        nargs="*",
        help="Question or text. If omitted, a default learning question is used.",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "stage", "summary", "rag"],
        default="stage",
        help="Workflow mode to run.",
    )
    parser.add_argument("--top-k", type=int, default=4, help="RAG chunks to retrieve.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = " ".join(args.question).strip() or "我完成环境配置后，下一步应该做什么？"
    llm = MiniCPMApiLLM(max_length=1200)

    chains = {
        "direct": build_direct_chain(llm),
        "stage": build_stage_chain(llm),
        "summary": build_summary_chain(llm),
        "rag": build_rag_chain(llm, top_k=args.top_k),
    }
    chain = chains[args.mode]
    payload = {"text": text} if args.mode == "summary" else text
    answer = chain.invoke(payload)

    print(f"mode: {args.mode}")
    print(f"input: {text}")
    print(f"answer: {answer}")


if __name__ == "__main__":
    main()
