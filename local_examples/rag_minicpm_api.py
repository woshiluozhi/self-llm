import sys
from pathlib import Path

from langchain_core.prompts import PromptTemplate

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
# This RAG example retrieves chunks from a SQLite vector store, puts them into
# a prompt, and asks the local MiniCPM API to answer from that context.


def ensure_index() -> None:
    stats = index_stats(DEFAULT_INDEX_PATH)
    if not stats["exists"] or stats["chunks"] == 0:
        build_index(DEFAULT_DOCS_DIR, DEFAULT_INDEX_PATH)


def main() -> None:
    question = " ".join(sys.argv[1:]) or "应用接入阶段要完成什么？"
    ensure_index()
    chunks = search(question, DEFAULT_INDEX_PATH, top_k=4)
    context = format_context(chunks)

    prompt = PromptTemplate.from_template(
        "请只根据下面的资料回答问题。如果资料不足，就说资料不足。\n\n"
        "资料：\n{context}\n\n"
        "问题：{question}\n"
        "回答："
    )
    chain = prompt | MiniCPMApiLLM()
    answer = chain.invoke({"context": context, "question": question})

    print(f"docs_dir: {DEFAULT_DOCS_DIR}")
    print(f"index_path: {DEFAULT_INDEX_PATH}")
    print(f"question: {question}")
    print(f"context:\n{context}")
    print(f"answer: {answer}")


if __name__ == "__main__":
    main()
