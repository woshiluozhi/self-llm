from pathlib import Path
import re
import sys

from langchain_core.prompts import PromptTemplate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_examples.langchain_minicpm_api import MiniCPMApiLLM


# Current stage: application integration.
# This is a minimal RAG example: retrieve relevant local document chunks, put
# them into a prompt, and ask the local MiniCPM API to answer from that context.
DEFAULT_DOC = Path(__file__).resolve().parent / "rag_docs" / "self_llm_stage.md"


def load_chunks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
    if not chunks:
        raise ValueError(f"No text chunks found in {path}")
    return chunks


def tokenize(text: str) -> set[str]:
    words = set(re.findall(r"[A-Za-z0-9_]+", text.lower()))
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    chinese_bigrams = {
        "".join(chinese_chars[index : index + 2])
        for index in range(max(len(chinese_chars) - 1, 0))
    }
    chinese_trigrams = {
        "".join(chinese_chars[index : index + 3])
        for index in range(max(len(chinese_chars) - 2, 0))
    }
    return words | set(chinese_chars) | chinese_bigrams | chinese_trigrams


def retrieve(question: str, chunks: list[str], limit: int = 2) -> list[str]:
    question_terms = tokenize(question)
    scored = []
    for index, chunk in enumerate(chunks):
        overlap = len(question_terms & tokenize(chunk))
        scored.append((overlap, index, chunk))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk for _, _, chunk in scored[:limit]]


def main() -> None:
    question = " ".join(sys.argv[1:]) or "应用接入阶段要完成什么？"
    chunks = load_chunks(DEFAULT_DOC)
    context = "\n\n".join(retrieve(question, chunks))

    prompt = PromptTemplate.from_template(
        "请只根据下面的资料回答问题。如果资料不足，就说资料不足。\n\n"
        "资料：\n{context}\n\n"
        "问题：{question}\n"
        "回答："
    )
    chain = prompt | MiniCPMApiLLM()
    answer = chain.invoke({"context": context, "question": question})

    print(f"doc: {DEFAULT_DOC}")
    print(f"question: {question}")
    print(f"context:\n{context}")
    print(f"answer: {answer}")


if __name__ == "__main__":
    main()
