import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from local_examples.rag_vector_store import (
    DEFAULT_DOCS_DIR,
    DEFAULT_INDEX_PATH,
    build_index,
    index_stats,
)


def main() -> None:
    total = build_index(DEFAULT_DOCS_DIR, DEFAULT_INDEX_PATH)
    stats = index_stats(DEFAULT_INDEX_PATH)
    print(f"docs_dir: {DEFAULT_DOCS_DIR}")
    print(f"index_path: {DEFAULT_INDEX_PATH}")
    print(f"indexed_chunks: {total}")
    print(f"stats: {stats}")


if __name__ == "__main__":
    main()
