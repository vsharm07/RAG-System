import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_retrieval import evaluate
from ingestion.chunker import Chunker
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.vector_retriever import VectorRetriever
from run_rag import build_vector_store, load_documents


def load_dataset():
    with (PROJECT_ROOT / "evaluation" / "dataset.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    documents = load_documents()
    chunker = Chunker()
    chunks = chunker.chunk(documents)
    vector_store = build_vector_store(chunks)
    vector = VectorRetriever(vector_store)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(vector, bm25)
    reranker = Reranker()
    dataset = load_dataset()
    evaluate(dataset, hybrid, reranker=reranker)


if __name__ == "__main__":
    main()