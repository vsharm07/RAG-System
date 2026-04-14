import json
from evaluation.evaluate_retrieval import evaluate

# import your existing pipeline pieces
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_retriever import VectorRetriever
from retrieval.bm25_retriever import BM25Retriever


def load_dataset():
    with open("evaluation/dataset.json", "r") as f:
        return json.load(f)


def main(vector_store, chunks):
    vector = VectorRetriever(vector_store)
    bm25 = BM25Retriever(chunks)

    hybrid = HybridRetriever(vector, bm25)

    dataset = load_dataset()

    evaluate(dataset, hybrid)


if __name__ == "__main__":
    # plug your existing objects here
    vector_store = None  # replace
    chunks = []          # replace

    main(vector_store, chunks)