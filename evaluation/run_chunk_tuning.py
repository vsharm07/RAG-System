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


def run_config(name, min_length, max_chars, documents, dataset, reranker):
    print(f"\n[Config: {name}] min_length={min_length}, max_chars={max_chars}")

    chunker = Chunker(max_chars=max_chars, min_length=min_length)
    chunks = chunker.chunk(documents)
    print(f"  → {len(chunks)} chunks produced")

    vector_store = build_vector_store(chunks)
    vector_retriever = VectorRetriever(vector_store)
    bm25_retriever = BM25Retriever(chunks)
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

    results = evaluate(dataset, hybrid_retriever, reranker=reranker)
    return results, len(chunks)


def print_summary(all_results):
    print(f"\n{'='*75}")
    print("  CHUNK TUNING SUMMARY")
    print(f"{'='*75}")
    print(f"  {'Config':<25} {'Chunks':>7} {'R@1':>8} {'R@3':>8} {'MRR':>8} {'nDCG@5':>10}")
    print(f"  {'-'*65}")

    for name, (results, chunk_count) in all_results.items():
        print(
            f"  {name:<25} {chunk_count:>7} "
            f"{results['recall@1']:>8.4f} "
            f"{results['recall@3']:>8.4f} "
            f"{results['mrr']:>8.4f} "
            f"{results['ndcg@5']:>10.4f}"
        )


def main():
    print("[Setup] Loading documents...")
    documents = load_documents()
    dataset = load_dataset()
    reranker = Reranker()

    configs = {
        "A: current (50/1800)":   (50,  1800),
        "B: lower min (20/1800)": (20,  1800),
        "C: smaller max (20/500)": (20,  500),
    }

    all_results = {}
    for name, (min_length, max_chars) in configs.items():
        results, chunk_count = run_config(
            name, min_length, max_chars,
            documents, dataset, reranker
        )
        all_results[name] = (results, chunk_count)

    print_summary(all_results)

    # Save
    out = {
        name: {"chunk_count": cc, **res}
        for name, (res, cc) in all_results.items()
    }
    out_path = PROJECT_ROOT / "evaluation" / "chunk_tuning_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Saved] → {out_path}")


if __name__ == "__main__":
    main()