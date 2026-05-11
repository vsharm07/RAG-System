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


def print_results(name, results):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Recall@1:  {results['recall@1']:.4f}")
    print(f"  Recall@3:  {results['recall@3']:.4f}")
    print(f"  Recall@5:  {results['recall@5']:.4f}")
    print(f"  MRR:       {results['mrr']:.4f}")
    print(f"  nDCG@5:    {results['ndcg@5']:.4f}")


def main():
    print("[Setup] Loading documents and building retrievers...")
    documents = load_documents()
    chunks = Chunker().chunk(documents)
    vector_store = build_vector_store(chunks)

    vector_retriever = VectorRetriever(vector_store)
    bm25_retriever = BM25Retriever(chunks)
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
    reranker = Reranker()
    dataset = load_dataset()

    all_results = {}

    # Run 1 — BM25 only
    print("\n[Run 1] BM25 only...")
    results_bm25 = evaluate(dataset, bm25_retriever, reranker=reranker)
    all_results["BM25 Only"] = results_bm25
    print_results("BM25 Only", results_bm25)

    # Run 2 — Vector only
    print("\n[Run 2] Vector only...")
    results_vector = evaluate(dataset, vector_retriever, reranker=reranker)
    all_results["Vector Only"] = results_vector
    print_results("Vector Only", results_vector)

    # Run 3 — Hybrid (BM25 + Vector + RRF)
    print("\n[Run 3] Hybrid (BM25 + Vector + RRF)...")
    results_hybrid = evaluate(dataset, hybrid_retriever, reranker=reranker)
    all_results["Hybrid"] = results_hybrid
    print_results("Hybrid", results_hybrid)

    # Comparison table
    print(f"\n{'='*70}")
    print("  ABLATION STUDY SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Metric':<15} {'BM25':>10} {'Vector':>10} {'Hybrid':>10} {'Winner':>10}")
    print(f"  {'-'*55}")

    metrics = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"]
    labels  = ["Recall@1", "Recall@3", "Recall@5", "MRR", "nDCG@5"]

    for metric, label in zip(metrics, labels):
        bm25_score   = all_results["BM25 Only"][metric]
        vector_score = all_results["Vector Only"][metric]
        hybrid_score = all_results["Hybrid"][metric]

        scores = {
            "BM25":   bm25_score,
            "Vector": vector_score,
            "Hybrid": hybrid_score,
        }
        winner = max(scores, key=scores.get)

        print(f"  {label:<15} {bm25_score:>10.4f} {vector_score:>10.4f} {hybrid_score:>10.4f} {winner:>10}")

    # Save results
    out_path = PROJECT_ROOT / "evaluation" / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] Full results → {out_path}")


if __name__ == "__main__":
    main()