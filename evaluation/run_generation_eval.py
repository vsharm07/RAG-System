import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import run_with_context
from run_rag import load_documents, build_vector_store, build_pipeline, LLMWrapper
from ingestion.chunker import Chunker
from evaluation.generation_metrics import evaluate_generation

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"


def main():
    # Boot the pipeline
    print("[Setup] Loading documents and building pipeline...")
    documents = load_documents()
    chunks = Chunker().chunk(documents)
    vector_store = build_vector_store(chunks)
    pipeline = build_pipeline(vector_store, chunks)
    llm = LLMWrapper()

    # Load eval queries
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    results = []
    scores = {"faithfulness": [], "answer_relevance": [], "context_precision": []}

    for item in dataset:
        query = item["query"]

        # Get answer + context
        output = run_with_context(query, pipeline, llm)

        # Evaluate generation
        eval_result = evaluate_generation(
            query=output["query"],
            context_texts=output["context_texts"],
            answer=output["answer"],
        )
        results.append(eval_result)

        # Collect scores for averaging
        for metric in scores:
            val = eval_result.get(metric)
            if val is not None:
                scores[metric].append(val)

    # Print averages
    print("\n" + "="*50)
    print("GENERATION EVAL SUMMARY")
    print("="*50)
    for metric, vals in scores.items():
        avg = sum(vals) / len(vals) if vals else 0
        print(f"Average {metric}: {avg:.4f}")

    # Save full results
    out_path = Path(__file__).resolve().parent / "generation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Full results → {out_path}")


if __name__ == "__main__":
    main()