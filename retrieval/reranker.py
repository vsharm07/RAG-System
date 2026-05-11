from pathlib import Path
import numpy as np
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        model_path = Path(__file__).resolve().parent.parent / "models" / "reranker"
        self.model = CrossEncoder(str(model_path))

    def rerank(self, query, docs, top_k=5, generation_k=3, score_threshold=0.3):
        """
        top_k           — how many docs to return for retrieval eval
        generation_k    — how many docs to pass to LLM (precision cutoff)
        score_threshold — drop anything below this score regardless of rank
        """
        if not docs:
            return []

        pairs = [(query, doc["text"]) for doc in docs]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Log scores so you can see what's being cut
        print("\n[Reranker scores]")
        for doc, score in ranked:
            print(f"  {score:.3f} | {doc['chunk_id']} | {doc['text'][:60]}")

        # For retrieval eval — return top_k (preserves your recall metrics)
        retrieval_results = [doc for doc, _ in ranked[:top_k]]

        # For generation — apply both generation_k and score threshold
        generation_results = [
            doc for doc, score in ranked[:4]
            if score > score_threshold
        ]

        return retrieval_results, generation_results