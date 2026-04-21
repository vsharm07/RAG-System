from pathlib import Path

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        model_path = Path(__file__).resolve().parent.parent / "models" / "reranker"
        self.model = CrossEncoder(str(model_path))

    def rerank(self, query, docs, top_k=5):
        if not docs:
            return []

        if self.model is None:
            return docs[:top_k]

        pairs = [(query, doc["text"]) for doc in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked[:top_k]]
