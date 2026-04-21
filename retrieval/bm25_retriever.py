from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.texts = [doc["text"] for doc in documents]
        self.tokenized = [text.lower().split() for text in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)

    def retrieve(self, query, top_k=20):
        scores = self.bm25.get_scores(query.lower().split())

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        docs = []
        for rank, (doc, score) in enumerate(ranked[:top_k], start=1):
            docs.append({
                "text": doc["text"],
                "metadata": {
                    "doc_id": doc.get("doc_id"),
                    "chunk_id": doc.get("chunk_id"),
                },
                "chunk_id": doc.get("chunk_id"),
                "retrieval_scores": {
                    "bm25_score": float(score),
                    "bm25_rank": rank,
                },
            })

        return docs