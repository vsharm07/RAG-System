class VectorRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query, top_k=20):
        results = self.vector_store.similarity_search(query, k=top_k)

        docs = []
        for rank, r in enumerate(results, start=1):
            docs.append({
                "text": r.page_content,
                "metadata": r.metadata,
                "chunk_id": r.metadata.get("chunk_id"),
                "retrieval_scores": {
                    "vector_rank": rank,
                },
            })

        return docs