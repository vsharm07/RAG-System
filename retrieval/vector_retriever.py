class VectorRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query, top_k=20):
        results = self.vector_store.similarity_search(query, k=top_k)

        docs = []
        for r in results:
            docs.append({
                "text": r.page_content,
                "metadata": r.metadata
            })

        return docs