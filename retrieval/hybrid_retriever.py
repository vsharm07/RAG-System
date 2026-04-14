class HybridRetriever:
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector = vector_retriever
        self.bm25 = bm25_retriever

    def retrieve(self, query, top_k=30):
        vector_docs = self.vector.retrieve(query, top_k=top_k)
        bm25_docs = self.bm25.retrieve(query, top_k=top_k)

        combined = vector_docs + bm25_docs
        print("combined", combined)
        # Deduplicate using text hash
        seen = set()
        unique_docs = []

        for doc in combined:
            print("doc", doc)
            key = doc["text"][:200]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        print("unique_docs", unique_docs)
        return unique_docs