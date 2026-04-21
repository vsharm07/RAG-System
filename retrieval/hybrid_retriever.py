class HybridRetriever:
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector = vector_retriever
        self.bm25 = bm25_retriever

    def _get_doc_key(self, doc):
        if doc.get("chunk_id"):
            return doc["chunk_id"]

        metadata = doc.get("metadata") or {}
        if metadata.get("chunk_id"):
            return metadata["chunk_id"]

        return doc["text"][:200]

    def _merge_scores(self, existing_doc, new_doc):
        merged_doc = dict(existing_doc)
        merged_scores = dict(existing_doc.get("retrieval_scores", {}))
        merged_scores.update(new_doc.get("retrieval_scores", {}))
        merged_doc["retrieval_scores"] = merged_scores
        return merged_doc

    def retrieve(self, query, top_k=30):
        vector_docs = self.vector.retrieve(query, top_k=top_k)
        bm25_docs = self.bm25.retrieve(query, top_k=top_k)

        fused_scores = {}
        fused_docs = {}
        rank_constant = 60

        for rank, doc in enumerate(vector_docs, start=1):
            key = self._get_doc_key(doc)
            fused_docs[key] = self._merge_scores(fused_docs[key], doc) if key in fused_docs else doc
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rank_constant + rank)

        for rank, doc in enumerate(bm25_docs, start=1):
            key = self._get_doc_key(doc)
            fused_docs[key] = self._merge_scores(fused_docs[key], doc) if key in fused_docs else doc
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rank_constant + rank)

        ranked_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)

        results = []
        for rank, key in enumerate(ranked_keys[:top_k], start=1):
            doc = dict(fused_docs[key])
            retrieval_scores = dict(doc.get("retrieval_scores", {}))
            retrieval_scores["hybrid_rrf_score"] = fused_scores[key]
            retrieval_scores["hybrid_rank"] = rank
            doc["retrieval_scores"] = retrieval_scores
            results.append(doc)

        return results