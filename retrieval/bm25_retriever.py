from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, documents):
        """
        documents = list of dicts with "text"
        """
        self.documents = documents
        self.texts = [doc["text"] for doc in documents]
        self.tokenized = [text.lower().split() for text in self.texts]
        print("tokenized", self.tokenized)
        self.bm25 = BM25Okapi(self.tokenized)
        print("bm25 retriever initialized", self.bm25)

    def retrieve(self, query, top_k=20):
        scores = self.bm25.get_scores(query.lower().split())

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked[:top_k]]