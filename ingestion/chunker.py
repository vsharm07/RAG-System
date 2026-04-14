class Chunker:
    def chunk(self, documents):
        """
        documents = list of dicts:
        [{"text": "...", "doc_id": "doc1"}]
        """
        chunks = []

        for doc in documents:
            paragraphs = doc["text"].split("\n\n")

            for i, para in enumerate(paragraphs):
                para = para.strip()
                if len(para) < 50:
                    continue

                chunks.append({
                    "text": para,
                    "doc_id": doc["doc_id"],
                    "chunk_id": f"{doc['doc_id']}_{i}"
                })

        return chunks