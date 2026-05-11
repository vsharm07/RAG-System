import hashlib

class Chunker:
    def __init__(self, max_chars=1800, overlap_sentences=1, min_length=50):
        self.max_chars = max_chars        # ~450 tokens @ 4 chars/token
        self.overlap_sentences = overlap_sentences
        self.min_length = min_length

    def _split_sentences(self, text):
        # Naive sentence splitter — swap for spaCy/nltk if needed
        import re
        return re.split(r'(?<=[.!?])\s+', text.strip())

    def chunk(self, documents):
        chunks = []
        skipped = 0

        for doc in documents:
            paragraphs = doc["text"].split("\n\n")

            for para in paragraphs:
                para = para.strip()
                if len(para) < self.min_length:
                    skipped += 1
                    continue

                if len(para) <= self.max_chars:
                    chunks.append(self._make_chunk(para, doc["doc_id"]))
                else:
                    # Split long paragraph into overlapping sentence windows
                    sentences = self._split_sentences(para)
                    window = []
                    window_len = 0

                    for sent in sentences:
                        if window_len + len(sent) > self.max_chars and window:
                            chunks.append(self._make_chunk(" ".join(window), doc["doc_id"]))
                            window = window[-self.overlap_sentences:]  # keep overlap
                            window_len = sum(len(s) for s in window)
                        window.append(sent)
                        window_len += len(sent)

                    if window:
                        chunks.append(self._make_chunk(" ".join(window), doc["doc_id"]))

        print(f"[Chunker] {len(chunks)} chunks created, {skipped} short paragraphs skipped")
        return chunks

    def _make_chunk(self, text, doc_id):
        chunk_id = f"{doc_id}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
        return {"text": text, "doc_id": doc_id, "chunk_id": chunk_id}