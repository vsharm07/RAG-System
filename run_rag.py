from pathlib import Path
from types import SimpleNamespace
import hashlib

import numpy as np

from embeddings import get_embedding
from ingestion.chunker import Chunker
from llm import call_llm
from main import run
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.vector_retriever import VectorRetriever
from vector_store import create_index

DATA_PATH = Path(__file__).resolve().parent / "data" / "data.txt"


def load_documents():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    docs = text.split("\n---\n")

    documents = []
    for i, doc in enumerate(docs):
        documents.append({
            "text": doc.strip(),
            "doc_id": f"doc{i}"
        })

    return documents


def build_vector_store(chunks):
    embeddings = [get_embedding(chunk["text"]) for chunk in chunks]
    index = create_index(embeddings)
    return VectorStoreWrapper(index, chunks)


def build_pipeline(vector_store, chunks):
    vector_retriever = VectorRetriever(vector_store)
    bm25_retriever = BM25Retriever(chunks)
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
    reranker = Reranker()
    cache_namespace = build_cache_namespace(chunks)

    return {
        "retriever": hybrid_retriever,
        "reranker": reranker,
        "cache_namespace": cache_namespace,
    }


def build_cache_namespace(chunks):
    digest = hashlib.sha256()

    for chunk in chunks:
        digest.update(chunk["chunk_id"].encode("utf-8"))
        digest.update(b"\n")

    return f"chunks:{len(chunks)}:{digest.hexdigest()[:12]}"

class VectorStoreWrapper:
    def __init__(self, index, chunks):
        self.index = index
        self.chunks = chunks

    def similarity_search(self, query, k=20):
        query_vector = np.array([get_embedding(query)]).astype("float32")
        distances, indices = self.index.search(query_vector, k)
        results = []
        for i in indices[0]:
            if 0 <= i < len(self.chunks):
                chunk = self.chunks[i]
                results.append(SimpleNamespace(
                    page_content=chunk["text"],
                    metadata={
                        "doc_id": chunk["doc_id"],
                        "chunk_id": chunk["chunk_id"]
                    }
                ))
        return results

class LLMWrapper:
    def invoke(self, prompt):
        return call_llm(prompt)


def main():
    documents = load_documents()
    chunks = Chunker().chunk(documents)
    vector_store = build_vector_store(chunks)
    pipeline = build_pipeline(vector_store, chunks)
    llm = LLMWrapper()

    query = "why are dogs loyal?"
    answer = run(query, pipeline, llm)
    print("answer", answer)


if __name__ == "__main__":
    main()
