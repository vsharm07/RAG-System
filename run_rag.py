from types import SimpleNamespace
from main import run
from embeddings import get_embedding
from vector_store import create_index
import numpy as np
import faiss
from llm import call_llm

documents = [
    {"doc_id": "doc1", "text": open("data/data.txt", "r", encoding="utf-8").read()}
]

chunks = []
for doc in documents:
    for i, para in enumerate(doc["text"].split("\n\n")):
        para = para.strip()
        if len(para) >= 50:
            print("para", para)
            chunks.append({
                "text": para,
                "doc_id": doc["doc_id"],
                "chunk_id": f"{doc['doc_id']}_{i}"
            })

embeddings = [get_embedding(c["text"]) for c in chunks]
index = create_index(embeddings)

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

vector_store = VectorStoreWrapper(index, chunks)
llm = LLMWrapper()

query = "why are dogs loyal?"
answer = run(query, vector_store, documents, llm)
print("answer",answer)
