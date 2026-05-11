# vector_store.py
import faiss
import numpy as np

def create_index(embeddings):
    vectors = np.array(embeddings).astype("float32")
    dimension = vectors.shape[1]

    # Normalize for cosine similarity
    faiss.normalize_L2(vectors)

    # IndexFlatIP = inner product = cosine on normalized vectors
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    return index