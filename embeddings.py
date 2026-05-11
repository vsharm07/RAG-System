import ollama
import numpy as np

EMBED_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768  # explicit — catches mismatches early

def get_embedding(text: str) -> np.ndarray:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return np.array(response["embedding"], dtype=np.float32)

def get_embeddings_batch(texts: list[str]) -> np.ndarray:
    """Embed multiple chunks — much faster than looping get_embedding()."""
    embeddings = [
        ollama.embeddings(model=EMBED_MODEL, prompt=t)["embedding"]
        for t in texts
    ]
    return np.array(embeddings, dtype=np.float32)