from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL_PATH = Path(__file__).resolve().parent / "models" / "embedder" / "all-MiniLM-L6-v2"
model = SentenceTransformer(str(MODEL_PATH), local_files_only=True)

def get_embedding(text):
    return model.encode(text)
