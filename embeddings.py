from sentence_transformers import SentenceTransformer

model = SentenceTransformer("models/all-MiniLM-L6-v2")

def get_embedding(text):
    return model.encode(text)