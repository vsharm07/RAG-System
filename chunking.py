def chunk_text(text):
    return  [p.strip() for p in text.split("\n\n") if p.strip()]