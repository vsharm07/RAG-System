def build_prompt(query, contexts):
    context_text = "\n\n".join([c["text"] for c in contexts])

    prompt = f"""
You are a helpful assistant.

Answer ONLY from the context below.
If the answer is not present, say "I don't know".

Context:
{context_text}

Question:
{query}
"""
    return prompt