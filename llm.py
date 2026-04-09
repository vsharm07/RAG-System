import requests

def call_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]

def generate_answer(query, context_chunks):
    if not context_chunks:
        return "I don't know"

    context = "\n".join(context_chunks)

    prompt = f"""
Answer ONLY using the exact sentences from the context.

Do NOT add any extra information.
Do NOT elaborate.
Do NOT include anything not present in the context.

If multiple topics exist:
- answer each separately

Context:
{context}

Question:
{query}
"""

    return call_llm(prompt).strip()