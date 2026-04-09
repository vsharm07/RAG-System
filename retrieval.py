import re

import numpy as np
from embeddings import get_embedding

THRESHOLD = 1.2

STOPWORDS = {"the", "is", "and", "about", "me", "tell", "what", "who", "a", "an"}

import re

def clean_subqueries(response):
    lines = response.split("\n")

    cleaned = []

    for line in lines:
        line = line.strip()

        # remove bullets (*, -, numbers)
        line = re.sub(r"^[\*\-\d\.\)\s]+", "", line)

        # skip empty or junk lines
        if not line:
            continue

        if "query" in line.lower():
            continue

        cleaned.append(line)

    return cleaned

def generate_subqueries(query, llm_call):
    prompt = f"""
Break the query into independent sub-queries.

Rules:
- Do NOT include headings, explanations, bullets, or markdown

Query: {query}

Return only the sub-queries
"""

    response = llm_call(prompt)
    
    return clean_subqueries(response)

def keyword_search(query, chunks):
    query_words = [
        word for word in re.findall(r"\b\w+\b", query.lower())
        if word not in STOPWORDS
    ]    
    print("query words:", query_words)

    scored = []

    for chunk in chunks:
        score = sum(1 for word in query_words if word in chunk.lower())
        if score > 0:
            scored.append((chunk, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    print("keyword scored:", scored)
    return [c[0] for c in scored]

def rerank_chunks(query, chunks, llm_call):
    scored_chunks = []

    for chunk in chunks:
        prompt = f"""
        Query: {query}

        Chunk: {chunk}

        Does this chunk answer the query?
        Reply with a score from 1 to 10.
        Only output the number.
        """

        score = llm_call(prompt)

        try:
            score = int(score.strip())
        except:
            score = 0

        scored_chunks.append((chunk, score))

    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [c[0] for c in scored_chunks]

def search(query, index, chunks, k=3):
    query_vector = np.array([get_embedding(query)]).astype("float32")

    distances, indices = index.search(query_vector, k)
    print("distances:", distances)
    print("indices:", indices)
    filtered_chunks = [
        chunks[i]
        for i, dist in zip(indices[0], distances[0])
        if dist < THRESHOLD
    ]
    return filtered_chunks

