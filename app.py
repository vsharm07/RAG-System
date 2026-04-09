from chunking import chunk_text
from embeddings import get_embedding
from vector_store import create_index
from retrieval import generate_subqueries, keyword_search, search, rerank_chunks
from llm import generate_answer, call_llm

with open("data/data.txt", "r") as f:
    text = f.read()

chunks = chunk_text(text)

print("chunks:", chunks)

embeddings = [get_embedding(chunk) for chunk in chunks]
# print("embeddings:", embeddings)

index = create_index(embeddings)
print("index created", index)

query = "Tell me about Paris and Python"
print("query:", query)

subqueries = generate_subqueries(query, call_llm)
print("subqueries:", subqueries)

all_results = []

for subq in subqueries:
    vector_results = search(subq, index, chunks)
    print("vector results:", vector_results)

    keyword_results = keyword_search(subq, chunks)
    print("keyword results:", keyword_results)

    combined = list(dict.fromkeys(vector_results + keyword_results))
    print("combined:", combined)

    all_results.extend(combined)
    print("all results so far:", all_results)

    all_results = list(dict.fromkeys(all_results))
    print("all results deduped:", all_results)

reranked = rerank_chunks(query, all_results, call_llm)
print("reranked:", reranked)

top_chunks = reranked[:2]

answer = generate_answer(query, top_chunks)

print("answer:", answer)