import json

def recall_at_k(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids) & set(relevant_ids)) / len(relevant_ids)


def evaluate(dataset, retriever):
    scores = []

    for item in dataset:
        query = item["query"]
        relevant = item["relevant_docs"]

        docs = retriever.retrieve(query)
        retrieved_ids = [d.get("chunk_id") for d in docs[:5]]

        score = recall_at_k(retrieved_ids, relevant)

        print(f"\nQuery: {query}")
        print("Retrieved:", retrieved_ids)
        print("Relevant:", relevant)
        print("Recall@5:", score)

        scores.append(score)

    avg_score = sum(scores) / len(scores)
    print("\nAverage Recall@5:", avg_score)

    return avg_score