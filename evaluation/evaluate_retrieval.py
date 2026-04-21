import math


def recall_at_k(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids) & set(relevant_ids)) / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant_set = set(relevant_ids)

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank

    return 0.0


def dcg_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)
    dcg = 0.0

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    return dcg


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0

    ideal_hits = min(k, len(relevant_ids))
    ideal_ids = list(relevant_ids)[:ideal_hits]
    ideal_dcg = dcg_at_k(ideal_ids, ideal_ids, ideal_hits)
    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(retrieved_ids, relevant_ids, k) / ideal_dcg


def get_chunk_id(doc):
    if doc.get("chunk_id"):
        return doc["chunk_id"]

    metadata = doc.get("metadata") or {}
    return metadata.get("chunk_id")


def evaluate(dataset, retriever, reranker=None, retrieve_k=20, final_k=5):
    recall_at_1_scores = []
    recall_at_3_scores = []
    recall_at_5_scores = []
    mrr_scores = []
    ndcg_at_5_scores = []

    for item in dataset:
        query = item["query"]
        relevant = item["relevant_docs"]

        docs = retriever.retrieve(query, top_k=retrieve_k)
        final_docs = reranker.rerank(query, docs, top_k=final_k) if reranker is not None else docs[:final_k]
        retrieved_ids = [get_chunk_id(doc) for doc in final_docs]

        recall_at_1 = recall_at_k(retrieved_ids[:1], relevant)
        recall_at_3 = recall_at_k(retrieved_ids[:3], relevant)
        recall_at_5 = recall_at_k(retrieved_ids[:5], relevant)
        mrr = reciprocal_rank(retrieved_ids, relevant)
        ndcg = ndcg_at_k(retrieved_ids, relevant, 5)

        print(f"\nQuery: {query}")
        print("Retrieved:", retrieved_ids)
        print("Relevant:", relevant)
        print("Recall@1:", recall_at_1)
        print("Recall@3:", recall_at_3)
        print("Recall@5:", recall_at_5)
        print("Reciprocal Rank:", mrr)
        print("nDCG@5:", ndcg)

        recall_at_1_scores.append(recall_at_1)
        recall_at_3_scores.append(recall_at_3)
        recall_at_5_scores.append(recall_at_5)
        mrr_scores.append(mrr)
        ndcg_at_5_scores.append(ndcg)

    avg_recall_at_1 = sum(recall_at_1_scores) / len(recall_at_1_scores)
    avg_recall_at_3 = sum(recall_at_3_scores) / len(recall_at_3_scores)
    avg_recall_at_5 = sum(recall_at_5_scores) / len(recall_at_5_scores)
    avg_mrr = sum(mrr_scores) / len(mrr_scores)
    avg_ndcg_at_5 = sum(ndcg_at_5_scores) / len(ndcg_at_5_scores)

    print("\nAverage Recall@1:", avg_recall_at_1)
    print("Average Recall@3:", avg_recall_at_3)
    print("Average Recall@5:", avg_recall_at_5)
    print("Average MRR:", avg_mrr)
    print("Average nDCG@5:", avg_ndcg_at_5)

    return {
        "recall@1": avg_recall_at_1,
        "recall@3": avg_recall_at_3,
        "recall@5": avg_recall_at_5,
        "mrr": avg_mrr,
        "ndcg@5": avg_ndcg_at_5,
    }