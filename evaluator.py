import json
import math
import os


def precision_at_k(ranked_docs, relevant_docs, k):
    # P@k = (# relevant docs in top k) / k
    if k <= 0:
        return 0.0
    top_k = ranked_docs[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    return hits / k

def reciprocal_rank(ranked_docs, relevant_docs):
    # Find the first relevant document
    # If its rank is r -> 1/r = RR
    # if not relevant = 0
    for rank, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant_docs:
            return 1.0 / rank
    return 0.0


def average_precision(ranked_docs, relevant_docs):
    # AP = (sum of Precision@r over ranks r where doc at r is relevant) / (# relevant docs)  for one query
    # هر جا که به یک relevant doc رسید، Precision@rank را حساب می‌کند
    # میانگین این precisionها می‌شود AP
    if not relevant_docs:
        return 0.0

    hit_count = 0
    precision_sum = 0.0

    for rank, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant_docs:
            hit_count += 1
            precision_sum += hit_count / rank

    return precision_sum / len(relevant_docs)


def dcg_at_k(ranked_docs, relevant_docs, k):
    # DCG@k = rel_1 + sum_{i=2..k} rel_i / log2(i+1)  -> because gain is binary
    dcg = 0.0
    for i, doc_id in enumerate(ranked_docs[:k], start=1):
        rel = 1 if doc_id in relevant_docs else 0
        dcg += rel / math.log2(i + 1)
    return dcg

def idcg_at_k(relevant_docs, k):
    ideal_rels = [1] * min(len(relevant_docs), k)
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        idcg += rel / math.log2(i + 1)
    return idcg


def ndcg_at_k(ranked_docs, relevant_docs, k):
    idcg = idcg_at_k(relevant_docs, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_docs, relevant_docs, k) / idcg


def evaluate(all_rankings, qrels):
    ap_scores = []
    rr_scores = []
    p1_scores = []
    p5_scores = []
    p10_scores = []
    ndcg5_scores = []
    ndcg10_scores = []

    for query_id in qrels.keys():   # evaluate over all dev queries
        ranked_docs = all_rankings.get(query_id, [])
        relevant_docs = qrels.get(query_id, set())

        ap_scores.append(average_precision(ranked_docs, relevant_docs))
        rr_scores.append(reciprocal_rank(ranked_docs, relevant_docs))
        p1_scores.append(precision_at_k(ranked_docs, relevant_docs, 1))
        p5_scores.append(precision_at_k(ranked_docs, relevant_docs, 5))
        p10_scores.append(precision_at_k(ranked_docs, relevant_docs, 10))
        ndcg5_scores.append(ndcg_at_k(ranked_docs, relevant_docs, 5))
        ndcg10_scores.append(ndcg_at_k(ranked_docs, relevant_docs, 10))

    def avg(scores):
        return sum(scores) / len(scores) if scores else 0.0

    return {
        "MAP": avg(ap_scores),
        "MRR": avg(rr_scores),
        "P@1": avg(p1_scores),
        "P@5": avg(p5_scores),
        "P@10": avg(p10_scores),
        "NDCG@5": avg(ndcg5_scores),
        "NDCG@10": avg(ndcg10_scores),
    }


def write_rankings(all_rankings, out_path, method_name, top_k=100):
    """
    dev_rankings.tsv and test_rankings.tsv:
    format: query_id, doc_id, rank, score, method
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("query_id\tdoc_id\trank\tscore\tmethod\n")

        for qid, results in all_rankings.items():
            for rank, (doc_id, score) in enumerate(results[:top_k], start=1):
                f.write(f"{qid}\t{doc_id}\t{rank}\t{score:.6f}\t{method_name}\n")


def write_metrics(metrics_dict, out_path):
    """ dev_metrics.json"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4)


def write_summary(summary_data, out_path):
    """
    summary.json
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)


def write_query_trace(traces, out_path):
    """
    query_trace.jsonl
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")


def build_query_trace(query_id, query_text, lexical_results, reranked_results, no_reranked, preprocessor, alpha):
    # ساخت دیکشنری برای دسترسی سریع به رتبه لکسیکال
    lexical_rank_map = {doc_id: i + 1 for i, (doc_id, score) in enumerate(lexical_results)}

    return {
        "query_id": query_id,
        "query_text": query_text,
        "query_tokens_raw": preprocessor.preprocess(query_text, True, False, False),
        "query_tokens_final": preprocessor.preprocess(query_text, True, True, True),
        "method": "wordnet_rerank",
        "lexical_top10": [
            {"doc_id": doc_id, "score": score, "rank": i + 1}
            for i, (doc_id, score) in enumerate(lexical_results[:10])
        ],
        "semantic_step": {
            "applied": True,
            "type": "rerank",
            "details": f"top {no_reranked} lexical documents reranked using WordNet similarity"
        },
        "final_top10": [
            {
                "doc_id": doc_id,
                "lexical_score": round(lexical_score, 4),
                "semantic_score": round(semantic_score, 4),
                "combined_score": round(final_score, 4),
                "rank_before": lexical_rank_map.get(doc_id, "N/A"),
                "rank_after": i + 1,
                "reason": f"Combined score of {final_score:.2f} (Lex: {lexical_score:.2f}, Sem: {semantic_score:.2f}) using alpha {alpha}."
            }
            for i, (doc_id, final_score, lexical_score, semantic_score) in enumerate(reranked_results[:10])
        ],
        "summary": {
            "num_docs_scored_lexically": len(lexical_results),
            "num_docs_semantically_reranked": no_reranked,
            "top10_changed": [doc_id for doc_id, _ in lexical_results[:10]] != [
                doc_id for doc_id, _, _, _ in reranked_results[:10]
            ]
        }
    }
