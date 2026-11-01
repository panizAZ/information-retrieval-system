import argparse
import os
import random

from preprocess import TextPreprocessor, load_documents, load_queries, load_qrels
from index import InvertedIndex
from ranker import TFIDFRanker, rank_dev_queries, show_sample_results
from semantic import WordNetSemanticReranker
from evaluator import evaluate, write_rankings, write_metrics, write_summary, write_query_trace, build_query_trace
import pickle
import nltk

# for running:
#python system.py --docs HW2_data/documents.jsonl --dev-queries HW2_data/queries_dev.tsv --dev-qrels HW2_data/qrels_dev.tsv --test-queries HW2_data/queries_test.tsv --outdir results

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def text_preparation(preprocessor, documents, queries):
    doc_tokens_before = []
    doc_tokens_after = []

    for _, doc_text in documents:
        doc_tokens_before.append(preprocessor.preprocess(doc_text, False, False, False))
        doc_tokens_after.append(preprocessor.preprocess(doc_text, True, True, True))

    query_tokens_before = []
    query_tokens_after = []

    for _, query_text in queries.items():
        query_tokens_before.append(preprocessor.preprocess(query_text, False, False, False))
        query_tokens_after.append(preprocessor.preprocess(query_text, True, True, True))

    print("Document vocab size before:", preprocessor.get_vocab_size(doc_tokens_before))
    print("Document vocab size after:", preprocessor.get_vocab_size(doc_tokens_after))
    print("Query vocab size before:", preprocessor.get_vocab_size(query_tokens_before))
    print("Query vocab size after:", preprocessor.get_vocab_size(query_tokens_after))

    example_doc_id, example_doc = documents[0]
    print("\n=== Example Document ===")
    print("Doc ID:", example_doc_id)
    print("Original:", example_doc)
    print("Fully Processed:", preprocessor.preprocess(example_doc, True, True, True))

    example_query_id, example_query = list(queries.items())[1]
    print("\n=== Example Query ===")
    print("Query ID:", example_query_id)
    print("Original:", example_query)
    print("Fully Processed:", preprocessor.preprocess(example_query, True, True, True))


def inverse_index(preprocessor, documents):
    inv_index = InvertedIndex(preprocessor)

    for doc_id, text in documents:
        inv_index.add_document(doc_id, text)

    stats = inv_index.get_stats()

    print("=== INDEX STATISTICS FOR REPORT ===")
    print(f"Number of Unique Terms: {stats['unique_terms_size']}")
    print(f"Number of Postings: {stats['total_postings']}")
    print("\nTop 10 Terms by Document Frequency (Term, DF):")
    for term, df in stats["top_10_terms"]:
        print(f"  - '{term}': {df}")

    print("\n=== 3 SAMPLE POSTING LISTS (10 shown for each) ===")
    for term, postings in stats["three_posting_samples"].items():
        print(f"  - '{term}': {list(postings.items())[:10]}")

    return inv_index


def lexical_retrieval(ranker, dev_queries, qrels_raw, qrels, outdir):
    query_list = list(dev_queries.items())  # if dev_queries is a dict
    all_rankings = rank_dev_queries(ranker, query_list, 100)

    show_sample_results(query_list, all_rankings, qrels_raw, num_queries=3, top_k=10)

    write_rankings(all_rankings, os.path.join(outdir, "dev_rankings.tsv"), "tfidf", top_k=100)

    lexical_metrics = evaluate(rankings_to_doc_ids(all_rankings), qrels)
    return lexical_metrics, all_rankings

def build_random_rankings(dev_queries, document_dict, top_k=100, seed=42):
    all_doc_ids = list(document_dict.keys())
    rng = random.Random(seed)

    rankings = {}
    for qid in dev_queries.keys():
        shuffled = all_doc_ids[:]
        rng.shuffle(shuffled)
        rankings[qid] = [(doc_id, float(top_k - i)) for i, doc_id in enumerate(shuffled[:top_k])]
    return rankings

def build_overlap_rankings(dev_queries, document_dict, preprocessor, top_k=100):
    doc_token_sets = {
        doc_id: set(preprocessor.preprocess(doc_text, True, True, True))
        for doc_id, doc_text in document_dict.items()
    }

    query_token_sets = {
        qid: set(preprocessor.preprocess(qtext, True, True, True))
        for qid, qtext in dev_queries.items()
    }

    rankings = {}
    for qid, q_tokens in query_token_sets.items():
        scored = []
        for doc_id, d_tokens in doc_token_sets.items():
            overlap = len(q_tokens & d_tokens)
            scored.append((doc_id, float(overlap)))

        scored.sort(key=lambda x: x[1], reverse=True)
        rankings[qid] = scored[:top_k]

    return rankings

def prepare_qrels_for_eval(qrels):
    grouped = {}
    for (qid, doc_id), rel in qrels.items():
        if rel > 0:
            grouped.setdefault(qid, set()).add(doc_id)
    return grouped


def rankings_to_doc_ids(rankings):
    return {qid: [doc_id for doc_id, _ in docs] for qid, docs in rankings.items()}


def rerank_results_to_writer_format(reranked):
    return [(doc_id, final_score) for doc_id, final_score, lexical_score, semantic_score in reranked]


def main():
    # CLI
    parser = argparse.ArgumentParser(description="NLP Information Retrieval System")
    parser.add_argument("--docs", required=True, help="Path to documents.jsonl")
    parser.add_argument("--dev-queries", required=True, help="Path to queries_dev.tsv")
    parser.add_argument("--dev-qrels", required=True, help="Path to qrels_dev.tsv")
    parser.add_argument("--test-queries", required=True, help="Path to queries_test.tsv")
    parser.add_argument("--outdir", required=True, help="Output directory for results")

    args = parser.parse_args()

    print("Loading data...")
    preprocessor = TextPreprocessor()
    documents = load_documents(args.docs)
    dev_queries = load_queries(args.dev_queries)
    test_queries = load_queries(args.test_queries)
    qrels_raw = load_qrels(args.dev_qrels)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    qrels = prepare_qrels_for_eval(qrels_raw)
    document_dict = dict(documents)

    # --- Uploading cache ---
    cache_file = os.path.join(outdir, "precomputed_scores.pkl")
    query_precomputed = {}

    if os.path.exists(cache_file):
        print("Loading precomputed scores from cache...")
        try:
            with open(cache_file, "rb") as f:
                query_precomputed = pickle.load(f)
            print("Precomputed scores loaded successfully.")
        except Exception as e:
            print(f"Error loading cache: {e}. Recomputing scores.")
    else:
        print("No cache found. Starting heavy computation for precomputed scores...")

    text_preparation(preprocessor, documents, dev_queries)
    inv_index = inverse_index(preprocessor,documents)
    ranker = TFIDFRanker(inv_index)
    lexical_metrics, lexical_dev_rankings = lexical_retrieval(ranker, dev_queries, qrels_raw, qrels, outdir)

    random_rankings = build_random_rankings(dev_queries, document_dict, top_k=100, seed=42)
    random_metrics = evaluate(rankings_to_doc_ids(random_rankings), qrels)
    overlap_rankings = build_overlap_rankings(dev_queries, document_dict, preprocessor, top_k=100)
    overlap_metrics = evaluate(rankings_to_doc_ids(overlap_rankings), qrels)


    # Alpha sweep over semantic reranking
    alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    semantic_ranker = WordNetSemanticReranker(ranker, alpha=0.0)
    print("\n--- Starting Alpha Sweep ---")
    #  محاسبات سنگین را فقط یک‌بار انجام بده
    if not query_precomputed:  # این شرط چک می‌کند که آیا کش بارگذاری شده یا نه
        print("\n--- Starting Heavy Computation for Precomputed Scores ---")

        for qid, qtext in dev_queries.items():
            lex_results = ranker.rank(qtext, 100)
            scored_top, tail = semantic_ranker.score_candidates(qtext, document_dict, lex_results, 20)
            query_precomputed[qid] = (scored_top, tail, lex_results)
        print("--- Heavy Computation Finished ---")

        # --- شروع کد برای ذخیره کش ---
        print("Saving precomputed scores to cache...")
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(query_precomputed, f)
            print("Precomputed scores saved successfully.")
        except Exception as e:
            print(f"Error saving cache: {e}")
        # --- پایان کد برای ذخیره کش ---
    else:
        print("Using cached precomputed scores.")

    alpha_sweep = {}
    best_map = -1.0
    best_alpha = None
    best_dev_rankings = None
    best_traces = None
    best_metrics = None

    for alpha in alpha_values:
        alpha_rankings = {}
        traces = []

        for qid, qtext in dev_queries.items():
            scored_top, tail, lex_results = query_precomputed[qid]

            # ترکیب امتیازها با آلفای فعلی
            reranked = semantic_ranker.combine_scores(scored_top, tail, alpha)

            alpha_rankings[qid] = rerank_results_to_writer_format(reranked)
            traces.append(build_query_trace(qid, qtext, lex_results, reranked, 20,preprocessor, alpha))

        # evaluation
        alpha_eval_rankings = {qid: [doc_id for doc_id, score in docs] for qid, docs in alpha_rankings.items()}
        metrics = evaluate(alpha_eval_rankings, qrels)

        alpha_sweep[str(alpha)] = {
            "MAP": metrics["MAP"],
            "NDCG@10": metrics["NDCG@10"]
        }

        print(f"Alpha: {alpha:.1f} | MAP: {metrics['MAP']:.4f} | NDCG@10: {metrics['NDCG@10']:.4f}")

        if metrics["MAP"] > best_map:
            best_map = metrics["MAP"]
            best_alpha = alpha
            best_dev_rankings = alpha_rankings
            best_traces = traces
            best_metrics = metrics

    print(f"\nBest Alpha Selected: {best_alpha} with MAP: {best_map:.4f}")

    # Save best dev dev_rankings.tsv
    write_rankings(best_dev_rankings, os.path.join(outdir, "dev_rankings.tsv"), "final", top_k=100)
    # Save best query_trace.jsonl
    write_query_trace(best_traces, os.path.join(outdir, "query_trace.jsonl"))

    # Test rankings with best alpha
    semantic_ranker.alpha = best_alpha
    test_rankings = {}

    for query_id, query_text in test_queries.items():
        reranked = semantic_ranker.rerank(
            query_text,
            document_dict,
            initial_k=100,
            rerank_k=20
        )
        test_rankings[query_id] = rerank_results_to_writer_format(reranked)

    # test_rankings.tsv
    write_rankings(test_rankings, os.path.join(outdir, "test_rankings.tsv"), "final", top_k=100)

    dev_metrics = {
        "random": random_metrics,
        "overlap": overlap_metrics,
        "tfidf": lexical_metrics,
        "final": best_metrics,
    }

    write_metrics(dev_metrics, os.path.join(outdir, "dev_metrics.json"))

    summary = {
        "student_name": "Paniz Azarnia",
        "student_id": "40435097",
        "token_pattern": r"(?u)\b\w\w+\b",
        "lowercase": True,
        "stopword_removal": True,
        "stemming": True,
        "lexical_method": "tfidf_cosine",
        "semantic_method": "wordnet_rerank",
        "topk": 100,
        "random_seed": 42,
        "score_weights": {
            "lexical": best_alpha,
            "semantic": 1.0 - best_alpha
        }
    }

    write_summary(summary, os.path.join(outdir, "summary.json"))

    print(f"Done. Outputs written to {outdir}/")

if __name__ == "__main__":
    main()