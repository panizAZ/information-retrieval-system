import math
from collections import Counter


class TFIDFRanker:
    def __init__(self, inverted_index):
        self.inverted_index =  inverted_index
        self.N = inverted_index.num_docs
        self.idf = {}
        self.doc_vectors = {}
        self.doc_norms = {}
        # To be sure this two ran already
        self.compute_idf()
        self.vectorize_docs()


    def compute_idf(self):
        # index is the inverted_index's inverted index
        for term in self.inverted_index.index:
            df = self.inverted_index.get_doc_frequency(term)
            self.idf[term] = math.log((self.N+1) / (df+1))+1


    def vectorize_docs(self):
        for term, postings in self.inverted_index.index.items():
            idf = self.idf[term]
            for doc_id, tf in postings.items():
                if doc_id not in self.doc_vectors:
                    self.doc_vectors[doc_id] = {}
                # compute tf_idf for each term in a doc, then putting them in a vector of that doc
                tf_weight = 1 + math.log(tf)
                self.doc_vectors[doc_id][term] = tf_weight*idf

        # Pre-calculating doc norm for cosine similarity
        for doc_id, vec in self.doc_vectors.items():
            self.doc_norms[doc_id] = math.sqrt(sum(v*v for v in vec.values()))

    def vectorize_query(self, query_tokens):
        tf_q = Counter(query_tokens)
        q_vec = {}
        for term, tf in tf_q.items():
            if term in self.idf.keys():
                idf = self.idf[term]
                tf_weight = 1 + math.log(tf)
                q_vec[term] = tf_weight*idf

        q_norm = math.sqrt(sum(v*v for v in q_vec.values()))
        return q_norm,q_vec

    def rank(self, query, top_k=100):
        query_tokens = self.inverted_index.preprocessor.preprocess(query)
        q_norm,q_vec = self.vectorize_query(query_tokens)


        # if query has no valid terms from vocab
        if q_norm == 0:
            return []

        candidate_docs = set()
        for term in q_vec.keys():
            candidate_docs.update(self.inverted_index.get_posting_list(term).keys())

        scores = []
        # cosine similarity
        for doc_id in candidate_docs:
             d_vec = self.doc_vectors.get(doc_id,{})
             dot = sum(q_vec[t]*d_vec.get(t, 0) for t in q_vec)
             d_norm = self.doc_norms.get(doc_id,0)
             if d_norm > 0:
                 score = dot/(d_norm*q_norm)
                 scores.append((doc_id, score))

        # sort based on values, if there's a tie, sort based on doc_id
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:top_k]


def rank_dev_queries(ranker, dev_queries, top_k):
    all_rankings = {}

    for qid, query_text in dev_queries:
        top = ranker.rank(query_text, top_k)
        all_rankings[qid] = top

    return all_rankings


def show_sample_results(dev_queries, all_rankings, qrels, num_queries=3, top_k=10):
    if dev_queries and dev_queries[0] == ("query_id", "query_text"):
        dev_queries = dev_queries[1:]

    # اطمینان از اینکه تعداد از طول لیست بیشتر نشود
    n = min(num_queries, len(dev_queries))
    sample_queries = dev_queries[:n]

    for qid, query_text in sample_queries:
        print("=" * 60)
        print(f"Query ID: {qid}")
        print(f"Query: {query_text}")
        print(f"Top {top_k} results:\n")

        for rank, (doc_id, score) in enumerate(all_rankings[qid][:top_k], start=1):
            rel = qrels.get((qid, doc_id), 0)
            print(f"{rank:2d}. DocID={doc_id:<15} Score={score:.4f} Relevance={rel}")