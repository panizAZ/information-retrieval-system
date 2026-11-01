from collections import defaultdict, Counter


class InvertedIndex:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.num_docs = 0
        self.doc_lengths = {}
        # {term: {doc_id, frequency}}
        self.index = defaultdict(dict)

    def add_document(self, doc_id, text):
        tokens = self.preprocessor.preprocess(text)
        self.num_docs += 1
        self.doc_lengths[doc_id] = len(tokens)
        term_counts = Counter(tokens)

        for term, tf in term_counts.items():
            # {term: {doc_id, frequency}}
            self.index[term][doc_id] = tf

    def get_doc_frequency(self, term):
        """
        Get document frequency (DF) for a term
        """
        if term in self.index:
            return len(self.index[term])
        return 0

    def get_term_frequency(self, term, doc_id):
        """
        Get term frequency (TF) for a term-document pair
        """
        if term in self.index and doc_id in self.index[term]:
            return self.index[term][doc_id]
        return 0

    def get_posting_list(self, term):
        """
        shows {doc_id, frequency} for a term
        """
        if term in self.index:
            return self.index[term]
        return {}

    def get_stats(self):
        # Number of unique terms
        unique_terms_size = len(self.index)
        # Number of postings
        total_postings = sum(len(self.index[term]) for term in self.index)
        # 3 sample posting lists
        df_dict = {term:len(self.index[term]) for term in self.index}
        three_posting_samples = {
            term: dict(postings)
            for term, postings in list(self.index.items())[:3]
        }
        # Top 10 term by document frequency
        top_10_terms = sorted(df_dict.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "unique_terms_size": unique_terms_size,
            "total_postings": total_postings,
            "three_posting_samples": three_posting_samples,
            "top_10_terms": top_10_terms
        }




