from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer


def penn_to_wordnet(tag):
    if tag.startswith("N"):
        return wn.NOUN
    if tag.startswith("V"):
        return wn.VERB
    return None


class WordNetSemanticReranker:
    def __init__(self, lexical_ranker, alpha=0.8):
        self.lexical_ranker = lexical_ranker
        self.alpha = alpha
        self.lemmatizer = WordNetLemmatizer()
        self.context_cache = {}
        self.synset_cache = {}

    def score_candidates(self, query_text, document_dicts, lexical_results, rerank_k=20):
        import time

        start_time = time.time() # شروع زمان‌سنجی
        q_words = self.extract_context_words(query_text)
        print(f"  - Extracted query words for {query_text[:30]}... in {time.time() - start_time:.2f}s") # زمان پردازش کوئری

        top_candidates = lexical_results[:rerank_k]
        lexical_tail = lexical_results[rerank_k:]

        scored_top = []
        doc_proc_start_time = time.time() # شروع زمان‌سنجی پردازش اسناد
        for i, (doc_id, lex_score) in enumerate(top_candidates):
            doc_text = document_dicts[doc_id]
            doc_words = self.extract_context_words(doc_text)
            sem_score = self.semantic_score_from_words(q_words, doc_words)
            scored_top.append((doc_id, lex_score, sem_score))
            if (i + 1) % 10 == 0: # پرینت هر 10 سند
                print(f"    Processed {i+1}/{len(top_candidates)} docs for query in {time.time() - doc_proc_start_time:.2f}s")

        print(f"  - Finished scoring {len(top_candidates)} candidates in {time.time() - start_time:.2f}s") # زمان کل متد
        return scored_top, lexical_tail

    def combine_scores(self, scored_top, lexical_tail, alpha):
        """
        اCalled in Alpha Sweep loop in system.py
        """
        reranked_top = []
        for doc_id, lex_score, sem_score in scored_top:
            #  alpha * lexical + (1-alpha) * semantic
            final_score = alpha * lex_score + (1 - alpha) * sem_score
            reranked_top.append((doc_id, final_score, lex_score, sem_score))

        # مرتب‌سازی بر اساس امتیاز نهایی
        reranked_top.sort(key=lambda x: (-x[1], x[0]))

        # چسباندن دمِ لیست (tail) که تغییری نکرده
        final_results = reranked_top[:]
        for doc_id, lex_score in lexical_tail:
            final_results.append((doc_id, lex_score, lex_score, None))

        return final_results

    def extract_context_words(self, text):
        if text in self.context_cache:
            return self.context_cache[text]

        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)

        content_words = set()
        for token, tag in pos_tags:
            wn_pos = penn_to_wordnet(tag)
            if wn_pos in {wn.NOUN, wn.VERB} and token.isalpha():
                lemma = self.lemmatizer.lemmatize(token.lower(), pos=wn_pos)
                content_words.add((lemma, wn_pos))

        self.context_cache[text] = content_words
        return content_words

    def get_synsets(self, word, pos):
        key = (word, pos)
        if key not in self.synset_cache:
            self.synset_cache[key] = wn.synsets(word, pos=pos)
        return self.synset_cache[key]

    def max_wup_similarity(self, query_word, query_pos, doc_words):
        query_synsets = self.get_synsets(query_word, query_pos)
        if not query_synsets:
            return 0.0

        best_score = 0.0
        for doc_word, doc_pos in doc_words:
            doc_synsets = self.get_synsets(doc_word, doc_pos)
            if not doc_synsets:
                continue

            for qs in query_synsets:
                for ds in doc_synsets:
                    sim = qs.wup_similarity(ds)
                    if sim is not None and sim > best_score:
                        best_score = sim

        return best_score

    def semantic_score_from_words(self, q_words, doc_words):
        if not q_words or not doc_words:
            return 0.0

        scores = []
        for q_word, q_pos in q_words:
            best = self.max_wup_similarity(q_word, q_pos, doc_words)
            scores.append(best)

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def rerank(self, query_text, document_dicts, initial_k=100, rerank_k=20,lexical_scores=None,):
        if lexical_scores is None:
            lexical_scores = self.lexical_ranker.rank(query_text, initial_k)

        top_candidates = lexical_scores[:rerank_k]
        remaining_docs = lexical_scores[rerank_k:]

        q_words = self.extract_context_words(query_text)

        reranked_top = []
        for doc_id, lexical_score in top_candidates:
            doc_text = document_dicts[doc_id]
            doc_words = self.extract_context_words(doc_text)
            sem_score = self.semantic_score_from_words(q_words, doc_words)
            final_score = self.alpha * lexical_score + (1 - self.alpha) * sem_score
            reranked_top.append((doc_id, final_score, lexical_score, sem_score))

        reranked_top.sort(key=lambda x: (-x[1], x[0]))

        final_results = reranked_top[:]
        for doc_id, lexical_score in remaining_docs:
            final_results.append((doc_id, lexical_score, lexical_score, None))

        return final_results