import csv
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.token_pattern = re.compile(r'(?u)\b\w\w+\b')

    def preprocess(self, text, lower_case=True, remove_stops=True, use_stemming=True):
        if not text:
            return []

        tokens = self.token_pattern.findall(text)

        if lower_case:
            tokens = [token.lower() for token in tokens]

        if remove_stops:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]

        if use_stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]


        return tokens

    def get_vocab_size(self, corpus_token_list):
        vocab = set()
        for token in corpus_token_list:
            vocab.update(token)
        return len(vocab)


def load_documents(path):
    documents = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_id = row["doc_id"]
            title = row.get("title", "") or ""
            text = row.get("text", "") or ""
            content = f"{title} {text}".strip()
            documents.append((doc_id, content))

    return documents


def load_queries(path):
    queries = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if not row:
                continue

            if len(row) < 2:
                continue

            query_id = row[0].strip()
            query_text = row[1].strip()

            queries[query_id] = query_text

    return queries

def load_qrels(path):
    qrels = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            query_id = row["query_id"]
            doc_id = row["doc_id"]
            relevance = int(row["relevance"])
            qrels[(query_id, doc_id)] = relevance

    return qrels

