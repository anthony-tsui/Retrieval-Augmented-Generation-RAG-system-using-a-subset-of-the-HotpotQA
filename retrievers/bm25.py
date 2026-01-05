# retrievers/bm25.py

import os
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from utils.data_loader import load_collection, get_texts_and_ids

import nltk
# Simple tokenization without NLTK
from nltk.corpus import stopwords, wordnet

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

class BM25Retriever:
    def __init__(self, collection_path="data/collection.jsonl", cache_path="data/bm25_tokenized.pkl", k1=1.5, b=0.75):
        """
        Initialize BM25 retriever with smart tokenization and optional caching.
        """
        self.collection = load_collection(collection_path)
        self.texts, self.ids = get_texts_and_ids(self.collection)
        self.stop_words = set(stopwords.words("english"))

        # Load or build tokenized corpus
        if os.path.exists(cache_path):
            print(f"🔁 Loading cached BM25 tokenized corpus from {cache_path}")
            with open(cache_path, "rb") as f:
                self.tokenized_corpus = pickle.load(f)
        else:
            print(f"⚙️ Tokenizing corpus and caching to {cache_path}")
            self.tokenized_corpus = [self.tokenize(text) for text in self.texts]
            with open(cache_path, "wb") as f:
                pickle.dump(self.tokenized_corpus, f)

        # Initialize BM25 with tunable parameters
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)

    def tokenize(self, text):
        """
        Tokenize text using NLTK with stopword removal and lowercasing.
        """
        return [w.lower() for w in text.split() if w.isalnum() and w.lower() not in self.stop_words]

    def expand_query(self, query):
        """
        Expand query using WordNet synonyms.
        """
        tokens = self.tokenize(query)
        expanded = set(tokens)
        for token in tokens:
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    name = lemma.name().lower()
                    if name.isalnum() and name not in self.stop_words:
                        expanded.add(name)
        return list(expanded)

    def retrieve(self, query, top_k=10, expand=False):
        """
        Retrieve top_k documents using BM25 scoring.
        Set expand=True to enable query expansion.
        """
        if not query:
            return []

        tokenized_query = self.expand_query(query) if expand else self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[i], float(scores[i])) for i in top_indices]


