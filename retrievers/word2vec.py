# retrievers/word2vec.py

import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.data_loader import load_collection, get_texts_and_ids

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

class Word2VecRetriever:
    def __init__(self, collection_path="data/collection.jsonl",
                 model_path="data/word2vec.model",
                 vector_path="data/word2vec_doc_vectors.npy",
                 vector_size=100, window=5, min_count=2):

        self.collection = load_collection(collection_path)
        self.texts, self.ids = get_texts_and_ids(self.collection)
        self.stop_words = set(stopwords.words("english"))
        self.tokenized_corpus = [self.tokenize(text) for text in self.texts]

        # Train or load Word2Vec model
        if os.path.exists(model_path) and os.path.exists(vector_path):
            print(f"🔁 Loading cached Word2Vec model and document vectors...")
            self.model = Word2Vec.load(model_path)
            self.doc_vectors = np.load(vector_path)
        else:
            print("⚙️ Training Word2Vec model and computing document vectors...")
            self.model = Word2Vec(sentences=self.tokenized_corpus,
                                  vector_size=vector_size,
                                  window=window,
                                  min_count=min_count,
                                  workers=4)
            self.doc_vectors = self._compute_doc_vectors()
            self.model.save(model_path)
            np.save(vector_path, self.doc_vectors)
            print(f"✅ Saved Word2Vec model to {model_path} and vectors to {vector_path}")

    def tokenize(self, text):
        """
        Tokenize text using NLTK with stopword removal and lowercasing.
        """
        return [w.lower() for w in word_tokenize(text) if w.isalnum() and w.lower() not in self.stop_words]

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

    def _compute_doc_vectors(self):
        """
        Compute TF-IDF-weighted average Word2Vec vectors for each document.
        """
        tfidf = TfidfVectorizer(tokenizer=self.tokenize, lowercase=True)
        tfidf.fit(self.texts)
        idf_weights = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))

        vectors = []
        for tokens in self.tokenized_corpus:
            vecs = []
            weights = []
            for word in tokens:
                if word in self.model.wv and word in idf_weights:
                    vecs.append(self.model.wv[word] * idf_weights[word])
                    weights.append(idf_weights[word])
            if vecs:
                vectors.append(np.sum(vecs, axis=0) / np.sum(weights))
            else:
                vectors.append(np.zeros(self.model.vector_size))
        return np.array(vectors)

    def retrieve(self, query, top_k=10, expand=False):
        """
        Retrieve top_k documents using cosine similarity to query vector.
        Set expand=True to enable query expansion.
        """
        if not query:
            return []

        query_tokens = self.expand_query(query) if expand else self.tokenize(query)
        query_vecs = [self.model.wv[word] for word in query_tokens if word in self.model.wv]
        if not query_vecs:
            return []

        query_vector = np.mean(query_vecs, axis=0).reshape(1, -1)
        sims = cosine_similarity(query_vector, self.doc_vectors)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.ids[i], float(sims[i])) for i in top_indices]
