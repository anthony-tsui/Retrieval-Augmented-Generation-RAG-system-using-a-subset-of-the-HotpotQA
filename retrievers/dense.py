# retrievers/dense.py

import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.data_loader import load_collection, get_texts_and_ids

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

class DenseRetriever:
    def __init__(self, model_name="thenlper/gte-small",
                 collection_path="data/collection.jsonl",
                 cache_path="data/gte_embeddings.npz",
                 normalize=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ DenseRetriever using device: {self.device}")

        self.model = SentenceTransformer(model_name, device=str(self.device))
        self.collection = load_collection(collection_path)
        self.texts, self.ids = get_texts_and_ids(self.collection)
        self.stop_words = set(stopwords.words("english"))
        self.normalize = normalize

        # Load or compute document embeddings
        if os.path.exists(cache_path):
            print(f"🔁 Loading cached GTE embeddings from {cache_path}")
            cache = np.load(cache_path, allow_pickle=True)
            self.doc_embeddings = cache["embeddings"]
            self.ids = list(cache["ids"])
        else:
            print("⚙️ Computing GTE document embeddings...")
            self.doc_embeddings = self.model.encode(
                self.texts,
                convert_to_numpy=True,
                show_progress_bar=True,
                normalize_embeddings=self.normalize,
                device=str(self.device)
            )
            np.savez(cache_path, embeddings=self.doc_embeddings, ids=np.array(self.ids))
            print(f"✅ Saved GTE embeddings to {cache_path}")

    def tokenize(self, text):
        return [w.lower() for w in word_tokenize(text) if w.isalnum() and w.lower() not in self.stop_words]

    def expand_query(self, query):
        tokens = self.tokenize(query)
        expanded = set(tokens)
        for token in tokens:
            for syn in wordnet.synsets(token):
                for lemma in syn.lemmas():
                    name = lemma.name().lower()
                    if name.isalnum() and name not in self.stop_words:
                        expanded.add(name)
        return " ".join(expanded)

    def retrieve(self, query, top_k=10, expand=False):
        if not query:
            return []

        query_text = self.expand_query(query) if expand else query
        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            device=str(self.device)
        )[0].reshape(1, -1)

        sims = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.ids[i], float(sims[i])) for i in top_indices]
