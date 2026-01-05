# retrievers/e5_mistral.py

import os
import numpy as np
import faiss
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer, CrossEncoder
from utils.data_loader import load_collection, get_texts_and_ids

class E5MistralRetriever:
    def __init__(self, model_name="intfloat/e5-large-v2",
                 collection_path="data/collection.jsonl",
                 cache_path="data/e5_embeddings.npz"):
        # Force GPU usage
        self.model = SentenceTransformer(model_name, device="cuda")

        self.collection = load_collection(collection_path)
        self.texts, self.ids = get_texts_and_ids(self.collection)

        if os.path.exists(cache_path):
            print(f"🔁 Loading cached embeddings from {cache_path}")
            cache = np.load(cache_path, allow_pickle=True)
            self.doc_embeddings = cache["embeddings"]
            self.ids = list(cache["ids"])
        else:
            print("⚙️ Computing embeddings on GPU...")
            passages = [f"passage: {text}" for text in self.texts]
            self.doc_embeddings = self.model.encode(
                passages, convert_to_numpy=True, show_progress_bar=True
            )
            self.doc_embeddings = normalize(self.doc_embeddings, norm="l2")
            np.savez(cache_path, embeddings=self.doc_embeddings, ids=np.array(self.ids))
            print(f"✅ Saved embeddings to {cache_path}")

        # FAISS index (GPU optional)
        dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.doc_embeddings)

        # Cross-encoder reranker (also on GPU)
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")

    def retrieve(self, query, top_k=10, rerank=True):
        query_embedding = self.model.encode([f"query: {query}"], convert_to_numpy=True)
        query_embedding = normalize(query_embedding, norm="l2")

        sims, indices = self.index.search(query_embedding, top_k*5)
        candidates = indices[0]

        if rerank:
            pairs = [(query, self.texts[i]) for i in candidates]
            scores = self.reranker.predict(pairs)
            reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]
            return [(self.ids[i], float(score)) for i, score in reranked]
        else:
            return [(self.ids[i], float(sims[0][j])) for j, i in enumerate(candidates[:top_k])]
