import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from utils.data_loader import load_collection, get_texts_and_ids
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

class ColBERTRetriever:
    def __init__(self, model_name="lightonai/GTE-ModernColBERT-v1",
                 collection_path="data/collection.jsonl",
                 cache_path="data/colbert_embeddings.npz",
                 normalize=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ ColBERT running on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.collection = load_collection(collection_path)
        self.texts, self.ids = get_texts_and_ids(self.collection)
        self.stop_words = set(stopwords.words("english"))
        self.normalize = normalize

        if os.path.exists(cache_path):
            print(f"🔁 Loading cached ColBERT embeddings from {cache_path}")
            cache = np.load(cache_path, allow_pickle=True)
            self.doc_embeddings = [torch.tensor(e, dtype=torch.float16).to(self.device) for e in cache["embeddings"]]
            self.ids = list(cache["ids"])
        else:
            print("⚙️ Computing ColBERT document embeddings...")
            self.doc_embeddings = self._encode_documents()
            np.savez_compressed(cache_path,
                                embeddings=np.array([e.cpu().numpy().astype(np.float16) for e in self.doc_embeddings], dtype=object),
                                ids=np.array(self.ids),
                                allow_pickle=True)
            print(f"✅ Saved ColBERT embeddings to {cache_path}")

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

    def _encode_documents(self):
        embeddings = []
        for i, text in enumerate(self.texts):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                output = self.model(**inputs).last_hidden_state.squeeze(0)  # [seq_len, hidden]
                if self.normalize:
                    output = torch.nn.functional.normalize(output, dim=-1)
            embeddings.append(output)
            if (i + 1) % 100 == 0:
                print(f"✅ Encoded {i + 1} documents")
        return embeddings

    def retrieve(self, query, top_k=10):
        """
        Retrieve top-k documents for a query using ColBERT MaxSim scoring.
        
        Args:
            query: Query string
            top_k: Number of documents to return
            
        Returns:
            List of tuples (doc_id, score)
        """
        # Encode query
        query_inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        query_inputs = {k: v.to(self.device) for k, v in query_inputs.items()}
        
        with torch.no_grad():
            query_output = self.model(**query_inputs).last_hidden_state.squeeze(0)  # [query_len, hidden_dim]
        
        # Calculate similarity with all documents
        sims = []
        for idx in range(len(self.doc_embeddings)):
            doc_tokens = self.doc_embeddings[idx]  # [doc_len, hidden_dim]
            
            # Ensure same dtype as query_output
            if doc_tokens.dtype != query_output.dtype:
                doc_tokens = doc_tokens.to(query_output.dtype)
            
            # Handle dimension issues
            if doc_tokens.dim() == 1:
                doc_tokens = doc_tokens.unsqueeze(0)
            
            # MaxSim: for each query token, find max similarity with doc tokens
            sim_matrix = torch.matmul(query_output, doc_tokens.mT)  # [query_len, doc_len]
            
            # Handle dimension for max
            if sim_matrix.dim() == 2:
                max_sim = sim_matrix.max(dim=1).values  # MaxSim over doc tokens
            else:
                max_sim = sim_matrix.max()
            
            # Average MaxSim scores across query tokens
            score = max_sim.mean().item()
            sims.append(score)
        
        # Sort and return top_k results
        top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        results = [(self.collection[idx]["id"], sims[idx]) for idx in top_indices]
        
        return results

