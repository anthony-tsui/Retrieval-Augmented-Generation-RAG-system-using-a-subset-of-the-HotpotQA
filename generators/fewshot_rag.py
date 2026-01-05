"""
Few-Shot RAG Generator Module

This module enhances the basic RAG by using training examples as in-context demonstrations
to guide the LLM toward better answer quality and formatting.

Key Features:
- Selects relevant training examples based on query similarity
- Uses these as few-shot demonstrations in the prompt
- Improves answer consistency and quality without fine-tuning
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FewShotRAG:
    """
    Few-shot RAG generator that uses training examples as demonstrations.
    
    This approach improves generation quality by showing the LLM similar 
    question-answer pairs from the training set.
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        train_path: str = "data/train.jsonl",
        device: str = "cuda",
        num_examples: int = 3
    ):
        """
        Initialize the few-shot RAG generator.
        
        Args:
            model_name: Qwen model identifier
            train_path: Path to training data (for few-shot examples)
            device: cuda or cpu
            num_examples: Number of few-shot examples to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Optimized model loading with proper dtype handling
        if device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                low_cpu_mem_usage=True
            ).to(torch.float16)  # Convert after loading for compatibility
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        
        self.model.eval()  # Set to evaluation mode to disable gradients
        self.device = device
        self.num_examples = num_examples
        
        # Load training data for few-shot examples
        self.train_data = self._load_train_data(train_path)
        
        # Build TF-IDF index for quick example retrieval
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.train_vectors = self.vectorizer.fit_transform(
            [item['text'] for item in self.train_data]
        )
        
        print(f"✓ Loaded {len(self.train_data)} training examples for few-shot learning")
    
    def _load_train_data(self, train_path: str) -> List[Dict]:
        """Load training data from JSONL file."""
        train_data = []
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                train_data.append(data)
        return train_data
    
    def _select_few_shot_examples(self, query: str) -> List[Dict]:
        """
        Select most relevant training examples for few-shot prompting.
        
        Args:
            query: User's question
            
        Returns:
            List of similar training examples
        """
        # Convert query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Compute similarity with all training examples
        similarities = cosine_similarity(query_vector, self.train_vectors)[0]
        
        # Get top-k most similar examples
        top_indices = np.argsort(similarities)[-self.num_examples:][::-1]
        
        examples = [self.train_data[idx] for idx in top_indices]
        return examples
    
    def create_prompt(
        self, 
        question: str, 
        retrieved_docs: List[Tuple[str, float]],
        collection: Dict[str, str] = None
    ) -> str:
        """
        Create prompt with few-shot examples.
        
        Args:
            question: User query
            retrieved_docs: List of (doc_id, score) or (doc_text, score)
            collection: Optional document collection mapping
            
        Returns:
            Formatted prompt string
        """
        # Select relevant few-shot examples
        examples = self._select_few_shot_examples(question)
        
        # Format retrieved documents
        if collection and isinstance(retrieved_docs[0][0], str) and retrieved_docs[0][0].startswith('doc-'):
            # If we have doc IDs, fetch the actual text
            context = "\n\n".join([
                f"[Document {i+1}]\n{collection.get(doc_id, 'Document not found')}"
                for i, (doc_id, score) in enumerate(retrieved_docs[:5])
            ])
        else:
            # Already have text
            context = "\n\n".join([
                f"[Document {i+1}]\n{doc[0]}"
                for i, doc in enumerate(retrieved_docs[:5])
            ])
        
        # Build few-shot demonstrations
        demonstrations = []
        for i, ex in enumerate(examples, 1):
            demo = f"""Example {i}:
Question: {ex['text']}
Answer: {ex['answer']}"""
            demonstrations.append(demo)
        
        few_shot_text = "\n\n".join(demonstrations)
        
        # System prompt with instructions
        system_prompt = """You are a helpful assistant that answers questions based on provided documents.

IMPORTANT INSTRUCTIONS:
1. Read the provided documents carefully
2. Answer the question using ONLY information from the documents
3. Keep answers concise and factual (typically 1-5 words or a short phrase)
4. If the answer cannot be found, respond with "I cannot answer based on the provided information"
5. Format your response as: Answer: <your answer>

Here are some examples of good answers:"""
        
        # Construct full prompt
        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n{few_shot_text}"},
            {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    def generate_answer(
        self, 
        question: str, 
        retrieved_docs: List[Tuple[str, float]],
        collection: Dict[str, str] = None,
        max_new_tokens: int = 128
    ) -> str:
        """
        Generate answer using few-shot prompting.
        
        Args:
            question: User query
            retrieved_docs: Retrieved documents
            collection: Document collection mapping
            max_new_tokens: Max tokens to generate
            
        Returns:
            Generated answer string
        """
        prompt = self.create_prompt(question, retrieved_docs, collection)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,  # Lower temperature for more focused answers
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return self._extract_answer(generated_text)
    
    def _extract_answer(self, generated_text: str) -> str:
        """Extract clean answer from generated text."""
        # Look for "Answer:" pattern
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
            return answer.split("\n")[0].strip()
        
        # Fallback: return first line
        return generated_text.split("\n")[0].strip()


if __name__ == "__main__":
    # Example usage
    print("Testing Few-Shot RAG Generator...")
    
    generator = FewShotRAG(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        train_path="data/train.jsonl",
        num_examples=3
    )
    
    # Test query
    test_question = "Which airport is located in Maine?"
    test_docs = [
        ("Knox County Regional Airport is a public airport located in Maine.", 0.95),
        ("Sacramento International Airport is located in California.", 0.85)
    ]
    
    answer = generator.generate_answer(test_question, test_docs)
    print(f"\nQuestion: {test_question}")
    print(f"Answer: {answer}")
