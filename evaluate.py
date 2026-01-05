import os
"""
Evaluation Script for RAG System on Validation Set

This script evaluates your RAG system on the validation set to help you:
1. Test different configurations (retrievers, generators, models)
2. Measure performance (Exact Match, nDCG@10)
3. Optimize settings before final test submission

Usage:
    python evaluate.py --retriever bm25 --generator basic --model Qwen/Qwen2.5-1.5B-Instruct
    python evaluate.py --retriever dense --generator basic --use_fewshot
    python evaluate.py --max_queries 50  # Quick test on 50 queries
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import time
from collections import defaultdict
import re

# Import your modules
from retrievers.bm25 import BM25Retriever
from retrievers.dense import DenseRetriever
from retrievers.colbert import ColBERTRetriever
from retrievers.word2vec import Word2VecRetriever
from retrievers.e5_mistral import E5MistralRetriever
from generators import BasicRAG, MultiTurnRAG, AgenticRAG, UnifiedRAG
try:
    from generators import UnifiedRAGLangChain, LANGCHAIN_AVAILABLE
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain modules not available")
from utils.data_loader import load_collection, load_dataset



def clean_answer(text):
    """Clean up common answer prefixes/suffixes"""
    if not text:
        return text
    
    # Remove common prefixes
    prefixes_to_remove = [
        "the answer is:",
        "answer:",
        "according to the documents,",
        "based on the information,",
        "the document states that",
        "it is stated that",
        "the text mentions",
    ]
    
    text_lower = text.lower().strip()
    for prefix in prefixes_to_remove:
        if text_lower.startswith(prefix):
            text = text[len(prefix):].strip()
            text_lower = text.lower()
    
    # Remove quotes around answer
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        text = text[1:-1]
    
    return text.strip()


def normalize_answer(s):
    """Normalize answer for exact match comparison."""
    import re
    import string
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction, ground_truth):
    prediction = clean_answer(prediction)
    ground_truth = clean_answer(ground_truth)
    """Calculate exact match score."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def calculate_ndcg(retrieved_ids, relevant_ids, k=10):
    """Calculate nDCG@k for a single query."""
    import math
    
    # DCG calculation
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because index starts at 0
    
    # Ideal DCG
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    
    # nDCG
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_system(
    val_data_path="data/validation.jsonl",
    collection_path="data/collection.jsonl",
    retriever_type="bm25",
    generator_type="basic",
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    use_fewshot=False,
    train_data_path="data/train.jsonl",
    num_examples=3,
    top_k=10,
    max_queries=None,
    device="cuda"
):
    # Setup results folder structure
    import os
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = 'evaluation_results'
    os.makedirs(results_dir, exist_ok=True)
    retriever_dir = os.path.join(results_dir, retriever_type)
    os.makedirs(retriever_dir, exist_ok=True)
    result_filename = os.path.join(retriever_dir, f'{generator_type}_{timestamp}.json')
    
    """
    Evaluate RAG system on validation set.
    
    Returns:
        dict: Evaluation results with EM, nDCG, and per-query results
    """
    print("=" * 80)
    print("🔍 RAG SYSTEM EVALUATION ON VALIDATION SET")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Retriever: {retriever_type}")
    print(f"   Generator: {generator_type}")
    print(f"   Model: {model_name}")
    print(f"   Few-shot: {use_fewshot}")
    if use_fewshot:
        print(f"   Num examples: {num_examples}")
    print(f"   Top-K: {top_k}")
    print(f"   Device: {device}")
    print()
    
    # Load data
    print("📂 Loading data...")
    val_data = load_dataset(val_data_path)
    if max_queries:
        val_data = val_data[:max_queries]
        print(f"   Testing on {max_queries} queries (quick evaluation)")
    
    # Load collection for text lookup
    collection = load_collection(collection_path)
    doc_map = {doc['id']: doc for doc in collection}
    
    # Initialize retriever
    print(f"🔎 Initializing {retriever_type} retriever...")
    if retriever_type == "bm25":
        retriever = BM25Retriever(collection_path=collection_path)
    elif retriever_type == "dense":
        retriever = DenseRetriever(collection_path=collection_path, device=device)
    elif retriever_type == "colbert":
        retriever = ColBERTRetriever(collection_path=collection_path)
    elif retriever_type == "word2vec":
        retriever = Word2VecRetriever(collection_path=collection_path)
    elif retriever_type == "e5":
        retriever = E5MistralRetriever(collection_path=collection_path)
    else:
        raise ValueError(f"Unknown retriever: {retriever_type}")
    
    # Initialize generator
    print(f"🤖 Initializing {generator_type} generator...")
    generator_kwargs = {
        "model_name": model_name,
        "device": device
    }
    
    if generator_type == "basic":
        if use_fewshot:
            generator_kwargs["use_fewshot"] = True
            generator_kwargs["train_data_path"] = train_data_path
            generator_kwargs["num_examples"] = num_examples
        generator = BasicRAG(**generator_kwargs)
    elif generator_type == "multiturn":
        generator = MultiTurnRAG(**generator_kwargs)
    elif generator_type == "agentic":
        generator = AgenticRAG(**generator_kwargs)
    elif generator_type == "unified":
        if use_fewshot:
            generator_kwargs["use_fewshot"] = True
            generator_kwargs["train_data_path"] = train_data_path
            generator_kwargs["num_examples"] = num_examples
        generator = UnifiedRAG(**generator_kwargs)
        print(f"   Mode: Will test all three modes (basic, multiturn, agentic)")
    elif generator_type == "langchain":
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain modules not installed. Run: uv pip install langchain-core langchain langgraph langchain-huggingface")
        
        generator_kwargs = {
            "model_name": model_name,
            "device": device,
            "temperature": 0.2,
            "use_fewshot": use_fewshot,
            "train_data_path": train_data_path,
            "num_examples": num_examples
        }
        generator = UnifiedRAGLangChain(**generator_kwargs)
        print(f"   Using LangChain-based UnifiedRAG with fine-tuned model")
    else:
        raise ValueError(f"Unknown generator: {generator_type}")
    
    # Evaluation
    print(f"\n⚡ Evaluating on {len(val_data)} queries...\n")
    
    results = []
    em_scores = []
    ndcg_scores = []
    
    start_time = time.time()
    
    for item in tqdm(val_data, desc="Processing"):
        query_id = item["id"]
        question = item["text"]
        ground_truth_answer = item["answer"]
        relevant_doc_ids = item.get("supporting_ids", [])
        
        # Retrieve documents
        retrieved_docs = retriever.retrieve(question, top_k=top_k)
        
        # Convert tuple format (id, score) to document objects for generator
        retrieved_docs_with_text = []
        for doc_id, score in retrieved_docs:
            if doc_id in doc_map:
                retrieved_docs_with_text.append((doc_map[doc_id]['text'], score))
        
        # Handle tuple format (id, score) from BM25Retriever
        retrieved_ids = [doc_id for doc_id, _ in retrieved_docs]
        retrieved_scores = [score for _, score in retrieved_docs]
        # Handle tuple format (id, score) from BM25Retriever
        retrieved_ids = [doc[0] if isinstance(doc, tuple) else doc["id"] for doc in retrieved_docs]
        retrieved_scores = [doc[1] if isinstance(doc, tuple) else doc["score"] for doc in retrieved_docs]
        
        # Generate answer
        if generator_type == "basic":
            predicted_answer = generator.generate_answer(question, retrieved_docs_with_text)

        elif generator_type == "langchain":
            # Use basic mode by default for LangChain
            predicted_answer = generator.generate(
                question,
                retrieved_docs_with_text,
                mode="basic"
            )
        elif generator_type == "multiturn":
            # For single-turn eval, use basic generation
            predicted_answer = generator.generate_answer(question, retrieved_docs_with_text)
        elif generator_type == "agentic":
            # Use retrieval function wrapper
            def retrieval_fn(q):
                docs = retriever.retrieve(q, top_k=top_k)
                return [(doc["text"], doc["score"]) for doc in docs]
            
            predicted_answer, _ = generator.generate_with_workflow(
                question,
                retrieval_fn
            )
        elif generator_type == "unified":
            # Test with basic mode by default
            result = generator.generate(
                question, 
                retrieved_docs_with_text, 
                mode="basic"
            )
            # Handle both string and tuple returns
            predicted_answer = result[0] if isinstance(result, tuple) else result
        
        # Calculate metrics
        em = 1 if exact_match(predicted_answer, ground_truth_answer) else 0
        ndcg = calculate_ndcg(retrieved_ids, relevant_doc_ids, k=10)
        
        em_scores.append(em)
        ndcg_scores.append(ndcg)
        
        results.append({
            "id": query_id,
            "question": question,
            "predicted_answer": predicted_answer,
            "ground_truth_answer": ground_truth_answer,
            "exact_match": em,
            "ndcg@10": ndcg,
            "retrieved_docs": list(zip(retrieved_ids, retrieved_scores))
        })
    
    elapsed_time = time.time() - start_time
    
    # Calculate final metrics
    avg_em = sum(em_scores) / len(em_scores) * 100
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) * 100
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS")
    print("=" * 80)
    print(f"\n✅ Exact Match (EM):  {avg_em:.2f}%")
    print(f"📈 nDCG@10:           {avg_ndcg:.2f}%")
    print(f"⏱️  Total Time:        {elapsed_time:.2f}s")
    print(f"⚡ Avg Time/Query:    {elapsed_time/len(val_data):.2f}s")
    print()
    
    # Show some examples
    print("=" * 80)
    print("🔍 SAMPLE PREDICTIONS (First 3)")
    print("=" * 80)
    for i, r in enumerate(results[:3]):
        print(f"\n[Example {i+1}]")
        print(f"Question: {r['question']}")
        print(f"Predicted: {r['predicted_answer']}")
        print(f"Ground Truth: {r['ground_truth_answer']}")
        print(f"EM: {'✓' if r['exact_match'] else '✗'}, nDCG@10: {r['ndcg@10']:.3f}")
    
    # Save detailed results
    output_path = Path(result_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "retriever": retriever_type,
                "generator": generator_type,
                "model": model_name,
                "use_fewshot": use_fewshot,
                "num_examples": num_examples if use_fewshot else 0,
                "top_k": top_k
            },
            "metrics": {
                "exact_match": avg_em,
                "ndcg@10": avg_ndcg,
                "num_queries": len(val_data),
                "total_time": elapsed_time
            },
            "per_query_results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {output_path}")
    print("=" * 80)
    
    return {
        "exact_match": avg_em,
        "ndcg@10": avg_ndcg,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system on validation set")
    
    # Data paths
    parser.add_argument("--val_path", type=str, default="data/validation.jsonl",
                        help="Path to validation data")
    parser.add_argument("--collection_path", type=str, default="data/collection.jsonl",
                        help="Path to document collection")
    parser.add_argument("--train_path", type=str, default="data/train.jsonl",
                        help="Path to training data (for few-shot)")
    
    # Configuration
    parser.add_argument("--retriever", type=str, default="bm25",
                        choices=["bm25", "dense", "colbert", "word2vec", "e5"],
                        help="Retriever type")
    parser.add_argument("--generator", type=str, default="basic",
                        choices=["basic", "multiturn", "agentic", "unified", "langchain"],
                        help="Generator type")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model name or path to fine-tuned model")
    
    # Few-shot options
    parser.add_argument("--use_fewshot", action="store_true",
                        help="Enable few-shot learning")
    parser.add_argument("--num_examples", type=int, default=3,
                        help="Number of few-shot examples")
    
    # Other options
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of documents to retrieve")
    parser.add_argument("--test_mode", action="store_true",
                        help="Generate predictions for test set (test.jsonl)")
    parser.add_argument("--max_queries", type=int, default=None,
                        help="Max queries to evaluate (for quick testing)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run models")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluate_system(
        val_data_path=args.val_path,
        collection_path=args.collection_path,
        retriever_type=args.retriever,
        generator_type=args.generator,
        model_name=args.model,
        use_fewshot=args.use_fewshot,
        train_data_path=args.train_path,
        num_examples=args.num_examples,
        top_k=args.top_k,
        max_queries=args.max_queries,
        device=args.device
    )


if __name__ == "__main__":
    main()
