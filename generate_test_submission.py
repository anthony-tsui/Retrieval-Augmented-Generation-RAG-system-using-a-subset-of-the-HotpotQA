"""
Generate test_prediction.jsonl for final submission
Uses data/test.jsonl (1052 queries, NO ground truth answers)
"""

import sys
import os

# Change to code directory
os.chdir("D:/OneDrive/PolyU Msc IT/Natural Language Processing/Project_updated/code")

# Import required modules
from retrievers.e5_mistral import E5MistralRetriever
from generators import UnifiedRAGLangChain
import json
from tqdm import tqdm
from datetime import datetime

print("="*80)
print("🎯 GENERATING TEST PREDICTIONS FOR SUBMISSION")
print("="*80)

print("\n📋 Configuration:")
print("   Retriever: E5")
print("   Generator: LangChain UnifiedRAG")  
print("   Model: models/finetuned-qwen-1.5b")
print("   Few-shot: 10 examples")
print("   Top-K: 15 documents")
print("   Input: data/test.jsonl (1052 queries)")
print("   Output: test_prediction.jsonl")

# Load test data
print("\n📂 Loading test data...")
with open("data/test.jsonl", 'r', encoding='utf-8') as f:
    test_queries = [json.loads(line) for line in f if line.strip()]

print(f"   Loaded {len(test_queries)} test queries")

# Load document collection
print("\n📚 Loading document collection...")
with open("data/collection.jsonl", 'r', encoding='utf-8') as f:
    collection = {}
    for line in f:
        if line.strip():
            doc = json.loads(line)
            # Handle both 'id' and '_id' formats
            doc_id = doc.get('id') or doc.get('_id')
            collection[doc_id] = doc['text']

print(f"   Loaded {len(collection)} documents")

# Initialize retriever
print("\n🔎 Initializing E5 retriever...")
retriever = E5MistralRetriever(
    model_name="intfloat/e5-large-v2",
    collection_path="data/collection.jsonl",
    cache_path="data/e5_embeddings.npz"
)

# Initialize generator
print("\n🤖 Initializing LangChain generator...")
generator = UnifiedRAGLangChain(
    model_name="models/finetuned-qwen-1.5b",
    device="cuda",
    temperature=0.2,
    use_fewshot=True,
    train_data_path="data/train.jsonl",
    num_examples=10
)

# Generate predictions
print("\n⚡ Generating predictions...")
print("   This will take ~2-3 hours for 1052 queries")
print("="*80 + "\n")

predictions = []

for query_data in tqdm(test_queries, desc="Processing"):
    query_id = query_data.get('id') or query_data.get('_id')
    question = query_data['text']
    
    # Retrieve documents
    retrieved_docs = retriever.retrieve(question, top_k=15)
    
    # Add document text
    retrieved_docs_with_text = [
        (collection.get(doc_id, ""), score)
        for doc_id, score in retrieved_docs
    ]
    
    # Generate answer
    answer = generator.generate(
        question,
        retrieved_docs_with_text,
        mode="basic"
    )
    
    # Format prediction for submission
    prediction = {
        "id": query_id,
        "question": question,
        "answer": answer,
        "retrieved_docs": retrieved_docs[:10]  # Top 10 docs with scores
    }
    
    predictions.append(prediction)

# Save to test_prediction.jsonl
output_file = "test_predict.jsonl"
print(f"\n💾 Saving predictions to {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    for pred in predictions:
        f.write(json.dumps(pred, ensure_ascii=False) + '\n')

print(f"\n✅ COMPLETED!")
print("="*80)
print(f"📁 Output file: {output_file}")
print(f"📊 Total predictions: {len(predictions)}")

# Verify format
print("\n🔍 Verifying submission format...")
with open(output_file, 'r', encoding='utf-8') as f:
    first_pred = json.loads(f.readline())

print("✅ First prediction:")
print(f"   id: {first_pred['id']}")
print(f"   question: {first_pred['question'][:60]}...")
print(f"   answer: {first_pred['answer']}")
print(f"   retrieved_docs: {len(first_pred['retrieved_docs'])} documents")

# Verify all required fields
required_fields = ['id', 'question', 'answer', 'retrieved_docs']
if all(field in first_pred for field in required_fields):
    print("\n✅ All required fields present!")
    print("\n🎯 READY TO SUBMIT!")
    print("\n📊 Expected performance (based on validation):")
    print("   - Exact Match: ~40%")
    print("   - nDCG@10: ~75%")
else:
    print("\n⚠️ WARNING: Missing required fields!")

print("="*80)
