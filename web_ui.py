"""
Web-based User Interface for RAG System
Uses E5MistralRetriever and fine-tuned model
Matches configuration from generate_test_submission.py
"""

from flask import Flask, render_template, request, jsonify, session
from retrievers.e5_mistral import E5MistralRetriever
from generators.basic_rag_langchain import BasicRAGLangChain
from generators.agentic_langgraph import AgenticRAGLangGraph
from generators.multiturn_langgraph import MultiTurnRAGLangGraph
import json
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'  # Required for sessions

# Global variables for RAG components
retriever = None
basic_rag = None
agentic_rag = None
multiturn_rag = None
collection = None

def initialize_rag():
    """Initialize RAG components with same config as generate_test_submission.py"""
    global retriever, basic_rag, agentic_rag, multiturn_rag, collection
    
    print("🔧 Initializing RAG components...")
    
    # Initialize retriever - E5MistralRetriever with e5_embeddings.npz
    print("   🔎 Loading E5 retriever...")
    retriever = E5MistralRetriever(
        model_name="intfloat/e5-large-v2",
        collection_path="data/collection.jsonl",
        cache_path="data/e5_embeddings.npz"
    )
    
    # Load collection
    print("   📚 Loading document collection...")
    with open("data/collection.jsonl", 'r', encoding='utf-8') as f:
        collection = {}
        for line in f:
            if line.strip():
                doc = json.loads(line)
                doc_id = doc.get('id') or doc.get('_id')
                collection[doc_id] = doc['text']
    
    # Initialize BasicRAG (always load)
    print("   🤖 Loading BasicRAGLangChain...")
    basic_rag = BasicRAGLangChain(
        model_name="models/finetuned-qwen-1.5b",
        device="cuda",
        temperature=0.2,
        use_fewshot=True,
        train_data_path="data/train.jsonl",
        num_examples=10
    )
    
    # Pre-load AgenticRAG
    print("   🤖 Loading AgenticRAGLangGraph...")
    agentic_rag = AgenticRAGLangGraph(
        model_name="models/finetuned-qwen-1.5b",
        device="cuda",
        temperature=0.2
    )
    
    print("✅ RAG components initialized!")
    print("   - BasicRAG: Ready")
    print("   - AgenticRAG: Ready")
    print("   - MultiTurnRAG: Will load on first use")

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    """Handle query requests"""
    global multiturn_rag
    
    data = request.json
    question = data.get('question', '')
    mode = data.get('mode', 'basic')
    
    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400
    
    try:
        # Retrieve documents
        retrieved_docs = retriever.retrieve(question, top_k=15)
        
        # Format retrieved docs for display
        retrieved_docs_display = []
        for i, (doc_id, score) in enumerate(retrieved_docs[:10], 1):
            doc_text = collection.get(doc_id, "Document not found")
            retrieved_docs_display.append({
                'rank': i,
                'doc_id': doc_id,
                'score': round(score, 3),
                'text': doc_text[:300] + "..." if len(doc_text) > 300 else doc_text
            })
        
        # Prepare docs with text for generation
        retrieved_docs_with_text = [
            (collection.get(doc_id, ""), score)
            for doc_id, score in retrieved_docs
        ]
        
        # Generate answer based on mode
        reasoning_steps = []
        answer = ""
        
        if mode == 'basic':
            # Use BasicRAG
            answer = basic_rag.generate_answer(question, retrieved_docs_with_text)
            
        elif mode == 'multiturn':
            # Lazy load MultiTurnRAG
            if multiturn_rag is None:
                print("   🔄 Loading MultiTurnRAGLangGraph...")
                multiturn_rag = MultiTurnRAGLangGraph(
                    model_name="models/finetuned-qwen-1.5b",
                    device="cuda",
                    temperature=0.2
                )
                print("   ✅ MultiTurnRAG loaded!")
            
            # CRITICAL: Restore conversation history BEFORE generate_answer
            if 'conversation_history' in session:
                multiturn_rag.conversation_history = session['conversation_history']
                print(f"   📚 Restored {len(multiturn_rag.conversation_history)} previous turns from session")
                print(f"   📝 Previous turns:")
                for idx, turn in enumerate(multiturn_rag.conversation_history):
                    print(f"      Turn {idx+1}: Q: {turn.get('question', 'N/A')[:50]}...")
                    print(f"               A: {turn.get('answer', 'N/A')[:50]}...")
            else:
                multiturn_rag.conversation_history = []
                print("   📚 Starting new conversation (no history)")
            
            print(f"   🔍 Current question: {question}")
            
            # Generate answer with metadata (this will use the restored history)
            answer, metadata = multiturn_rag.generate_answer(question, retrieved_docs_with_text)
            
            # Save updated conversation history to session
            session['conversation_history'] = multiturn_rag.conversation_history
            session.modified = True
            print(f"   💾 Saved conversation history ({len(multiturn_rag.conversation_history)} turns)")
            
            # Add reasoning steps for multi-turn
            reasoning_steps = [{
                'step': 'Query Reformulation',
                'original_query': question,
                'reformulated_query': metadata.get('reformulated_query', question),
                'history_length': len(multiturn_rag.conversation_history) - 1  # Exclude current turn
            }]
            
        elif mode == 'agentic':
            # Use AgenticRAG with workflow
            def retrieval_fn(query, top_k=10):
                docs = retriever.retrieve(query, top_k=top_k)
                return [(collection.get(doc_id, ""), score) for doc_id, score in docs]
            
            answer, reasoning_steps = agentic_rag.generate_with_workflow(
                question,
                retrieval_fn
            )
        
        return jsonify({
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs_display,
            'reasoning_steps': reasoning_steps,
            'mode': mode
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear multi-turn conversation history"""
    global multiturn_rag
    
    # Clear session history
    if 'conversation_history' in session:
        session.pop('conversation_history')
        session.modified = True
    
    # Clear instance history
    if multiturn_rag is not None:
        multiturn_rag.clear_history()
    
    return jsonify({'message': 'Conversation history cleared'})

if __name__ == '__main__':
    initialize_rag()
    app.run(debug=True, host='0.0.0.0', port=5000)
