# RAG System - Setup and Running Instructions

## 1. Step-by-Step Instructions for Running the System

### 1.1 Environment Setup

#### Step 1: Create Virtual Environment
cd code
python -m venv .venv

#### Step 2: Activate Virtual Environment
Windows: .venv\Scripts\activate
Linux/Mac: source .venv/bin/activate

#### Step 3: Install Dependencies
pip install -r requirements.txt

#### Step 4: Download NLTK Data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

### 1.2 Reproducing test_predict.jsonl
To generate the test_predict.jsonl file:
cd code
python generate_test_submission.py

#### What this does:
1. Loads the E5 Mistral retriever with pre-computed embeddings (data/e5_embeddings.npz)
2. Loads the fine-tuned Qwen 1.5B model from models/finetuned-qwen-1.5b/
3. Reads test queries from data/test.jsonl
4. For each query:
5. Retrieves top-15 relevant documents using E5 Mistral retriever
6. Generates answer using fine-tuned model with few-shot prompting
7. Outputs predictions to test_predict.jsonl

#### Output format:
{"id": "query-id", "answer": "generated answer"}
Expected runtime: ~10-15 minutes for full test set (depends on GPU)

### 1.3 Running the Web Interface
To explore the three RAG modes interactively:
cd code
python web_ui.py

Then open: http://localhost:5000

Available modes:

⚡ Basic RAG: Simple Q&A without memory
🔄 MultiTurn RAG: Conversation with memory & query reformulation (Feature A)
🧠 Agentic Workflow: Multi-step reasoning with decomposition (Feature B)

## 2. Environment Setup - Package Versions
### 2.1 Python Version
**Python 3.11.12** (Required)

### 2.2 Core Dependencies

**Deep Learning & Transformers:**
- `torch==2.5.1` (install separately with CUDA/CPU flag)
- `transformers==4.47.1` - Transformer models (ColBERT, E5, all generators)
- `accelerate==1.2.1` - Model acceleration
- `peft==0.14.0` - Parameter-efficient fine-tuning (LoRA for finetune_qwen.py)
- `bitsandbytes==0.45.0` - 8-bit optimizers (GPU only, used in finetune_qwen.py)
- `safetensors==0.4.5` - Safe tensor serialization

**LangChain & LangGraph (Required for ALL generators):**
- `langchain==0.3.13` - LangChain framework (BasicRAG, MultiTurnRAG, AgenticRAG)
- `langchain-core==0.3.28` - LangChain core components (prompts, messages, base classes)
- `langchain-community==0.3.13` - LangChain community integrations
- `langgraph==0.2.59` - LangGraph state graphs (MultiTurnRAG, AgenticRAG workflows)
- `langchain-huggingface==0.1.2` - HuggingFace pipeline integration (all generators)
- `langsmith==0.2.5` - LangSmith for tracing and debugging

**Embeddings & Retrieval:**
- `sentence-transformers==3.3.1` - Sentence embedding models (Dense, E5, ColBERT retrievers)
- `rank-bm25==0.2.2` - BM25 ranking algorithm (BM25 retriever)
- `faiss-cpu==1.9.0.post1` - Vector similarity search (E5 retriever)
- `gensim==4.3.3` - Word2Vec and topic modeling (Word2Vec retriever)

**NLP Utilities:**
- `nltk==3.9.1` - Natural Language Toolkit (all retrievers use this)

**Web Framework (for web_ui.py):**
- `Flask==3.1.0` - Web framework for web UI
- `Werkzeug==3.1.3` - WSGI utilities for Flask
- `Jinja2==3.1.4` - Template engine for Flask
- `MarkupSafe==3.0.2` - Safe string handling for templates

**Data Processing & ML:**
- `numpy==1.26.4` - Numerical computing (all retrievers, evaluate.py)
- `pandas==2.2.3` - Data manipulation
- `scikit-learn==1.5.2` - Machine learning utilities (Dense, E5, Word2Vec, answer_validator.py)
- `datasets==3.2.0` - HuggingFace datasets (finetune_qwen.py)

**Utilities:**
- `tqdm==4.67.1` - Progress bars (evaluate.py, finetune_qwen.py)
- `requests==2.32.5` - HTTP library
- `huggingface-hub==0.27.0` - HuggingFace Hub client
- `tokenizers==0.21.0` - Fast tokenizers
- `filelock==3.20.0` - File locking
- `fsspec==2024.10.0` - Filesystem spec
- `regex==2024.11.6` - Regular expressions
- `PyYAML==6.0.2` - YAML parser
- `packaging==24.2` - Package version utilities

### 2.3 System Requirements
**Minimum Requirements:**
- RAM: 16GB (32GB recommended)
- Storage: ~10GB for models and data
- GPU: CUDA-compatible GPU (optional but recommended for faster inference)

**Recommended Setup:**
- GPU: NVIDIA GPU with 8GB+ VRAM
- CUDA: 12.1 or compatible
- OS: Windows 10/11, Linux, or macOS

### 2.4 Installation Notes
**For GPU Support:**
- Ensure CUDA toolkit is installed
- Install PyTorch FIRST before other packages:
  ```bash
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
- Visit: https://pytorch.org/get-started/locally/

If Dependencies Conflict:
pip install --upgrade pip
pip install -r requirements.txt --upgrade

Complete Installation Commands:
GPU with CUDA 12.1:
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

## 3. File Descriptions
Key Files:
generate_test_submission.py: Main script for reproducing test_predict.jsonl
web_ui.py: Interactive web interface for exploring RAG modes
requirements.txt: Complete list of dependencies with versions
data/e5_embeddings.npz: Pre-computed embeddings (used by generate_test_submission.py)
models/finetuned-qwen-1.5b/: Fine-tuned model weights

Configuration:
The submission script uses:
Retriever: E5MistralRetriever (intfloat/e5-large-v2 with pre-computed embeddings)
Generator: Fine-tuned Qwen 1.5B with LoRA adapters
Few-shot: 10 examples from train.jsonl
Top-k: 15 documents per query
Temperature: 0.2

## 4. Notes
The system uses a fine-tuned Qwen 1.5B model which has limitations for complex entity disambiguation
MultiTurn RAG (Feature A) maintains conversation memory via Flask sessions
Agentic Workflow (Feature B) implements query decomposition, retrieval, synthesis, verification, and reflection steps
For best results on simple questions, use Basic RAG mode; for follow-up questions, use MultiTurn mode
