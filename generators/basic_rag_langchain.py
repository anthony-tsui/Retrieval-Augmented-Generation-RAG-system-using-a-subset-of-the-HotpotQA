"""
Basic RAG using LangChain with Fine-tuned Model
Uses train.jsonl to optimize system prompts for better Exact Match
"""
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
import json
from collections import Counter
import re

class BasicRAGLangChain:
    """
    Basic RAG with fine-tuned model and train.jsonl-optimized prompts.
    """
    
    def __init__(
        self,
        model_name="models/finetuned-qwen-3b",  # Use fine-tuned model by default
        device="cuda",
        temperature=0.2,
        max_new_tokens=128,  # Shorter for more concise answers
        use_fewshot=True,
        train_data_path="data/train.jsonl",
        num_examples=10
    ):
        """
        Initialize with fine-tuned model and train.jsonl analysis.
        """
        self.model_name = model_name

        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.use_fewshot = use_fewshot
        self.num_examples = num_examples
        
        print(f"🤖 Initializing BasicRAGLangChain with fine-tuned model: {model_name}...")
        
        # Load fine-tuned model
        # Load tokenizer with local path support
        from pathlib import Path
        
        # Determine if path is local
        is_local_path = os.path.isabs(model_name) and os.path.exists(model_name)
        
        # Load tokenizer
        # Normalize Windows paths to forward slashes for HuggingFace compatibility
        normalized_path = model_name.replace('\\', '/') if isinstance(model_name, str) else model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            normalized_path,
            local_files_only=True,
            trust_remote_code=True
        )
        
        if os.path.exists(os.path.join(model_name, "adapter_config.json")):
            print("   Loading LoRA fine-tuned model...")
            base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name.replace("\\", "/"),
                dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            self.model = PeftModel.from_pretrained(base_model, model_name)
            self.model.eval()
        else:
            print("   Loading full fine-tuned model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            self.model.eval()
        
        # Create pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.8,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        
        # Analyze train.jsonl to optimize prompts
        self.train_data = []
        self.answer_patterns = {}
        if train_data_path:
            self._analyze_training_data(train_data_path)
        
        # Create optimized prompt based on train.jsonl analysis
        # Prompt template not needed - using direct text prompts
        
        print(f"✅ BasicRAGLangChain with fine-tuned model initialized!")
    
    def _analyze_training_data(self, train_data_path):
        """Analyze train.jsonl to understand answer patterns"""
        print(f"   Analyzing train.jsonl for prompt optimization...")
        
        try:
            with open(train_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.train_data.append(json.loads(line))
            
            # Analyze answer patterns
            answer_lengths = [len(ex['answer'].split()) for ex in self.train_data[:1000]]
            answer_types = []
            
            for ex in self.train_data[:500]:
                answer = ex['answer'].lower().strip()
                # Classify answer types
                if answer.replace('.', '').isdigit() or any(c.isdigit() for c in answer[:4]):
                    answer_types.append('numeric')
                elif len(answer.split()) <= 3:
                    answer_types.append('short')
                elif ' and ' in answer or ',' in answer:
                    answer_types.append('list')
                else:
                    answer_types.append('phrase')
            
            self.answer_patterns = {
                'avg_length': sum(answer_lengths) / len(answer_lengths),
                'median_length': sorted(answer_lengths)[len(answer_lengths)//2],
                'type_distribution': Counter(answer_types)
            }
            
            print(f"   Loaded {len(self.train_data)} training examples")
            print(f"   Average answer length: {self.answer_patterns['avg_length']:.1f} words")
            print(f"   Most common answer type: {self.answer_patterns['type_distribution'].most_common(1)[0]}")
            
        except Exception as e:
            print(f"   ⚠️ Could not analyze training data: {e}")
    
    def _create_optimized_prompt(self):
        """Create prompt optimized based on train.jsonl patterns"""
        
        # System prompt learned from train.jsonl patterns
        system_prompt = """You are an expert question-answering system trained on HotpotQA.

ANSWER FORMAT (learned from training data):
- Average answer length: 2-4 words
- Be extremely concise - most answers are just names, places, dates, or short phrases
- NEVER add: "The answer is", "According to", "Based on the documents"
- Output ONLY the exact answer

EXAMPLES OF CORRECT FORMAT:
Q: "Who directed Inception?" → "Christopher Nolan" ✓
Q: "When was it released?" → "2010" ✓  
Q: "Where was he born?" → "London, England" ✓

CRITICAL RULES:
1. Extract the EXACT answer from documents
2. No explanations, no extra words
3. If uncertain, output the most likely answer from documents
4. For multi-part questions, give concise combined answer"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        return prompt
    
    def _get_fewshot_examples(self, query, num_examples=None):
        """Retrieve most similar training examples using word overlap"""
        if not self.use_fewshot or not self.train_data:
            return []
        
        num_examples = num_examples or self.num_examples
        
        def similarity(q1, q2):
            words1 = set(q1.lower().split())
            words2 = set(q2.lower().split())
            if not words1 or not words2:
                return 0
            return len(words1 & words2) / len(words1 | words2)
        
        # Find most similar examples
        scored = [(ex, similarity(query, ex['text'])) for ex in self.train_data]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [ex for ex, score in scored[:num_examples] if score > 0.1]
    
    def _format_documents(self, retrieved_docs):
        """Format retrieved documents concisely"""
        formatted = []
        for i, doc in enumerate(retrieved_docs[:10], 1):  # Limit to top 10
            doc_text = doc[0] if isinstance(doc, tuple) else doc.get('text', str(doc))
            # Truncate very long docs
            if len(doc_text) > 300:
                doc_text = doc_text[:300] + "..."
            formatted.append(f"[Doc{i}] {doc_text}")
        return "\n".join(formatted)
    
    def generate_answer(self, question, retrieved_docs, use_fewshot=None):
        """
        Generate answer using fine-tuned model and optimized prompts.
        """
        # Format documents
        context = self._format_documents(retrieved_docs)
        
        # Get few-shot examples
        use_fs = use_fewshot if use_fewshot is not None else self.use_fewshot
        examples_text = ""
        
        if use_fs:
            examples = self._get_fewshot_examples(question)
            if examples:
                examples_formatted = []
                for ex in examples[:5]:  # Use top 5 most similar
                    examples_formatted.append(f"Q: {ex['text']}\nA: {ex['answer']}")
                examples_text = "TRAINING EXAMPLES:\n" + "\n\n".join(examples_formatted) + "\n\n---\n\n"
        
        # Build prompt
        user_input = f"{examples_text}DOCUMENTS:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
        
        # Generate
        
        response = self.llm.invoke(user_input)
        
        # Extract and clean answer
        answer = self._extract_clean_answer(response)
# DEBUG: Print what we're generating
        print(f"[DEBUG] Raw response from LLM: {repr(response)[:200]}")
        print(f"[DEBUG] Cleaned answer: {repr(answer)}")
        print(f"[DEBUG] Answer length: {len(answer)} chars")
        return answer
    

    def _extract_clean_answer(self, generated_text):
        """Extract clean answer - aggressive cleaning for Exact Match"""
        if not generated_text:
            return "unknown"
        
        text = str(generated_text).strip()
        
        # CRITICAL FIX: Remove training examples if they were generated
        # The model sometimes repeats the training examples from the prompt
        if 'TRAINING EXAMPLES:' in text:
            # Split by the current question's answer marker
            # Look for pattern and take everything after that
            # Take the LAST answer (which should be for the current question)
            # and take everything after that
            parts = text.split('ANSWER:')
            if len(parts) > 1:
                # Take the LAST answer (which should be for the current question)
                text = parts[-1].strip()
        
        # Also handle case where it generates the Q: A: format
        if '\nQ:' in text:
            # Split by Q: and take the part before the next Q:
            parts = text.split('\nQ:')
            text = parts[0].strip()
            # If it starts with A:, remove it
            if text.startswith('A:'):
                text = text[2:].strip()
        
        # Remove XML/HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove common prefixes (case insensitive)
        prefixes_to_remove = [
            r"^answer:\s*",
            r"^the answer is:?\s*",
            r"^according to (?:the )?documents?,?\s*",
            r"^based on (?:the )?(?:provided )?(?:documents|information),?\s*",
            r"^(?:the )?document(?:s)? (?:state|mention|say)(?:s)? that\s*",
            r"^it (?:is )?(?:state|mention|say)(?:s)? that\s*",
            r"^(?:from|in) (?:the )?documents?,?\s*",
            r"^(?:as )?(?:stated|mentioned) in (?:the )?documents?,?\s*",
        ]
        
        for prefix_pattern in prefixes_to_remove:
            text = re.sub(prefix_pattern, '', text, flags=re.IGNORECASE)
            text = text.strip()
        
        # Take only first sentence/line
        lines = text.split('\n')
        text = lines[0].strip()
        
        # If there's a period, take first sentence
        if '.' in text:
            sentences = text.split('.')
            text = sentences[0].strip()
        
        # Remove surrounding quotes
        text = text.strip('"')
        
        # Limit length (most answers are short)
        words = text.split()
        if len(words) > 20:
            text = ' '.join(words[:20])
        
        return text if text else "unknown"
