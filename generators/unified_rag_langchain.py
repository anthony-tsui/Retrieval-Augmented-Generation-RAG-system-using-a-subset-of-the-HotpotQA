"""
Unified RAG using LangChain/LangGraph
Combines all three generation approaches with fine-tuned model
"""

from .basic_rag_langchain import BasicRAGLangChain
from .multiturn_langgraph import MultiTurnRAGLangGraph
from .agentic_langgraph import AgenticRAGLangGraph


class UnifiedRAGLangChain:
    """
    Unified RAG interface combining Basic, Multi-turn, and Agentic approaches.
    Uses fine-tuned model with lazy loading for better performance.
    """
    
    def __init__(
        self,
        model_name="models/finetuned-qwen-1.5b",
        device="cuda",
        temperature=0.2,
        use_fewshot=True,
        train_data_path="data/train.jsonl",
        num_examples=10
    ):
        """
        Initialize Unified RAG with lazy loading for better performance.
        
        Args:
            model_name: Fine-tuned model path
            device: cuda or cpu
            temperature: Generation temperature
            use_fewshot: Whether to use few-shot learning
            train_data_path: Path to training data
            num_examples: Number of few-shot examples
        """
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.use_fewshot = use_fewshot
        self.train_data_path = train_data_path
        self.num_examples = num_examples
        
        print(f"🤖 Initializing UnifiedRAGLangChain with {model_name}...")
        
        # Lazy loading - only load models when first used
        self._basic_rag = None
        self._multiturn_rag = None
        self._agentic_rag = None
        
        # Load only BasicRAG by default (most commonly used)
        print("   Loading BasicRAGLangChain...")
        self._basic_rag = BasicRAGLangChain(
            model_name=model_name,
            device=device,
            temperature=temperature,
            use_fewshot=use_fewshot,
            train_data_path=train_data_path,
            num_examples=num_examples
        )
        
        print(f"✅ UnifiedRAGLangChain initialized successfully!")
        print(f"   - BasicRAG: Ready (Few-shot: {use_fewshot})")
        print(f"   - MultiTurnRAG: Will load on first use")
        print(f"   - AgenticRAG: Will load on first use")
    
    @property
    def basic_rag(self):
        """Access BasicRAG"""
        return self._basic_rag
    
    @property
    def multiturn_rag(self):
        """Lazy load MultiTurnRAG on first access"""
        if self._multiturn_rag is None:
            print("   Loading MultiTurnRAGLangGraph...")
            self._multiturn_rag = MultiTurnRAGLangGraph(
                model_name=self.model_name,
                device=self.device,
                temperature=self.temperature
            )
            print("   ✅ MultiTurnRAG loaded!")
        return self._multiturn_rag
    
    @property
    def agentic_rag(self):
        """Lazy load AgenticRAG on first access"""
        if self._agentic_rag is None:
            print("   Loading AgenticRAGLangGraph...")
            self._agentic_rag = AgenticRAGLangGraph(
                model_name=self.model_name,
                device=self.device,
                temperature=self.temperature
            )
            print("   ✅ AgenticRAG loaded!")
        return self._agentic_rag
    
    def generate(
        self,
        question: str,
        retrieved_docs: list,
        mode: str = "basic",
        **kwargs
    ) -> str:
        """
        Generate answer using selected RAG mode.
        
        Args:
            question: Input question
            retrieved_docs: List of (doc_text, score) tuples
            mode: "basic", "multiturn", or "agentic"
            **kwargs: Additional arguments for specific modes
            
        Returns:
            Generated answer string
        """
        if mode == "basic":
            return self.basic_rag.generate_answer(question, retrieved_docs)
        elif mode == "multiturn":
            # MultiTurn mode
            return self.multiturn_rag.generate_answer(question, retrieved_docs)
        elif mode == "agentic":
            # Agentic mode with workflow
            retrieval_fn = kwargs.get('retrieval_fn')
            if retrieval_fn:
                answer, reasoning_steps = self.agentic_rag.generate_with_workflow(
                    question,
                    retrieval_fn
                )
                return answer, reasoning_steps
            else:
                # Fallback to basic generation
                return self.agentic_rag.generate_answer(question, retrieved_docs)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'basic', 'multiturn', or 'agentic'")
