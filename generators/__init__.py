"""
Generators module - LangChain and LangGraph based RAG implementations
"""

# LangChain/LangGraph implementations (only these exist)
from .basic_rag_langchain import BasicRAGLangChain
from .multiturn_langgraph import MultiTurnRAGLangGraph
from .agentic_langgraph import AgenticRAGLangGraph
from .unified_rag_langchain import UnifiedRAGLangChain

# For backward compatibility, create aliases
BasicRAG = BasicRAGLangChain
MultiTurnRAG = MultiTurnRAGLangGraph
AgenticRAG = AgenticRAGLangGraph
UnifiedRAG = UnifiedRAGLangChain

LANGCHAIN_AVAILABLE = True

__all__ = [
    # Original names (aliases)
    'BasicRAG',
    'MultiTurnRAG',
    'AgenticRAG',
    'UnifiedRAG',
    # LangChain/LangGraph names
    'BasicRAGLangChain',
    'MultiTurnRAGLangGraph',
    'AgenticRAGLangGraph',
    'UnifiedRAGLangChain',
    'LANGCHAIN_AVAILABLE'
]
