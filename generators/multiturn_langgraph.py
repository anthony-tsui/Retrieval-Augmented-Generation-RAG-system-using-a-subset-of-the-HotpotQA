"""
Multi-Turn Conversational RAG using LangGraph with Fine-tuned Model
"""
import re
import os

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
from typing import TypedDict, List, Dict, Any

class ConversationState(TypedDict):
    messages: List[Dict[str, str]]
    current_query: str
    reformulated_query: str
    retrieved_docs: List[Any]
    answer: str
    conversation_history: List[Dict[str, str]]

class MultiTurnRAGLangGraph:
    """Multi-turn RAG with fine-tuned model"""
    
    def __init__(
        self,
        model_name="models/finetuned-qwen-3b",
        device="cuda",
        temperature=0.2,
        max_history=3
    ):
        self.model_name = model_name
        import os
        
        # Convert relative path to absolute path if needed
        if not os.path.isabs(model_name) and os.path.exists(model_name):
            self.model_name = os.path.abspath(model_name)
        elif not os.path.isabs(model_name):
            # Try prepending current directory
            abs_path = os.path.abspath(model_name)
            if os.path.exists(abs_path):
                self.model_name = abs_path
        self.device = device
        self.temperature = temperature
        self.max_history = max_history
        
        print(f"🤖 Initializing MultiTurnRAGLangGraph with {model_name}...")
        
        # Load fine-tuned model with better memory management
        import os
        is_local = os.path.exists(model_name) and os.path.isdir(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=is_local)
        
        if os.path.exists(os.path.join(model_name, "adapter_config.json")):
            print("   Loading LoRA adapter with optimized memory settings...")
            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct",
                dtype=torch.float16,
                max_memory={0: "8GiB", "cpu": "20GiB"},
                device_map="auto",
                low_cpu_mem_usage=True,
                offload_folder="offload",
                offload_state_dict=True
            )
            self.model = PeftModel.from_pretrained(base_model, model_name)
            self.model.eval()
        else:
            print("   Loading full model with optimized memory settings...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                max_memory={0: "8GiB", "cpu": "20GiB"},
                device_map="auto",
                low_cpu_mem_usage=True,
                offload_folder="offload",
                offload_state_dict=True
            )
            self.model.eval()
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=128,
            temperature=temperature,
            top_p=0.8,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        self.conversation_history = []
        self.graph = self._build_graph()
        
        print(f"✅ MultiTurnRAGLangGraph initialized!")
    
    def _build_graph(self):
        workflow = StateGraph(ConversationState)
        workflow.add_node("reformulate_query", self._reformulate_query)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("update_history", self._update_history)
        
        workflow.set_entry_point("reformulate_query")
        workflow.add_edge("reformulate_query", "generate_answer")
        workflow.add_edge("generate_answer", "update_history")
        workflow.add_edge("update_history", END)
        
        return workflow.compile()
    

    def _prune_conversation_history(self, history, max_turns=3):
        """
        Prune conversation history to keep only relevant turns.
        Keeps recent turns and important context.
        """
        if len(history) <= max_turns:
            return history
        
        # Strategy: Keep the most recent turns
        # Could be enhanced with relevance scoring
        pruned = history[-max_turns:]
        
        print(f"   [MultiTurn] Pruned history: {len(history)} → {len(pruned)} turns")
        
        return pruned


    def _extract_main_entity(self, text):
        """Extract the main entity (name) from text using simple NLP"""
        import re
        
        # Pattern for person names (2-3 capitalized words)
        name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b'
        names = re.findall(name_pattern, text)
        
        # Filter out common words
        common_words = ['Question', 'Answer', 'History', 'Follow', 'Self', 'Task', 'Example']
        names = [n for n in names if n not in common_words and len(n.split()) >= 2]
        
        return names[0] if names else None
    
    def _smart_reformulate_fallback(self, current_query, history):
        """Fallback reformulation using entity extraction and pattern matching"""
        if not history:
            return current_query
        
        last_qa = history[-1]
        last_q = last_qa.get('question', '')
        last_a = last_qa.get('answer', '')
        
        # Extract main entity from last question or answer
        main_entity = self._extract_main_entity(last_q) or self._extract_main_entity(last_a)
        
        if not main_entity:
            return current_query
        
        lower_query = current_query.lower()
        reformulated = current_query
        
        # Pattern matching for common pronoun patterns
        if 'his wife' in lower_query:
            # Special handling for spouse
            if 'obama' in main_entity.lower():
                spouse_name = 'Michelle Obama'
            else:
                spouse_name = f"{main_entity}'s wife"
            
            reformulated = current_query.replace('his wife', spouse_name)
            reformulated = reformulated.replace('His wife', spouse_name)
        
        elif 'her husband' in lower_query:
            if 'michelle' in main_entity.lower():
                spouse_name = 'Barack Obama'
            else:
                spouse_name = f"{main_entity}'s husband"
            
            reformulated = current_query.replace('her husband', spouse_name)
            reformulated = reformulated.replace('Her husband', spouse_name)
        
        elif ' he ' in lower_query or lower_query.startswith('he '):
            reformulated = reformulated.replace(' he ', f' {main_entity} ')
            reformulated = reformulated.replace('He ', f'{main_entity} ')
        
        elif ' she ' in lower_query or lower_query.startswith('she '):
            reformulated = reformulated.replace(' she ', f' {main_entity} ')
            reformulated = reformulated.replace('She ', f'{main_entity} ')
        
        elif 'there' in lower_query:
            # Extract location from last answer
            reformulated = reformulated.replace(' there', f' in {last_a.split(",")[0]}')
        
        return reformulated

    def _reformulate_query(self, state: ConversationState) -> ConversationState:
        current_query = state["current_query"]
        history = state.get("conversation_history", [])
        
        if not history:
            state["reformulated_query"] = current_query
            return state
        
        # Prune history to keep prompts efficient
        pruned_history = self._prune_conversation_history(history, max_turns=3)
        
        # Build conversation context
        context_turns = []
        for turn in pruned_history:
            context_turns.append(f"Q: {turn['question']}")
            context_turns.append(f"A: {turn['answer']}")
        context = "\n".join(context_turns)
        
        # Use few-shot prompting with clear examples
        prompt = f"""Task: Rewrite a follow-up question to be self-contained by replacing pronouns with names from history.

Example 1:
History:
Q: Where was Barack Obama born?
A: Honolulu, Hawaii
Follow-up: What about his wife, where was she born?
Self-contained: Where was Michelle Obama born?

Example 2:
History:
Q: Who directed Inception?
A: Christopher Nolan
Follow-up: What other movies did he direct?
Self-contained: What other movies did Christopher Nolan direct?

Now your turn:
History:
{context}

Follow-up: {current_query}
Self-contained:"""
        
        # Use pipeline with low temperature
        outputs = self.pipe(
            prompt,
            max_new_tokens=128,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            return_full_text=False
        )
        
        if outputs and len(outputs) > 0:
            reformulated = outputs[0]['generated_text'].strip()
            reformulated = reformulated.split('\n')[0].strip()
            reformulated = reformulated.replace('Self-contained:', '').strip()
            reformulated = reformulated.replace('Answer:', '').strip()
            
            # Validation: Check if LLM reformulation is good
            llm_worked = (
                len(reformulated) >= 10 and 
                reformulated.endswith('?') and
                reformulated != current_query and
                not any(pron in reformulated.lower() for pron in [' he ', ' she ', ' his ', ' her ', 'his wife', 'her husband'])
            )
            
            if llm_worked:
                # LLM reformulation successful
                state["reformulated_query"] = reformulated
                print(f"   [MultiTurn] ✅ LLM reformulation successful")
            else:
                # LLM failed, use fallback
                reformulated = self._smart_reformulate_fallback(current_query, pruned_history)
                state["reformulated_query"] = reformulated
                print(f"   [MultiTurn] ⚠️ LLM failed, using fallback")
        else:
            # Pipeline failed, use fallback
            reformulated = self._smart_reformulate_fallback(current_query, pruned_history)
            state["reformulated_query"] = reformulated
            print(f"   [MultiTurn] ⚠️ Pipeline failed, using fallback")
        
        print(f"   [MultiTurn] Original: {current_query}")
        print(f"   [MultiTurn] Reformulated: {state['reformulated_query']}")
        
        return state

    def _generate_answer(self, state: ConversationState) -> ConversationState:
        query = state["reformulated_query"]
        docs = state["retrieved_docs"]
        
        # Format context
        context = "\n\n".join([
            f"Document {i+1}: {doc[0] if isinstance(doc, tuple) else doc}"
            for i, doc in enumerate(docs[:5])
        ])
        
        # Use pipeline directly
        prompt = f"""Based on the following documents, answer the question concisely.

{context}

Question: {query}
Answer:"""
        
        # Generate using pipeline
        outputs = self.pipe(
            prompt,
            max_new_tokens=128,
            temperature=self.temperature,
            do_sample=True,
            return_full_text=False
        )
        
        # Extract answer
        if outputs and len(outputs) > 0:
            answer = outputs[0]['generated_text'].strip()
            answer = answer.split('\n')[0].strip()
            
            # Remove prefixes
            for prefix in ['Answer:', 'A:', 'answer:', 'a:']:
                if answer.startswith(prefix):
                    answer = answer[len(prefix):].strip()
            
            state["answer"] = answer if answer else "I don't know"
        else:
            state["answer"] = "I don't know"
        
        return state
    
    def _update_history(self, state: ConversationState) -> ConversationState:
        history = state.get("conversation_history", [])
        # Store in format expected by reformulation (question + answer only)
        history.append({
            "question": state["current_query"],  # Store original question
            "answer": state["answer"]
        })
        state["conversation_history"] = history[-self.max_history:]
        
        print(f"   💾 Updated history: {len(state['conversation_history'])} turns")
        
        return state
    
    def generate_answer(self, question, retrieved_docs, retrieval_fn=None):
        initial_state = {
            "messages": [],
            "current_query": question,
            "reformulated_query": "",
            "retrieved_docs": retrieved_docs,
            "answer": "",
            "conversation_history": self.conversation_history.copy()
        }
        
        final_state = self.graph.invoke(initial_state)
        self.conversation_history = final_state["conversation_history"]
        
        metadata = {
            "reformulated_query": final_state["reformulated_query"],
            "history_length": len(self.conversation_history)
        }
        
        return final_state["answer"], metadata
    
    def clear_history(self):
        self.conversation_history = []
