"""
Answer Validator Module

This module uses training data patterns to validate and improve generated answers.

Key Features:
- Validates answer format consistency
- Checks answer length appropriateness
- Compares with similar training examples
- Filters out hallucinated or off-topic answers
"""

import json
import re
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np


class AnswerValidator:
    """
    Validates and refines generated answers based on training data patterns.
    """
    
    def __init__(self, train_path: str = "data/train.jsonl"):
        """
        Initialize validator with training data statistics.
        
        Args:
            train_path: Path to training JSONL file
        """
        self.train_data = self._load_train_data(train_path)
        self._build_statistics()
        print(f"✓ Answer validator initialized with {len(self.train_data)} training examples")
    
    def _load_train_data(self, train_path: str) -> List[Dict]:
        """Load training data from JSONL file."""
        train_data = []
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                train_data.append(data)
        return train_data
    
    def _build_statistics(self):
        """Build statistics from training answers."""
        # Answer length distribution
        self.answer_lengths = [len(item['answer'].split()) for item in self.train_data]
        self.avg_answer_length = np.mean(self.answer_lengths)
        self.median_answer_length = np.median(self.answer_lengths)
        self.max_reasonable_length = np.percentile(self.answer_lengths, 95)
        
        # Common answer patterns
        self.answer_patterns = {
            'yes_no': sum(1 for item in self.train_data if item['answer'].lower() in ['yes', 'no']),
            'single_word': sum(1 for item in self.train_data if len(item['answer'].split()) == 1),
            'short_phrase': sum(1 for item in self.train_data if 2 <= len(item['answer'].split()) <= 5),
            'long_answer': sum(1 for item in self.train_data if len(item['answer'].split()) > 5)
        }
        
        print(f"  - Average answer length: {self.avg_answer_length:.1f} words")
        print(f"  - Median answer length: {self.median_answer_length:.0f} words")
        print(f"  - Answer patterns: {self.answer_patterns}")
    
    def validate_answer(self, answer: str, question: str, retrieved_docs: List[str]) -> Tuple[bool, str, str]:
        """
        Validate generated answer quality.
        
        Args:
            answer: Generated answer
            question: Original question
            retrieved_docs: Retrieved document texts
            
        Returns:
            (is_valid, confidence_level, explanation)
        """
        issues = []
        score = 100
        
        # Check 1: Answer is not empty
        if not answer or answer.strip() == "":
            return False, "invalid", "Empty answer"
        
        # Check 2: Answer length is reasonable
        answer_words = answer.split()
        if len(answer_words) > self.max_reasonable_length:
            issues.append("Answer too long")
            score -= 30
        
        # Check 3: Answer doesn't contain common error phrases
        error_phrases = [
            "i cannot answer",
            "i don't know",
            "not mentioned",
            "no information",
            "cannot be found",
            "based on the provided information"
        ]
        if any(phrase in answer.lower() for phrase in error_phrases):
            issues.append("Contains uncertainty phrase")
            score -= 20
        
        # Check 4: Answer contains content from retrieved docs (grounding check)
        answer_lower = answer.lower()
        doc_text = " ".join(retrieved_docs).lower()
        
        # Check if answer words appear in documents
        answer_words_in_docs = sum(
            1 for word in answer_words 
            if len(word) > 3 and word.lower() in doc_text
        )
        grounding_ratio = answer_words_in_docs / max(len(answer_words), 1)
        
        if grounding_ratio < 0.3:
            issues.append(f"Low grounding (only {grounding_ratio:.0%} of answer found in docs)")
            score -= 40
        
        # Check 5: Answer doesn't repeat the question
        question_words = set(question.lower().split())
        answer_words_set = set(answer.lower().split())
        overlap = len(question_words & answer_words_set) / max(len(answer_words_set), 1)
        
        if overlap > 0.7:
            issues.append("Answer repeats question")
            score -= 25
        
        # Determine confidence level
        if score >= 80:
            confidence = "high"
        elif score >= 60:
            confidence = "medium"
        else:
            confidence = "low"
        
        is_valid = score >= 50
        explanation = "; ".join(issues) if issues else "Answer looks good"
        
        return is_valid, confidence, explanation
    
    def refine_answer(self, answer: str) -> str:
        """
        Refine and clean up the generated answer.
        
        Args:
            answer: Raw generated answer
            
        Returns:
            Refined answer
        """
        # Remove common prefixes
        prefixes_to_remove = [
            "answer:",
            "the answer is:",
            "based on the documents,",
            "according to the information,",
            "from the documents,",
        ]
        
        answer_lower = answer.lower()
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
                break
        
        # Remove trailing punctuation artifacts
        answer = answer.rstrip('.!?;,')
        
        # Capitalize first letter if it's a proper answer
        if answer and not answer[0].isupper() and len(answer.split()) > 1:
            answer = answer[0].upper() + answer[1:]
        
        return answer
    
    def get_answer_type(self, question: str) -> str:
        """
        Determine expected answer type from question.
        
        Args:
            question: User question
            
        Returns:
            Expected answer type
        """
        question_lower = question.lower()
        
        if any(q in question_lower for q in ['what year', 'when', 'what date']):
            return 'date'
        elif any(q in question_lower for q in ['where', 'what city', 'what country', 'what location']):
            return 'location'
        elif any(q in question_lower for q in ['who', 'what person']):
            return 'person'
        elif any(q in question_lower for q in ['how many', 'how much']):
            return 'number'
        elif any(q in question_lower for q in ['is ', 'are ', 'was ', 'were ', 'did ', 'does ']):
            return 'yes_no'
        else:
            return 'general'
    
    def suggest_improvements(self, answer: str, question: str, answer_type: str) -> str:
        """
        Suggest improvements based on answer type.
        
        Args:
            answer: Generated answer
            question: Original question
            answer_type: Detected answer type
            
        Returns:
            Improvement suggestions
        """
        suggestions = []
        
        if answer_type == 'yes_no':
            if answer.lower() not in ['yes', 'no']:
                suggestions.append("Consider simplifying to 'yes' or 'no'")
        
        if answer_type == 'date' and not re.search(r'\d{4}', answer):
            suggestions.append("Answer should contain a year")
        
        if answer_type == 'number' and not any(c.isdigit() for c in answer):
            suggestions.append("Answer should contain a number")
        
        if len(answer.split()) > 20:
            suggestions.append("Consider making answer more concise")
        
        return "; ".join(suggestions) if suggestions else "No suggestions"


if __name__ == "__main__":
    # Test the validator
    print("Testing Answer Validator...")
    
    validator = AnswerValidator(train_path="data/train.jsonl")
    
    # Test cases
    test_cases = [
        {
            "answer": "Knox County Regional Airport",
            "question": "Which airport is located in Maine?",
            "docs": ["Knox County Regional Airport is located in Maine.", "Sacramento is in California."]
        },
        {
            "answer": "I cannot answer based on the provided information",
            "question": "What is the capital?",
            "docs": ["Some random text here"]
        },
        {
            "answer": "yes",
            "question": "Were both authors?",
            "docs": ["Both were famous authors in their time."]
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n Test {i}:")
        is_valid, confidence, explanation = validator.validate_answer(
            test['answer'], test['question'], test['docs']
        )
        print(f"  Answer: {test['answer']}")
        print(f"  Valid: {is_valid}, Confidence: {confidence}")
        print(f"  Explanation: {explanation}")
