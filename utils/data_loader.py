"""
Data Loader Utilities

This module provides utility functions for loading HotpotQA dataset files
and extracting relevant information from the collection.

Functions:
    - load_collection: Load document collection from JSONL file
    - load_dataset: Load train/validation/test splits
    - get_texts_and_ids: Extract texts and IDs from collection
"""

import json
from typing import List, Dict, Tuple


def load_collection(path: str) -> List[Dict]:
    """
    Load document collection from a JSONL file.
    
    Args:
        path: Path to collection.jsonl file
        
    Returns:
        List of document dictionaries with 'id' and 'text' keys
    """
    collection = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            collection.append(json.loads(line.strip()))
    return collection


def load_dataset(path: str) -> List[Dict]:
    """
    Load train/validation/test split from JSONL file.
    
    Args:
        path: Path to dataset JSONL file (train.jsonl, validation.jsonl, test.jsonl)
        
    Returns:
        List of query dictionaries containing:
            - id: Query ID
            - text: Query text
            - answer: Answer text (if available)
            - supporting_ids: List of supporting document IDs (if available)
    """
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line.strip()))
    return dataset


def get_texts_and_ids(collection: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    Extract document texts and IDs from collection.
    
    Args:
        collection: List of document dictionaries
        
    Returns:
        Tuple of (texts, ids) where texts is a list of document texts
        and ids is a list of corresponding document IDs
    """
    texts = [doc['text'] for doc in collection]
    ids = [doc['id'] for doc in collection]
    return texts, ids


def save_predictions(predictions: List[Dict], output_path: str):
    """
    Save predictions to JSONL file in the required format.
    
    Args:
        predictions: List of prediction dictionaries containing:
            - id: Query ID
            - question: Query text
            - answer: Generated answer
            - retrieved_docs: List of [doc_id, score] pairs (top 10)
        output_path: Path to output JSONL file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    print(f"Predictions saved to {output_path}")
