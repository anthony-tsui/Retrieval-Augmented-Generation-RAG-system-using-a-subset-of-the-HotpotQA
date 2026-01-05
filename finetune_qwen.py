"""
Fine-tune Qwen2.5-1.5B-Instruct on HotpotQA train.jsonl with supporting documents.

This script fine-tunes the model using LoRA (Parameter-Efficient Fine-Tuning)
by linking train.jsonl questions/answers with supporting documents from collection.jsonl.

Usage:
    # Install required packages first
    pip install peft bitsandbytes accelerate
    
    # Run fine-tuning
    python finetune_qwen.py --epochs 3 --batch_size 2 --device cuda
    
    # Use fine-tuned model
    python evaluate.py --model_name models/finetuned-qwen-3b --device cuda
"""

import json
import os
import argparse
from typing import List, Dict
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm


class HotpotQADataset(Dataset):
    """Dataset linking train.jsonl questions with supporting docs from collection."""
    
    def __init__(self, train_file: str, collection_file: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print("Loading collection...")
        self.collection = self._load_collection(collection_file)
        print(f"✅ Loaded {len(self.collection)} documents")
        
        print("Loading training data...")
        self.examples = self._load_train_data(train_file)
        print(f"✅ Loaded {len(self.examples)} training examples")
    
    def _load_collection(self, collection_file: str) -> Dict[str, str]:
        """Load collection.jsonl into {id: text} dictionary."""
        collection = {}
        with open(collection_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                collection[doc['id']] = doc['text']
        return collection
    
    def _load_train_data(self, train_file: str) -> List[Dict]:
        """Load train.jsonl and link supporting_ids to document texts."""
        examples = []
        skipped = 0
        
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing"):
                item = json.loads(line)
                
                # Get supporting documents using supporting_ids
                supporting_docs = []
                for doc_id in item.get('supporting_ids', []):
                    if doc_id in self.collection:
                        supporting_docs.append(self.collection[doc_id])
                
                if not supporting_docs:
                    skipped += 1
                    continue
                
                examples.append({
                    'question': item['text'],
                    'answer': item['answer'],
                    'supporting_docs': supporting_docs
                })
        
        if skipped > 0:
            print(f"⚠️  Skipped {skipped} examples without supporting docs")
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format supporting documents
        docs_text = "\n\n".join([
            f"[Document {i+1}]\n{doc}"
            for i, doc in enumerate(example['supporting_docs'])
        ])
        
        # Create chat messages
        messages = [
            {"role": "system", "content": "Answer questions based on provided documents."},
            {"role": "user", "content": f"Documents:\n{docs_text}\n\nQuestion: {example['question']}"},
            {"role": "assistant", "content": f"Answer: {example['answer']}"}
        ]
        
        # Apply chat template and tokenize
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encodings = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length")
        
        return {
            'input_ids': torch.tensor(encodings['input_ids']),
            'attention_mask': torch.tensor(encodings['attention_mask']),
            'labels': torch.tensor(encodings['input_ids'])
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/train.jsonl")
    parser.add_argument("--collection_file", default="data/collection.jsonl")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", default="models/finetuned-qwen-3b")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 FINE-TUNING QWEN2.5-3B WITH LORA ON HOTPOTQA")
    print("=" * 80)
    
    # Check GPU
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available! Using CPU (very slow)")
            args.device = "cpu"
        else:
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load tokenizer
    print(f"\n📦 Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 8-bit quantization (optimized for 8GB GPU)
    print("📦 Loading model with 8-bit quantization...")
    from transformers import BitsAndBytesConfig
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Prepare for LoRA training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Create dataset
    print("\n📊 Creating dataset...")
    dataset = HotpotQADataset(args.train_file, args.collection_file, tokenizer, args.max_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Train!
    print(f"\n🏋️ Training for {args.epochs} epochs...")
    trainer.train()
    
    # Save
    print(f"\n💾 Saving to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("\n✅ TRAINING COMPLETE!")
    print(f"\nTo use: model_name='{args.output_dir}'")


if __name__ == "__main__":
    main()
