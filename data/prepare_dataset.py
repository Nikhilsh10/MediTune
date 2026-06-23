import json
import hashlib
from datasets import load_dataset
from transformers import AutoTokenizer

# Need to import locally or relatively depending on execution context, but usually run from root
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import build_instruction_prompt

def is_valid(sample):
    return (
        sample.get("final_decision") in ["yes", "no", "maybe"]
        and sample.get("long_answer") is not None
        and len(sample.get("long_answer", "")) > 10
        and sample.get("context", {}).get("contexts") is not None
        and len(sample["context"]["contexts"]) > 0
    )

def prepare_and_save(
    output_train="data/train.jsonl",
    output_eval="data/eval.jsonl",
    max_train=50000,
    max_seq_length=512,
    model_id="mistralai/Mistral-7B-Instruct-v0.3"
):
    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    os.makedirs(os.path.dirname(output_eval), exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Training data: pqa_artificial
    print("Loading training data...")
    ds_train = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    ds_train = ds_train.filter(is_valid)
    if max_train:
        ds_train = ds_train.select(range(min(max_train, len(ds_train))))
    
    # Eval data: pqa_labeled
    print("Loading eval data...")
    ds_eval = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    ds_eval = ds_eval.filter(is_valid)
    # Use first 500 as test/eval
    eval_samples = ds_eval.select(range(min(500, len(ds_eval))))
    
    seen_ids = set()
    
    def write_split(dataset, output_path):
        written = 0
        skipped_dup = 0
        skipped_long = 0
        
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in dataset:
                pubid = sample.get("pubid", "")
                
                # Deduplication
                if pubid in seen_ids:
                    skipped_dup += 1
                    continue
                seen_ids.add(pubid)
                
                # Format
                text = build_instruction_prompt(sample)
                token_count = len(tokenizer.encode(text))
                
                # Filter by length
                if token_count > max_seq_length:
                    skipped_long += 1
                    continue
                
                record = {
                    "pubid": pubid,
                    "text": text,
                    "decision": sample["final_decision"],
                    "token_count": token_count
                }
                f.write(json.dumps(record) + "\n")
                written += 1
        
        print(f"  Written: {written}, Skipped (dup): {skipped_dup}, Skipped (long): {skipped_long}")
        return written
    
    print("\nProcessing training split...")
    train_count = write_split(ds_train, output_train)
    
    print("\nProcessing eval split...")
    eval_count = write_split(eval_samples, output_eval)
    
    print(f"\nDone. Train: {train_count}, Eval: {eval_count}")

if __name__ == "__main__":
    prepare_and_save()
