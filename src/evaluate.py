import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import build_inference_prompt, extract_decision

def evaluate_pubmedqa(model, tokenizer, n_samples=500, split="pqa_labeled"):
    ds = load_dataset("qiaojin/PubMedQA", split, split="train")
    test_samples = [s for s in ds if s.get("final_decision") in ["yes", "no", "maybe"]][:n_samples]
    
    results = {"yes": {"correct": 0, "total": 0},
               "no": {"correct": 0, "total": 0},
               "maybe": {"correct": 0, "total": 0}}
    
    correct = 0
    unknowns = 0
    
    for i, sample in enumerate(test_samples):
        abstract = " ".join(sample["context"]["contexts"])
        prompt = build_inference_prompt(abstract, sample["question"])
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        new_tokens = output[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred = extract_decision(response)
        gt = sample["final_decision"]
        
        results[gt]["total"] += 1
        if pred == gt:
            correct += 1
            results[gt]["correct"] += 1
        if pred == "unknown":
            unknowns += 1
        
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_samples}] Running accuracy: {correct/(i+1):.3f}")
    
    accuracy = correct / n_samples
    per_label = {
        label: (v["correct"] / v["total"] if v["total"] > 0 else 0)
        for label, v in results.items()
    }
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": n_samples,
        "unknown_predictions": unknowns,
        "per_label_accuracy": per_label
    }

if __name__ == "__main__":
    # Base model eval by default, modify locally to eval finetuned
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Evaluating PubMedQA (500 samples)...")
    results = evaluate_pubmedqa(model, tokenizer, n_samples=500)
    print(f"\n=== MODEL RESULTS ===")
    print(f"PubMedQA Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"Per-label: {results['per_label_accuracy']}")
    print(f"Unknown predictions: {results['unknown_predictions']}")
    
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/baseline_results.json", "w") as f:
        json.dump({"model": "mistralai/Mistral-7B-Instruct-v0.3", **results}, f, indent=2)
