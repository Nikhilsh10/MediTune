import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import build_inference_prompt

def load_inference_model(base_model_id="mistralai/Mistral-7B-Instruct-v0.3", adapter_path="outputs/meditune-checkpoint"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading base model {base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def run_inference(model, tokenizer, context, question, max_new_tokens=100):
    prompt = build_inference_prompt(context, question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    new_tokens = output[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response

if __name__ == "__main__":
    # Example usage
    context = "This study examined the effect of aspirin on colorectal cancer prevention in high-risk patients. Patients received 300mg aspirin daily for 2 years. The primary endpoint was colorectal adenoma recurrence. Results showed a 47% reduction in adenoma recurrence (p<0.001) in the aspirin group compared to placebo."
    question = "Does aspirin significantly reduce colorectal adenoma recurrence in high-risk patients?"
    
    model, tokenizer = load_inference_model()
    response = run_inference(model, tokenizer, context, question)
    print("\n--- Model Response ---")
    print(response)
