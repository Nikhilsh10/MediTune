---
language: en
license: apache-2.0
base_model: mistralai/Mistral-7B-Instruct-v0.3
tags:
  - medical
  - question-answering
  - qlora
  - peft
  - fine-tuned
datasets:
  - qiaojin/PubMedQA
metrics:
  - accuracy
  - rouge
pipeline_tag: text-generation
---

# MediTune — Mistral-7B Fine-tuned for Medical QA

## Model Description
QLoRA fine-tune of Mistral-7B-Instruct-v0.3 on PubMedQA for closed-domain medical 
question answering (yes/no/maybe classification with explanation).

## Training Details
- **PEFT Method:** QLoRA (NF4 4-bit quantization)
- **LoRA Rank:** r=16, α=32
- **Trainable Parameters:** 41M / 7.24B (0.57%)
- **Hardware:** 2×T4 16GB (Kaggle free tier)
- **Training Data:** PubMedQA pqa_artificial (50k samples, 2 epochs)

## Evaluation Results

| Dataset | Metric | Base Model | MediTune | Delta |
|---------|--------|-----------|----------|-------|
| PubMedQA (500) | Accuracy | 61.4% | 72.8% | +11.4% |
| MedQuAD (200) | ROUGE-L | 0.31 | 0.44 | +41.9% |
| MMLU Medicine (50) | Accuracy | 62% | 59% | -3% |

## ⚠️ Important Limitations
This model is a PORTFOLIO PROJECT only. It must NOT be used for real clinical decisions.
Medical decisions require qualified healthcare professionals.

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    load_in_4bit=True
)
model = PeftModel.from_pretrained(base, "nikhilsh10/meditune-mistral-7b")
tokenizer = AutoTokenizer.from_pretrained("nikhilsh10/meditune-mistral-7b")
```
