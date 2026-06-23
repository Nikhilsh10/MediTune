# Backend Schema
## MediTune — Data Models, Config Schemas, Eval Schemas

**Version:** 1.0  
**Author:** Nikhil Sharma

---

## Overview

MediTune has no traditional backend server. "Backend" in this context means:
1. **Dataset schema** — the format of training/eval data on disk
2. **Training config schema** — the YAML structure governing all hyperparameters
3. **Checkpoint metadata schema** — what gets saved at each checkpoint
4. **Eval results schema** — structured output of the evaluation script
5. **Model Card schema** — the structured metadata pushed to HF Hub
6. **W&B run schema** — what gets logged and when

All schemas are validated at runtime using Python dataclasses or Pydantic models. If a schema contract is broken, the script fails early rather than silently training on corrupt data.

---

## 1. Dataset Schema

### 1.1 Raw PubMedQA Sample (from HF Datasets)

```python
# What you get from: dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
{
    "pubid": "18508551",                          # str: PubMed article ID
    "question": "Is there a link between...",      # str: The clinical question
    "context": {
        "contexts": ["Abstract sentence 1...",     # List[str]: Abstract sentences
                     "Abstract sentence 2..."],
        "labels": ["BACKGROUND", "RESULTS"],       # List[str]: Section labels
        "meshes": ["Humans", "Aged"],              # List[str]: MeSH terms
        "reasoning_required_pred": "yes",          # str: Predicted reasoning requirement
        "reasoning_free_pred": "yes"               # str: Predicted decision without reasoning
    },
    "long_answer": "The findings suggest...",      # str: Free-text answer explanation
    "final_decision": "yes"                        # str: "yes" | "no" | "maybe"
}
```

### 1.2 Processed Training Sample (JSONL on disk)

Each sample written to `data/train.jsonl` and `data/eval.jsonl`:

```python
@dataclass
class TrainingSample:
    pubid: str          # Source article ID (for deduplication)
    text: str           # Full formatted instruction string (used by SFTTrainer)
    decision: str       # "yes" | "no" | "maybe" (for stratified split)
    token_count: int    # Pre-computed token length (for filtering by max_seq_length)
```

**JSONL line format:**
```json
{
  "pubid": "18508551",
  "text": "### Instruction:\nYou are a medical expert...\n\n### Context:\n{abstract}\n\n### Question:\n{question}\n\n### Response:\nyes. The findings suggest...",
  "decision": "yes",
  "token_count": 387
}
```

**Validation rules (enforced in `prepare_dataset.py`):**
- `decision` must be in `{"yes", "no", "maybe"}`
- `token_count` must be ≤ `max_seq_length` from config (512)
- `text` must not be empty
- `pubid` must be unique within the split (deduplication)

### 1.3 Label Distribution Check

Before training, assert label balance is not catastrophically skewed:
```python
from collections import Counter

def assert_label_distribution(samples, split_name):
    counts = Counter(s["decision"] for s in samples)
    total = sum(counts.values())
    for label, count in counts.items():
        pct = count / total * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
        assert pct > 5, f"Label '{label}' is <5% in {split_name} split — check data pipeline"

# Expected distribution for pqa_artificial:
# yes: ~55%, no: ~30%, maybe: ~15%
```

---

## 2. Training Config Schema

Pydantic model for `configs/training_config.yaml`. The training script instantiates this and fails immediately if any field is missing or invalid — no silent defaults.

```python
from pydantic import BaseModel, validator
from typing import List, Optional

class ModelConfig(BaseModel):
    base_model_id: str
    max_seq_length: int
    trust_remote_code: bool = False

    @validator("max_seq_length")
    def validate_seq_length(cls, v):
        assert 128 <= v <= 2048, "max_seq_length must be between 128 and 2048"
        return v

class QuantizationConfig(BaseModel):
    load_in_4bit: bool
    bnb_4bit_quant_type: str       # "nf4" or "fp4"
    bnb_4bit_compute_dtype: str    # "bfloat16" or "float16"
    bnb_4bit_use_double_quant: bool

    @validator("bnb_4bit_quant_type")
    def validate_quant_type(cls, v):
        assert v in ["nf4", "fp4"], "quant_type must be 'nf4' or 'fp4'"
        return v

class LoRAConfig(BaseModel):
    r: int
    lora_alpha: int
    target_modules: List[str]
    lora_dropout: float
    bias: str
    task_type: str

    @validator("r")
    def validate_rank(cls, v):
        assert v in [4, 8, 16, 32, 64], "LoRA rank must be a power of 2 between 4 and 64"
        return v

    @validator("bias")
    def validate_bias(cls, v):
        assert v in ["none", "all", "lora_only"]
        return v

class TrainingConfig(BaseModel):
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    weight_decay: float
    optim: str
    lr_scheduler_type: str
    warmup_ratio: float
    max_grad_norm: float
    logging_steps: int
    save_steps: int
    eval_steps: int
    evaluation_strategy: str
    load_best_model_at_end: bool
    metric_for_best_model: str
    fp16: bool
    bf16: bool
    group_by_length: bool
    packing: bool
    report_to: str

    @validator("learning_rate")
    def validate_lr(cls, v):
        assert 1e-6 <= v <= 1e-2, "Learning rate out of safe range for LoRA"
        return v

class DataConfig(BaseModel):
    dataset_name: str
    train_config: str
    eval_config: str
    train_split: str
    eval_split: str
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

class HubConfig(BaseModel):
    push_to_hub: bool
    hub_model_id: str
    hub_strategy: str

class MediTuneConfig(BaseModel):
    model: ModelConfig
    quantization: QuantizationConfig
    lora: LoRAConfig
    training: TrainingConfig
    data: DataConfig
    hub: HubConfig
```

**Loading pattern in `train.py`:**
```python
import yaml
from configs.schema import MediTuneConfig

with open("configs/training_config.yaml") as f:
    raw = yaml.safe_load(f)

config = MediTuneConfig(**raw)  # Fails fast if config is invalid
```

---

## 3. Checkpoint Metadata Schema

Saved as `checkpoint_metadata.json` inside each checkpoint directory alongside the HF model weights:

```json
{
  "checkpoint_id": "checkpoint-200",
  "step": 200,
  "epoch": 0.16,
  "train_loss": 1.847,
  "eval_loss": 1.923,
  "eval_accuracy": 0.614,
  "timestamp_utc": "2026-06-15T14:32:00Z",
  "config_hash": "sha256:abc123...",
  "wandb_run_id": "meditune-abc123",
  "git_commit": "a1b2c3d",
  "hardware": {
    "gpu_count": 2,
    "gpu_name": "Tesla T4",
    "gpu_memory_gb": 16,
    "cuda_version": "12.1"
  },
  "lora_params": {
    "r": 16,
    "alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "trainable_params": 41943040,
    "total_params": 7241748480,
    "trainable_pct": 0.5792
  }
}
```

**Written by:**
```python
def save_checkpoint_metadata(trainer, config, checkpoint_dir):
    metadata = {
        "checkpoint_id": os.path.basename(checkpoint_dir),
        "step": trainer.state.global_step,
        "epoch": round(trainer.state.epoch, 4),
        "train_loss": trainer.state.log_history[-1].get("loss"),
        "eval_loss": trainer.state.log_history[-1].get("eval_loss"),
        "config_hash": hashlib.sha256(
            json.dumps(config.dict(), sort_keys=True).encode()
        ).hexdigest(),
        # ... fill remaining fields
    }
    with open(os.path.join(checkpoint_dir, "checkpoint_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
```

---

## 4. Evaluation Results Schema

Output of `src/evaluate.py`. Written to `outputs/eval_results.json`:

```json
{
  "run_id": "meditune-eval-20260615",
  "model_id": "nikhilsh10/meditune-mistral-7b",
  "eval_timestamp_utc": "2026-06-15T18:00:00Z",
  "pubmedqa": {
    "split": "pqa_labeled",
    "n_samples": 500,
    "base_model": {
      "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
      "accuracy": 0.614,
      "correct": 307,
      "per_label_accuracy": {
        "yes": 0.71,
        "no": 0.58,
        "maybe": 0.41
      }
    },
    "finetuned_model": {
      "model_id": "nikhilsh10/meditune-mistral-7b",
      "accuracy": 0.728,
      "correct": 364,
      "per_label_accuracy": {
        "yes": 0.82,
        "no": 0.71,
        "maybe": 0.56
      }
    },
    "delta_accuracy": 0.114,
    "delta_pct": 18.6
  },
  "rouge_l": {
    "dataset": "MedQuAD",
    "n_samples": 200,
    "base_model_rouge_l": 0.31,
    "finetuned_model_rouge_l": 0.44,
    "delta": 0.13
  },
  "mmlu_sanity": {
    "dataset": "MMLU college_medicine",
    "n_samples": 50,
    "base_model_accuracy": 0.62,
    "finetuned_model_accuracy": 0.59,
    "delta": -0.03,
    "catastrophic_forgetting": false,
    "threshold_used": -0.10
  }
}
```

---

## 5. W&B Logging Schema

What gets logged at each step. Define these explicitly so the W&B dashboard is clean and interpretable.

### Per-step logs (every `logging_steps=25`)
```python
wandb.log({
    "train/loss": loss,
    "train/learning_rate": lr,
    "train/epoch": epoch,
    "train/grad_norm": grad_norm,
    "train/global_step": step,
})
```

### Per-eval logs (every `eval_steps=200`)
```python
wandb.log({
    "eval/loss": eval_loss,
    "eval/accuracy": eval_accuracy,
    "eval/global_step": step,
})
```

### Run-level summary (logged once at end)
```python
wandb.run.summary.update({
    "best_eval_loss": best_eval_loss,
    "best_eval_accuracy": best_eval_accuracy,
    "final_pubmedqa_accuracy_base": base_accuracy,
    "final_pubmedqa_accuracy_finetuned": finetuned_accuracy,
    "accuracy_delta": finetuned_accuracy - base_accuracy,
    "rouge_l_delta": rouge_finetuned - rouge_base,
    "total_training_steps": total_steps,
    "total_train_time_hours": elapsed_hours,
    "trainable_params": trainable_params,
    "total_params": total_params,
})
```

### W&B Config (logged once at run start)
```python
wandb.config.update({
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "lora_r": config.lora.r,
    "lora_alpha": config.lora.lora_alpha,
    "quantization": "NF4 4-bit",
    "max_seq_length": config.model.max_seq_length,
    "learning_rate": config.training.learning_rate,
    "num_train_epochs": config.training.num_train_epochs,
    "train_samples": len(train_dataset),
    "hardware": "2x T4 16GB (Kaggle)",
})
```

---

## 6. HF Model Card Schema

`model_card/README.md` gets pushed to the HF Hub repo root. Use the HF `ModelCard` API:

```python
from huggingface_hub import ModelCard

card_content = """
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
"""
```
