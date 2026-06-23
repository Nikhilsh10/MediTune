# Technical Requirements Document (TRD)
## MediTune — QLoRA Fine-Tuning Pipeline for Medical Question Answering

**Version:** 1.0  
**Author:** Nikhil Sharma  
**Status:** Draft  

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MEDITUNE PIPELINE                           │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  DATA LAYER  │───▶│ TRAIN LAYER  │───▶│   EVAL + PUBLISH     │  │
│  │              │    │              │    │                      │  │
│  │  PubMedQA    │    │  QLoRA on    │    │  PubMedQA test eval  │  │
│  │  MedQuAD     │    │  Mistral-7B  │    │  ROUGE-L eval        │  │
│  │              │    │  2×T4 GPU    │    │  W&B logging         │  │
│  │  JSONL       │    │              │    │  HF Hub push         │  │
│  │  formatting  │    │  W&B stream  │    │  Gradio demo         │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                                     │
│  Runtime: Kaggle Notebook (Python 3.11)                            │
│  GPU: 2×T4 16GB (DataParallel)                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tech Stack

| Layer | Technology | Version | Justification |
|-------|-----------|---------|--------------|
| Base Model | Mistral-7B-Instruct-v0.3 | v0.3 | Apache 2.0; best open instruction model at 7B; HF Hub native |
| PEFT Library | `peft` | ≥0.11.0 | Official LoRA/QLoRA implementation; TRL integration |
| Training Framework | `trl` (SFTTrainer) | ≥0.9.0 | Abstracts the SFT training loop; packing support |
| Quantization | `bitsandbytes` | ≥0.43.0 | NF4 4-bit quantization; CUDA kernel support on T4 |
| Tokenizer/Model Load | `transformers` | ≥4.41.0 | Required for Mistral-v0.3 tokenizer |
| Experiment Tracking | `wandb` | ≥0.17.0 | Best-in-class ML experiment tracking; public run sharing |
| Evaluation | `evaluate` + `rouge_score` | latest | ROUGE-L computation |
| Data | `datasets` (HF) | ≥2.19.0 | PubMedQA streaming load |
| Model Publishing | `huggingface_hub` | ≥0.23.0 | Hub push with Model Card |
| Demo | Gradio | ≥4.36.0 | HF Spaces native; no server management |
| Environment | Kaggle Kernels | Python 3.11 | Free 2×T4; no billing |

---

## 3. Model Architecture: Why QLoRA

### Full Fine-Tuning vs QLoRA — Memory Comparison

| Method | VRAM Needed (7B model) | Available (2×T4) | Feasible? |
|--------|----------------------|------------------|----------|
| Full FT (bfloat16) | ~112 GB | 32 GB | ❌ |
| Full FT (float32) | ~224 GB | 32 GB | ❌ |
| LoRA (bfloat16, no quant) | ~28 GB | 32 GB | Barely |
| QLoRA (4-bit NF4) | ~10–12 GB | 32 GB | ✅ Comfortable |

### QLoRA Mechanics (What You Need to Explain in Interviews)

QLoRA (Dettmers et al., 2023) combines:
1. **NF4 quantization**: Quantize base model weights to 4-bit NormalFloat, which is information-theoretically optimal for normally distributed weights. This freezes the base model.
2. **Double quantization**: Quantize the quantization constants themselves (saves ~0.37 bits/parameter extra).
3. **LoRA adapters**: Add low-rank decomposition matrices `A` and `B` at selected layers. Only A and B are trained (in bfloat16). For a weight matrix `W`, the update is `W + α/r * BA`.
4. **Paged optimizers**: Use NVIDIA unified memory to page optimizer states to CPU RAM when GPU memory spikes, preventing OOM during backprop.

### LoRA Target Modules for Mistral-7B

Apply LoRA to these projection matrices only (not all layers — that's a common mistake):
```python
target_modules = [
    "q_proj",   # Query projection
    "k_proj",   # Key projection
    "v_proj",   # Value projection
    "o_proj",   # Output projection
    "gate_proj",  # MLP gate
    "up_proj",    # MLP up
    "down_proj",  # MLP down
]
```
Applying to ALL 7 projection types gives better domain adaptation than just `q_proj`/`v_proj`.

---

## 4. LoRA Hyperparameter Decisions

| Hyperparameter | Value | Reasoning |
|---------------|-------|-----------|
| `r` (rank) | 16 | r=8 underfits on medical vocabulary; r=32 risks overfitting on 211k samples and doubles adapter size |
| `lora_alpha` | 32 | α = 2r; standard scaling. Effective learning rate for the adapter = lr × (α/r) = lr × 2 |
| `lora_dropout` | 0.05 | Low dropout; PubMedQA is large enough that regularization via early stopping is better |
| `bias` | "none" | Don't train bias terms; increases adapter parameter count for marginal gain |
| Trainable params | ~41M of 7.24B (0.57%) | Only the LoRA A and B matrices are in bfloat16 and trained |

---

## 5. Known Failure Points & Exact Mitigations

This section is the most important part of the TRD. Read it before running a single cell.

### F1 — CUDA Out of Memory (Most Common)

**Symptom:** `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X GB`

**Root cause:** Gradient accumulation buffer + activation checkpoints + optimizer state all reside on GPU simultaneously.

**Mitigation Stack (apply in order):**
```python
# In BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # ← critical for saving memory
)

# In TrainingArguments
training_args = TrainingArguments(
    per_device_train_batch_size=1,         # ← never go above 2 on T4
    gradient_accumulation_steps=4,         # ← effective batch = 4
    gradient_checkpointing=True,           # ← trades compute for memory
    optim="paged_adamw_32bit",             # ← paged optimizer, prevents spike OOM
    fp16=False,                            # ← DON'T use fp16 with bfloat16 model
    bf16=True,                             # ← use bf16 compute
)

# In SFTTrainer
trainer = SFTTrainer(
    max_seq_length=512,      # ← not 2048; T4 can't handle long sequences at batch>1
    dataset_text_field="text",
    packing=True,            # ← pack short samples to fill context; improves GPU utilisation
)
```

**If still OOM:** Reduce `max_seq_length` to 384. PubMedQA abstracts average ~300 tokens; 384 is sufficient.

---

### F2 — Tokenizer Chat Template Error

**Symptom:** `ValueError: Tokenizer does not have a pad_token` or model generates endlessly without stopping.

**Root cause:** Mistral-7B-Instruct uses `<unk>` as pad by default; the tokenizer doesn't set `pad_token_id` explicitly.

**Fix:**
```python
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token       # ← Set pad = eos
tokenizer.padding_side = "right"                # ← Right padding for SFT
```

**Warning:** Setting `pad_token = eos_token` can cause the model to predict padding in some edge cases. It is acceptable for training; for inference, generate with `pad_token_id=tokenizer.eos_token_id`.

---

### F3 — Gradient Overflow / NaN Loss

**Symptom:** Loss becomes `nan` after a few hundred steps.

**Root cause:** bfloat16 has a narrower dynamic range than float32. Gradient norms explode and saturate.

**Fix:**
```python
TrainingArguments(
    max_grad_norm=0.3,      # ← Clip gradients hard at 0.3; default is 1.0
    warmup_ratio=0.03,      # ← Warm up LR for first 3% of steps
    lr_scheduler_type="cosine",
    learning_rate=2e-4,     # ← Standard for LoRA; if NaN persists, drop to 1e-4
)
```

If NaN appears after warmup: your learning rate is still too high. Drop by 2× and restart from step 0 (Kaggle checkpointing handles this via `save_steps`).

---

### F4 — W&B API Key in Kaggle

**Symptom:** `wandb.errors.UsageError: api_key not configured`

**Fix:**
1. In Kaggle Notebook → Add-ons → Secrets → Add secret: `WANDB_API_KEY` = your key
2. In your notebook cell:
```python
import os
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
import wandb
wandb.login()
```

---

### F5 — HF Hub Push Fails (File Too Large)

**Symptom:** `huggingface_hub.utils._errors.HfHubHTTPError: 413 Request Entity Too Large`

**Root cause:** The merged model (base + adapter) is ~14GB. The default upload splits into shards but requires your HF account to have LFS enabled.

**Fix — push adapter only first, then optionally push merged:**
```python
# Option A: Push only the LoRA adapter (~300MB) — always works
model.save_pretrained("./meditune-adapter", save_adapter=True, save_config=True)
tokenizer.save_pretrained("./meditune-adapter")
api = HfApi()
api.upload_folder(
    folder_path="./meditune-adapter",
    repo_id="nikhilsh10/meditune-mistral-7b-adapter",
    repo_type="model",
)

# Option B: Merge and push full model — only if you have sufficient storage
model = model.merge_and_unload()
model.push_to_hub("nikhilsh10/meditune-mistral-7b")
```

---

### F6 — `packing=True` Causes Label Leakage

**Symptom:** Suspiciously low training loss from step 1 (~0.3); model generates the next training sample's text.

**Root cause:** When packing, if you don't set `dataset_num_proc` and proper attention masks, the model attends across sample boundaries.

**Fix:**
```python
trainer = SFTTrainer(
    packing=True,
    dataset_kwargs={"skip_prepare_dataset": False},  # let SFTTrainer handle masking
)
```
SFTTrainer ≥0.9.0 handles cross-sample attention masking correctly with packing. If your `trl` version is older, set `packing=False` and accept ~20% lower GPU utilization.

---

### F7 — Catastrophic Forgetting

**Symptom:** Fine-tuned model achieves high PubMedQA accuracy but cannot answer basic factual questions.

**Root cause:** LoRA with high learning rate can overwrite general instruction-following ability.

**Mitigation:**
- Keep `learning_rate ≤ 2e-4`
- Use `lora_dropout=0.05`
- Train for ≤ 3 epochs (PubMedQA is large; 1–2 epochs is usually sufficient)
- **Measure it:** Run 50 MMLU samples on the fine-tuned model; if accuracy drops >10% vs base, reduce epochs

---

## 6. Data Pipeline

### 6.1 PubMedQA

- **Source:** HF Datasets `"qiaojin/PubMedQA"`, config `"pqa_labeled"` (1,000 labeled) + `"pqa_artificial"` (211,269 artificial)
- **Task type:** Closed-domain QA — given a PubMed abstract (context), answer yes/no/maybe
- **Train split:** Use `pqa_artificial` (211k) for training
- **Test split:** Use `pqa_labeled` test portion (500 samples) for evaluation

### 6.2 Instruction Template

Format every sample into this Alpaca-style instruction tuple before tokenization:
```
### Instruction:
You are a medical expert. Based ONLY on the provided context, answer the question with 'yes', 'no', or 'maybe'. Then briefly explain your reasoning.

### Context:
{abstract}

### Question:
{question}

### Response:
{final_decision}. {long_answer}
```

**Why this template matters:** Mistral-7B-Instruct expects `[INST]...[/INST]` formatting, but Alpaca-style is simpler for SFTTrainer and produces equivalent results. Do NOT mix both.

### 6.3 Data Quality Filter

Before training, filter samples where:
- `long_answer` is `None` or empty (removes ~3% of artificial split)
- `abstract` length > 4096 characters (outliers; removes <0.5%)
- `final_decision` not in `["yes", "no", "maybe"]` (malformed labels)

```python
def is_valid(sample):
    return (
        sample["final_decision"] in ["yes", "no", "maybe"]
        and sample["long_answer"] is not None
        and len(sample["long_answer"]) > 10
        and sample["context"]["contexts"] is not None
    )

dataset = dataset.filter(is_valid, num_proc=4)
```

---

## 7. Training Configuration (Reference YAML)

Save this as `configs/training_config.yaml`. The training script reads from this file — no hardcoded values.

```yaml
# configs/training_config.yaml
model:
  base_model_id: "mistralai/Mistral-7B-Instruct-v0.3"
  max_seq_length: 512
  trust_remote_code: false

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true

lora:
  r: 16
  lora_alpha: 32
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"

training:
  output_dir: "./outputs/meditune-checkpoint"
  num_train_epochs: 2
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  learning_rate: 2.0e-4
  weight_decay: 0.001
  optim: "paged_adamw_32bit"
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03
  max_grad_norm: 0.3
  logging_steps: 25
  save_steps: 200
  eval_steps: 200
  evaluation_strategy: "steps"
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  fp16: false
  bf16: true
  group_by_length: true
  packing: true
  report_to: "wandb"

data:
  dataset_name: "qiaojin/PubMedQA"
  train_config: "pqa_artificial"
  eval_config: "pqa_labeled"
  train_split: "train"
  eval_split: "validation"
  max_train_samples: 50000    # start with 50k; full 211k for final run
  max_eval_samples: 500

hub:
  push_to_hub: true
  hub_model_id: "nikhilsh10/meditune-mistral-7b"
  hub_strategy: "every_save"
```

---

## 8. Evaluation Plan

### 8.1 PubMedQA Accuracy (Primary)

```python
def extract_decision(text):
    """Extract yes/no/maybe from model output."""
    text = text.lower().strip()
    for label in ["yes", "no", "maybe"]:
        if text.startswith(label):
            return label
    # Fallback: find first occurrence
    for label in ["yes", "no", "maybe"]:
        if label in text[:50]:
            return label
    return "unknown"

def evaluate_pubmedqa(model, tokenizer, test_dataset, n=500):
    correct = 0
    for sample in test_dataset.select(range(n)):
        prompt = build_prompt(sample)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extract_decision(response)
        if pred == sample["final_decision"]:
            correct += 1
    return correct / n
```

Run this for **both** base model and fine-tuned model. Report absolute accuracy and delta.

### 8.2 ROUGE-L (Secondary)

Use `evaluate.load("rouge")` on the `long_answer` field for 200 MedQuAD samples.

### 8.3 MMLU Sanity Check (Catastrophic Forgetting Guard)

Sample 50 questions from MMLU `college_medicine` subset. Run base model and fine-tuned model. If delta < -10%, document it in the Model Card as a known limitation.

---

## 9. Repository Structure

```
meditune/
├── configs/
│   └── training_config.yaml       # All hyperparameters here
├── data/
│   └── prepare_dataset.py         # Download + format + filter PubMedQA
├── src/
│   ├── train.py                   # Main training script
│   ├── evaluate.py                # PubMedQA accuracy + ROUGE-L
│   ├── inference.py               # Inference helper
│   └── utils.py                   # Prompt builder, metric helpers
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA on PubMedQA
│   ├── 02_training_kaggle.ipynb   # The actual Kaggle training notebook
│   └── 03_evaluation.ipynb        # Before/after eval notebook
├── app/
│   └── app.py                     # Gradio demo for HF Spaces
├── model_card/
│   └── README.md                  # HF Model Card (auto-pushed to Hub)
├── requirements.txt
├── requirements-train.txt         # Training-only heavy deps
└── README.md                      # Project README with results table
```

---

## 10. Environment & Dependencies

### requirements-train.txt (Kaggle)
```
torch>=2.3.0
transformers>=4.41.0
peft>=0.11.0
trl>=0.9.0
bitsandbytes>=0.43.0
datasets>=2.19.0
accelerate>=0.30.0
wandb>=0.17.0
evaluate>=0.4.2
rouge_score>=0.1.2
huggingface_hub>=0.23.0
scipy>=1.13.0
sentencepiece>=0.2.0
protobuf>=4.25.0
```

### requirements.txt (Demo / Inference only)
```
torch>=2.3.0+cpu
transformers>=4.41.0
peft>=0.11.0
bitsandbytes>=0.43.0
gradio>=4.36.0
huggingface_hub>=0.23.0
sentencepiece>=0.2.0
```

### Kaggle-specific setup cell (run first in every session)
```python
!pip install -q peft trl bitsandbytes accelerate wandb evaluate rouge_score -U
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU 0: {torch.cuda.get_device_name(0)}")
# Expected: CUDA available: True, GPU count: 2, GPU 0: Tesla T4
```

---

## 11. Compute Estimate

| Phase | Steps | Estimated Time (2×T4) |
|-------|-------|----------------------|
| Data download + preprocessing | — | ~15 min |
| Training (50k samples, 2 epochs) | ~25,000 | ~2.5 hours |
| Training (211k samples, 2 epochs) | ~105,000 | ~8–10 hours |
| Evaluation (500 samples, base) | 500 forward passes | ~20 min |
| Evaluation (500 samples, fine-tuned) | 500 forward passes | ~20 min |
| Model merge + push | — | ~30 min |

**Recommendation:** Start with 50k samples. If accuracy delta > 8%, that's your portfolio story. Only run full 211k if you have Kaggle quota remaining and want to maximize accuracy.
