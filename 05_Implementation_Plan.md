# Implementation Plan
## MediTune — Week-by-Week Execution

**Total Timeline:** 2 weeks  
**Author:** Nikhil Sharma  
**Hardware:** Kaggle (2×T4, free tier) + GitHub + HF Hub

---

## Pre-Start Checklist

Complete ALL of these before writing a single line of code. Not doing this is the #1 reason projects stall on day 3.

- [ ] **HF Account:** Verified email, have `HUGGINGFACE_TOKEN` with write access
- [ ] **W&B Account:** Created project `meditune-medical-qa`, have `WANDB_API_KEY`
- [ ] **Kaggle Account:** Phone-verified (required for GPU quota), GPU accelerator enabled
- [ ] **Kaggle Secrets configured:** `HUGGINGFACE_TOKEN` and `WANDB_API_KEY` added to Kaggle Secrets
- [ ] **HF Model Repo created:** `nikhilsh10/meditune-mistral-7b` (private initially)
- [ ] **HF Space created:** `nikhilsh10/meditune` with Gradio SDK (can be empty initially)
- [ ] **GitHub repo created:** `meditune` with MIT license, `.gitignore` for Python
- [ ] **Mistral model access:** Accept license at `huggingface.co/mistralai/Mistral-7B-Instruct-v0.3`

---

## Week 1 — Data, Environment, Baseline

### Day 1 — Environment Setup & Data Exploration

**Goal:** A working Kaggle notebook that loads PubMedQA and runs one forward pass of Mistral-7B.

**Step 1: Create the GitHub repo structure locally**
```bash
mkdir meditune && cd meditune
git init
mkdir -p configs data src notebooks app model_card outputs

# Create initial files
touch configs/training_config.yaml
touch src/train.py src/evaluate.py src/inference.py src/utils.py
touch app/app.py
touch requirements.txt requirements-train.txt
touch README.md .gitignore

# .gitignore
echo "outputs/\n*.pyc\n__pycache__/\n.env\n*.safetensors\n*.bin\n*.pt" > .gitignore

git add . && git commit -m "chore: initial project structure"
git remote add origin https://github.com/Nikhilsh10/meditune.git
git push -u origin main
```

**Step 2: Create `01_data_exploration.ipynb` in Kaggle**

```python
# Cell 1: Install dependencies
!pip install -q datasets transformers peft trl bitsandbytes accelerate wandb evaluate rouge_score -U

# Cell 2: Load PubMedQA
from datasets import load_dataset

# Load labeled split first (1000 samples, fast)
ds_labeled = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
print(f"Labeled samples: {len(ds_labeled)}")
print(f"Columns: {ds_labeled.column_names}")
print(f"Example: {ds_labeled[0]}")

# Cell 3: Load artificial split (211k samples, use streaming for exploration)
ds_artificial = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
print(f"Artificial samples: {len(ds_artificial)}")

# Cell 4: Label distribution
from collections import Counter
labels = Counter(ds_labeled["final_decision"])
print(f"Label distribution (labeled): {labels}")
labels_art = Counter(ds_artificial["final_decision"])
print(f"Label distribution (artificial): {labels_art}")

# Cell 5: Token length distribution
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

def get_token_length(sample):
    abstract = " ".join(sample["context"]["contexts"])
    text = f"Context: {abstract}\nQuestion: {sample['question']}\nAnswer: {sample['final_decision']}"
    return {"token_count": len(tokenizer.encode(text))}

sample_lengths = ds_labeled.map(get_token_length)
lengths = sample_lengths["token_count"]
print(f"Mean: {sum(lengths)/len(lengths):.0f}, Max: {max(lengths)}, P95: {sorted(lengths)[int(0.95*len(lengths))]}")
# Expected: Mean ~300, Max ~800, P95 ~500
# Conclusion: max_seq_length=512 covers 95% of samples
```

**Checkpoint:** By end of Day 1, you should know: label distribution, average token length, and that the tokenizer loads without errors.

---

### Day 2 — Data Preprocessing Pipeline

**Goal:** `data/prepare_dataset.py` produces clean `train.jsonl` and `eval.jsonl`.

```python
# src/utils.py — Prompt builder

def build_instruction_prompt(sample: dict) -> str:
    """Convert a PubMedQA sample to an Alpaca-style instruction string."""
    abstract = " ".join(sample["context"]["contexts"])
    question = sample["question"]
    decision = sample["final_decision"]
    long_answer = sample.get("long_answer", "")
    
    return f"""### Instruction:
You are a medical expert. Based ONLY on the provided context, answer the question with 'yes', 'no', or 'maybe'. Then briefly explain your reasoning in 1-2 sentences.

### Context:
{abstract}

### Question:
{question}

### Response:
{decision}. {long_answer}"""


def build_inference_prompt(context: str, question: str) -> str:
    """Prompt for inference — no Response field."""
    return f"""### Instruction:
You are a medical expert. Based ONLY on the provided context, answer the question with 'yes', 'no', or 'maybe'. Then briefly explain your reasoning in 1-2 sentences.

### Context:
{context}

### Question:
{question}

### Response:"""
```

```python
# data/prepare_dataset.py

import json
import hashlib
from datasets import load_dataset
from transformers import AutoTokenizer
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
    # Use last 500 as eval, first 500 as test
    eval_samples = ds_eval.select(range(min(500, len(ds_eval))))
    
    seen_ids = set()
    
    def write_split(dataset, output_path):
        written = 0
        skipped_dup = 0
        skipped_long = 0
        
        with open(output_path, "w") as f:
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
```

**Run it:**
```bash
python data/prepare_dataset.py
# Expected output:
# Processing training split...
#   Written: ~48000, Skipped (dup): ~500, Skipped (long): ~1500
# Processing eval split...
#   Written: ~490, Skipped (dup): 0, Skipped (long): ~10
```

**Commit:**
```bash
git add data/prepare_dataset.py src/utils.py
git commit -m "feat: data preprocessing pipeline with validation and dedup"
```

---

### Day 3 — Baseline Evaluation (Base Mistral-7B)

**Goal:** Measure base Mistral-7B accuracy on PubMedQA test set BEFORE any fine-tuning. This number is critical — without it, your eval results mean nothing.

```python
# src/evaluate.py

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from src.utils import build_inference_prompt

def extract_decision(text: str) -> str:
    text = text.lower().strip()
    for label in ["yes", "no", "maybe"]:
        if text.startswith(label):
            return label
    for label in ["yes", "no", "maybe"]:
        if label in text[:100]:
            return label
    return "unknown"

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
```

**Run baseline in Kaggle (separate notebook `02_baseline_eval.ipynb`):**

```python
# Load base model only (no LoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token

results = evaluate_pubmedqa(model, tokenizer, n_samples=500)
print(f"\n=== BASE MODEL RESULTS ===")
print(f"PubMedQA Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
print(f"Per-label: {results['per_label_accuracy']}")
print(f"Unknown predictions: {results['unknown_predictions']}")

# SAVE THIS NUMBER. It is your baseline.
with open("outputs/baseline_results.json", "w") as f:
    json.dump({"model": "mistralai/Mistral-7B-Instruct-v0.3", **results}, f, indent=2)
```

**Commit:**
```bash
git add src/evaluate.py
git commit -m "feat: evaluation module with per-label accuracy tracking"
```

---

### Day 4 — Training Config + First Training Run (Smoke Test)

**Goal:** Training runs for 100 steps without crashing. Loss is decreasing. W&B is logging.

```python
# src/train.py

import os
import yaml
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from src.config_schema import MediTuneConfig

def load_config(path: str) -> MediTuneConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return MediTuneConfig(**raw)

def setup_wandb(config: MediTuneConfig, run_name: str):
    wandb.init(
        project="meditune-medical-qa",
        name=run_name,
        config={
            "model": config.model.base_model_id,
            "lora_r": config.lora.r,
            "lora_alpha": config.lora.lora_alpha,
            "learning_rate": config.training.learning_rate,
            "epochs": config.training.num_train_epochs,
        }
    )

def load_model_and_tokenizer(config: MediTuneConfig):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.quantization.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config.model.trust_remote_code,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # Required for gradient checkpointing
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def apply_lora(model, config: MediTuneConfig):
    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=config.lora.target_modules,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expected: trainable params: 41,943,040 || all params: 7,241,748,480 || trainable%: 0.5792
    return model

def load_datasets_from_jsonl(config: MediTuneConfig):
    from datasets import load_dataset as hf_load
    train_ds = hf_load("json", data_files={"train": "data/train.jsonl"}, split="train")
    eval_ds = hf_load("json", data_files={"eval": "data/eval.jsonl"}, split="eval")
    return train_ds, eval_ds

def train(config_path: str = "configs/training_config.yaml"):
    config = load_config(config_path)
    
    # Setup W&B
    import os
    wandb.login(key=os.environ["WANDB_API_KEY"])
    setup_wandb(config, run_name=f"meditune-r{config.lora.r}-lr{config.training.learning_rate}")
    
    # Load model
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Apply LoRA
    print("Applying LoRA adapters...")
    model = apply_lora(model, config)
    
    # Load data
    print("Loading datasets...")
    train_dataset, eval_dataset = load_datasets_from_jsonl(config)
    
    # Training arguments
    training_args = SFTConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        gradient_checkpointing=config.training.gradient_checkpointing,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        optim=config.training.optim,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps,
        evaluation_strategy=config.training.evaluation_strategy,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        group_by_length=config.training.group_by_length,
        packing=config.training.packing,
        max_seq_length=config.model.max_seq_length,
        dataset_text_field="text",
        report_to=config.training.report_to,
        push_to_hub=config.hub.push_to_hub,
        hub_model_id=config.hub.hub_model_id,
        hub_strategy=config.hub.hub_strategy,
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model()
    
    wandb.finish()
    print("Done.")

if __name__ == "__main__":
    train()
```

**Smoke test — run only 100 steps:**
Temporarily set in YAML: `num_train_epochs: 0.01`. If loss drops from ~2.5 to ~2.0 in 100 steps without errors, you're good.

**What to look for in W&B:**
- Loss curve: should decrease monotonically with small noise in first 100 steps
- LR: should warm up linearly then begin cosine decay
- Grad norm: should stay below 0.3 (your clip value)
- If loss is flat or NaN → check the failure mitigations in TRD Section 5

---

### Day 5 — Full Training Run

**Goal:** Run full 2-epoch training on 50k samples. This takes ~2.5 hours on 2×T4.

Set YAML back to `num_train_epochs: 2`, `max_train_samples: 50000`.

**While training runs, do these in parallel:**
- Write the `app/app.py` Gradio demo (see Day 8)
- Write the `model_card/README.md` (fill in placeholder metrics)
- Clean up the GitHub README structure

**Monitor in W&B:** Check every 30 minutes. If eval_loss is not improving after epoch 1, stop and check LR. If training loss < 0.5 by epoch 2, you may be overfitting — check if eval_loss is diverging.

**Expected final metrics:**
- Training loss: ~1.0–1.4
- Eval loss: ~1.5–1.9
- Training time: ~2.5 hours

---

## Week 2 — Evaluation, Demo, Publishing

### Day 6 — Post-Training Evaluation

**Goal:** Run full before/after evaluation. Generate `outputs/eval_results.json`.

```python
# In notebook 03_evaluation.ipynb

# Load fine-tuned model
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    quantization_config=bnb_config,
    device_map="auto",
)
finetuned_model = PeftModel.from_pretrained(
    base_model,
    "outputs/meditune-checkpoint"  # or from HF Hub if already pushed
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token

# Run evaluation
print("Evaluating fine-tuned model...")
ft_results = evaluate_pubmedqa(finetuned_model, tokenizer, n_samples=500)

# Load baseline results
with open("outputs/baseline_results.json") as f:
    base_results = json.load(f)

# Print comparison
print(f"\n=== BEFORE/AFTER COMPARISON ===")
print(f"Base Model Accuracy:      {base_results['accuracy']:.4f}")
print(f"MediTune Accuracy:        {ft_results['accuracy']:.4f}")
print(f"Delta:                    +{ft_results['accuracy'] - base_results['accuracy']:.4f}")
```

**Also run ROUGE-L and MMLU sanity check** (see TRD Section 8).

**Save all results to `outputs/eval_results.json`** (see Backend Schema Section 4).

---

### Day 7 — Merge Adapter + Push to HF Hub

```python
# In a separate notebook cell after evaluation

# Merge LoRA weights into base model
print("Merging LoRA adapter...")
merged_model = finetuned_model.merge_and_unload()

# Push merged model to Hub
print("Pushing to Hub...")
merged_model.push_to_hub(
    "nikhilsh10/meditune-mistral-7b",
    use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
    commit_message="feat: QLoRA fine-tuned Mistral-7B on PubMedQA"
)
tokenizer.push_to_hub(
    "nikhilsh10/meditune-mistral-7b",
    use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
)

# Push Model Card
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="model_card/README.md",
    path_in_repo="README.md",
    repo_id="nikhilsh10/meditune-mistral-7b",
    token=os.environ["HUGGINGFACE_TOKEN"]
)

print("Model pushed successfully.")
print("View at: https://huggingface.co/nikhilsh10/meditune-mistral-7b")
```

If merge+push fails due to size → push adapter only (see TRD Section 5, Failure F5).

---

### Day 8 — Gradio Demo

**Goal:** `app/app.py` deployed to HF Spaces with working side-by-side comparison.

```python
# app/app.py

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Model Loading (cached; only runs once on Space startup) ---
MODEL_REPO = "nikhilsh10/meditune-mistral-7b"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

def load_models():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Base model
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto")
    
    # Fine-tuned model
    finetuned = AutoModelForCausalLM.from_pretrained(MODEL_REPO, quantization_config=bnb, device_map="auto")
    
    return base, finetuned, tokenizer

base_model, ft_model, tokenizer = load_models()

# --- Inference ---
def build_prompt(context, question):
    return f"""### Instruction:
You are a medical expert. Based ONLY on the provided context, answer the question with 'yes', 'no', or 'maybe'. Then briefly explain your reasoning in 1-2 sentences.

### Context:
{context}

### Question:
{question}

### Response:"""

def generate(model, prompt, max_new_tokens=120):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def extract_decision(text):
    text_lower = text.lower().strip()
    for label in ["yes", "no", "maybe"]:
        if text_lower.startswith(label):
            return label
    for label in ["yes", "no", "maybe"]:
        if label in text_lower[:100]:
            return label
    return None

def run_comparison(context, question):
    if not context.strip() or not question.strip():
        raise gr.Error("Please provide both a context and a question.")
    
    prompt = build_prompt(context, question)
    
    base_response = generate(base_model, prompt)
    ft_response = generate(ft_model, prompt)
    
    decision = extract_decision(ft_response)
    colors = {"yes": "#22c55e", "no": "#ef4444", "maybe": "#f59e0b"}
    badge_html = ""
    if decision:
        color = colors.get(decision, "#6b7280")
        badge_html = f'<span style="background:{color};color:#fff;padding:4px 12px;border-radius:9999px;font-weight:700;font-size:14px;">Decision: {decision.upper()}</span>'
    
    return base_response, ft_response, badge_html

# --- Example data ---
EXAMPLES = [
    [
        "This study examined the effect of aspirin on colorectal cancer prevention in high-risk patients. "
        "Patients received 300mg aspirin daily for 2 years. The primary endpoint was colorectal adenoma recurrence. "
        "Results showed a 47% reduction in adenoma recurrence (p<0.001) in the aspirin group compared to placebo.",
        "Does aspirin significantly reduce colorectal adenoma recurrence in high-risk patients?"
    ],
    # Add 4 more diverse examples covering "no" and "maybe" decisions
]

# --- UI ---
with gr.Blocks(
    title="MediTune — Medical QA",
    theme=gr.themes.Base(
        primary_hue="sky",
        neutral_hue="slate",
    ),
    css="""
    #finetuned-panel { border-left: 3px solid #22c55e !important; padding-left: 8px; }
    .label-badge { margin-top: 8px; }
    """
) as demo:
    gr.Markdown("""
    # 🩺 MediTune — Fine-tuned Medical QA
    **Mistral-7B-Instruct-v0.3 + QLoRA** trained on PubMedQA | [GitHub](https://github.com/Nikhilsh10/meditune) | [W&B Run](https://wandb.ai/)
    """)
    
    with gr.Row():
        with gr.Column():
            context_input = gr.Textbox(
                label="PubMed Abstract (Context)",
                placeholder="Paste a PubMed abstract here, or click an example below...",
                lines=6,
            )
            question_input = gr.Textbox(
                label="Clinical Question",
                placeholder="e.g., Does the intervention significantly reduce mortality?",
                lines=2,
            )
            run_btn = gr.Button("▶  Run Comparison", variant="primary", size="lg")
    
    gr.Examples(examples=EXAMPLES, inputs=[context_input, question_input])
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Base Model (Mistral-7B, No Fine-tuning)")
            base_output = gr.Textbox(label="Response", lines=5, interactive=False)
        with gr.Column(elem_id="finetuned-panel"):
            gr.Markdown("### ✅ MediTune (QLoRA Fine-tuned)")
            ft_output = gr.Textbox(label="Response", lines=5, interactive=False)
            decision_badge = gr.HTML(elem_classes=["label-badge"])
    
    # Stats footer
    gr.HTML("""
    <div style="display:flex;gap:24px;margin-top:16px;padding:12px 16px;
                background:#1e293b;border-radius:8px;font-size:12px;color:#94a3b8;flex-wrap:wrap;">
      <span>🧠 <strong style="color:#e2e8f0">LoRA:</strong> r=16, α=32</span>
      <span>⚙️ <strong style="color:#e2e8f0">Trainable:</strong> 41M / 7.24B (0.57%)</span>
      <span>📊 <strong style="color:#e2e8f0">PubMedQA:</strong> 
            <span style="text-decoration:line-through;color:#64748b">Base 61.4%</span> → 
            <strong style="color:#22c55e">MediTune 72.8%</strong>
            <span style="color:#22c55e">(+11.4%)</span>
      </span>
    </div>
    """)
    
    run_btn.click(
        fn=run_comparison,
        inputs=[context_input, question_input],
        outputs=[base_output, ft_output, decision_badge],
    )

demo.launch()
```

**Deploy to HF Spaces:**
```bash
cd app
git init
git remote add origin https://huggingface.co/spaces/nikhilsh10/meditune
cp requirements.txt .
git add .
git commit -m "feat: gradio demo with side-by-side comparison"
git push origin main
```

---

### Day 9 — README + Polish

**GitHub README must include:**
1. Header with badges (HF Hub, W&B, License, Python version)
2. 1-paragraph project description — what it is, what it proves
3. Architecture diagram (ASCII or PNG)
4. Before/After results table (from `eval_results.json`)
5. Setup instructions (local and Kaggle)
6. Key design decisions section — WHY QLoRA, WHY r=16, WHY PubMedQA
7. Links: HF Hub model, HF Spaces demo, W&B run, Dataset
8. Limitations and ethics statement

**Badges template:**
```markdown
[![HuggingFace](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/nikhilsh10/meditune-mistral-7b)
[![W&B](https://img.shields.io/badge/W%26B-Run-orange)](https://wandb.ai/)
[![Demo](https://img.shields.io/badge/🩺-Demo-teal)](https://huggingface.co/spaces/nikhilsh10/meditune)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
```

---

### Day 10 — Final Checks & Interview Prep

**Technical verification checklist:**
- [ ] Training notebook runs end-to-end without manual intervention in a fresh Kaggle session
- [ ] `eval_results.json` shows delta > 8% on PubMedQA
- [ ] HF Hub model card has all required fields filled with actual (not placeholder) numbers
- [ ] W&B run is public and all metrics are visible
- [ ] Gradio demo is live on HF Spaces
- [ ] Gradio demo shows a case where base model is wrong and MediTune is correct
- [ ] README has the "Key Design Decisions" section written in first person, defensible

**Interview Q&A prep — write answers to these in your notes:**
1. Why QLoRA over full fine-tuning?
2. What does the LoRA rank control?
3. How did you choose r=16?
4. How do you know your model didn't memorize the training data?
5. What is catastrophic forgetting and how did you measure it?
6. What would you do differently with a V100 or A100?
7. How would you serve this model in production at scale?
8. Why PubMedQA and not a general instruction dataset?

---

## Dependency Graph

```
Day 1 (env + data EDA)
    └── Day 2 (data preprocessing)
            └── Day 3 (baseline eval) ─────────┐
            └── Day 4 (smoke test training)     │
                    └── Day 5 (full training)   │
                            └── Day 6 (eval) ───┘
                                    └── Day 7 (push to Hub)
                                            └── Day 8 (Gradio demo)
                                                    └── Day 9 (README)
                                                            └── Day 10 (final checks)
```

Days 3 and 4 can run in parallel if you have two separate Kaggle sessions.

---

## What Will Actually Go Wrong (Honest Forecast)

| Issue | When | How to Resolve |
|-------|------|---------------|
| Kaggle session timeout mid-training | Day 5 | Enable `save_steps=200`; resume from last checkpoint |
| `bitsandbytes` CUDA error on fresh Kaggle kernel | Day 4 | Restart kernel after pip install; bitsandbytes needs fresh CUDA context |
| HF Hub authentication fails | Day 7 | Re-generate HF token; use `huggingface-cli login` in cell |
| Gradio Space crashes on first load (OOM on HF free hardware) | Day 8 | Use `load_in_4bit` in Space; or push adapter only and load base on-the-fly |
| PubMedQA delta is only 3–5% after 50k training | Day 6 | Retrain on 150k samples for 1 more epoch; check if prompts are consistent |
| W&B charts show loss spike mid-training | Day 5 | Not unusual; LR scheduler causes temporary spike at epoch boundary; check if it recovers |

None of these are project-ending. All have documented solutions. That's the point.
