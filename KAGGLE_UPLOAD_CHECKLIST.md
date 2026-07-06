# KAGGLE UPLOAD CHECKLIST — MediTune QLoRA Fine-Tuning
# Prepared for: Nikhil Sharma (Nikhilsh10)
# Project: MediTune — Mistral-7B-Instruct-v0.3 on PubMedQA

---

## 1. FILES TO UPLOAD TO KAGGLE AS A DATASET

Upload these files/folders as a Kaggle Dataset named `meditune-codebase`:

```
MediTune-Final/
├── src/
│   ├── config_schema.py
│   ├── utils.py          ← (note: may appear as Utils.py — rename to utils.py)
│   ├── train.py
│   ├── evaluate.py       ← (note: may appear as Evaluate.py — rename to evaluate.py)
│   └── inference.py
├── data/
│   └── prepare_dataset.py
├── configs/
│   └── training_config.yaml
├── requirements-train.txt
└── app/
    └── app.py
```

> WARNING: Kaggle is Linux (case-sensitive). Ensure filenames are all lowercase:
>   - src/Utils.py → src/utils.py
>   - src/Evaluate.py → src/evaluate.py

Steps to upload:
1. Go to https://www.kaggle.com/datasets
2. Click "+ New Dataset"
3. Name it: meditune-codebase
4. Upload the folder contents listed above
5. Click "Create"
6. Note the dataset path: /kaggle/input/meditune-codebase/

---

## 2. KAGGLE SECRETS TO CONFIGURE

Go to your Kaggle Notebook → Add-ons → Secrets → Add New Secret

| Secret Name          | Value Source                         |
|----------------------|--------------------------------------|
| WANDB_API_KEY        | https://wandb.ai/authorize           |
| HUGGINGFACE_TOKEN    | https://huggingface.co/settings/tokens |

Both should have "Notebook access" enabled.
HUGGINGFACE_TOKEN must have "write" permission scope on HF Hub.

After adding, verify in notebook with:
```python
import os
print("WandB OK:", "WANDB_API_KEY" in os.environ)
print("HF OK:", "HUGGINGFACE_TOKEN" in os.environ)
```

Expected output:
```
WandB OK: True
HF OK: True
```

---

## 3. EXACT PIP INSTALL CELL (run this first — Cell 1)

```python
# Cell 1 — Install dependencies
# Runtime: GPU T4 x2, Python 3.11, CUDA 12.1

import subprocess, sys

result = subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "-r", "/kaggle/input/meditune-codebase/requirements-train.txt"
], capture_output=True, text=True)

print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.returncode != 0:
    print("INSTALL ERROR:", result.stderr)
    raise RuntimeError("Dependency install failed — check output above")

print("✅ All dependencies installed successfully")
```

If you see any version conflict errors, add this override after the pip install:
```python
# Conflict override if needed:
!pip install -q "torch>=2.3.0,<2.5" --upgrade
```

---

## 4. EXACT CELL ORDER FOR THE SMOKE TEST

Run cells in this exact order. Do not skip any cell.

### Cell 1 — Install dependencies
(See Section 3 above)

### Cell 2 — Mount codebase and verify secrets
```python
# Cell 2 — Setup: paths + secrets
import os, sys

REPO = "/kaggle/input/meditune-codebase"
sys.path.insert(0, REPO)

# Verify secrets
assert "WANDB_API_KEY" in os.environ, "Missing WANDB_API_KEY secret"
assert "HUGGINGFACE_TOKEN" in os.environ, "Missing HUGGINGFACE_TOKEN secret"

# Set HF token for authenticated downloads
os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

print(f"Repo path:     {REPO}")
print(f"Python:        {sys.version}")
print("Secrets:       ✅")
```

### Cell 3 — Prepare data (smoke-test size: 2000 train, 100 eval)
```python
# Cell 3 — Data preparation
import subprocess
result = subprocess.run([
    sys.executable,
    f"{REPO}/data/prepare_dataset.py",
    "--max-train", "2000",
    "--max-eval",  "100",
    "--max-seq-length", "512",
    "--model-id", "mistralai/Mistral-7B-Instruct-v0.3"
], capture_output=True, text=True, cwd="/kaggle/working")

print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr)
    raise RuntimeError("Data preparation failed")
```

### Cell 4 — Verify JSONL outputs
```python
# Cell 4 — Verify data
import json, os

for fname in ["train.jsonl", "eval.jsonl"]:
    path = f"/kaggle/working/data/{fname}"
    assert os.path.exists(path), f"Missing: {path}"
    with open(path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    print(f"{fname}: {len(lines)} records")
    print(f"  Sample keys:  {list(lines[0].keys())}")
    print(f"  Sample text[:80]: {lines[0]['text'][:80]}")
    print()
```

Expected output:
```
train.jsonl: ~1800-2000 records  (some filtered by token length)
eval.jsonl:  ~90-100 records
```

### Cell 5 — Run smoke test (max_steps=10)
```python
# Cell 5 — Smoke test training run
import subprocess
result = subprocess.run([
    sys.executable, f"{REPO}/src/train.py",
    "--config", f"{REPO}/configs/training_config.yaml",
    "--smoke-test"   # overrides max_steps to 10
], capture_output=True, text=True, cwd="/kaggle/working")

print(result.stdout[-5000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError("Training smoke test failed")
```

### Cell 6 — Verify checkpoint saved
```python
# Cell 6 — Verify outputs exist
import os
output_dir = "/kaggle/working/outputs/meditune-checkpoint"
assert os.path.isdir(output_dir), f"Output dir missing: {output_dir}"

files = os.listdir(output_dir)
print("Checkpoint files:", files)

# Must contain adapter weights
assert any("adapter" in f for f in files), \
    "No adapter files found! Training may have failed silently."
print("✅ Adapter weights confirmed")
```

---

## 5. WHAT OUTPUT TO LOOK FOR TO CONFIRM SMOKE TEST PASSED

### ✅ PASS signals — look for ALL of these:
1. **Cell 3 output contains:**
   ```
   train: written=XXXX, skipped_invalid=X, skipped_dup=X, skipped_long=X
   eval:  written=XXX,  skipped_invalid=X, skipped_dup=X, skipped_long=X
   ```

2. **Cell 5 output contains:**
   ```
   Loading model and tokenizer...
   Applying LoRA adapters...
   trainable params: 41,943,040 || all params: 7,289,966,592 || trainable%: 0.5754
   Loading datasets...
   Starting training...
   {'loss': X.XXX, 'grad_norm': X.XXX, 'learning_rate': X.XXXXXX, ...}
   ```
   Loss must be a finite number (not `nan` or `inf`).

3. **Cell 6 output contains:**
   ```
   Checkpoint files: ['adapter_config.json', 'adapter_model.safetensors', ...]
   ✅ Adapter weights confirmed
   ```

4. **W&B dashboard** (if WANDB_API_KEY is set) shows a new run named
   `meditune-r16-lr0.0002` with a smoothly decreasing train/loss curve.

### ❌ FAIL signals — stop immediately if you see:
- `nan` in any loss value
- `CUDA out of memory` (see Section 6 below)
- `TypeError: SFTConfig.__init__() got unexpected keyword argument`
- `FileNotFoundError: data/train.jsonl`
- `AssertionError` in Cell 6

---

## 6. WHAT TO DO IF CUDA OOM OCCURS

This corresponds to TRD Section 5, Failure Mode F1: GPU OOM.

### Immediate triage steps (in order):

**Step 1 — Confirm OOM is the cause**
```
CUDA out of memory. Tried to allocate XXX GiB
```
If you see this, proceed to Step 2.

**Step 2 — Reduce batch size first (lowest risk)**
In `configs/training_config.yaml`, change:
```yaml
per_device_train_batch_size: 1    # already minimum — do NOT reduce further
gradient_accumulation_steps: 4    # reduce to 2 if needed
```

**Step 3 — Reduce max sequence length**
In `configs/training_config.yaml`, change:
```yaml
max_seq_length: 512    →    max_seq_length: 384
```
Then re-run `prepare_dataset.py` with `--max-seq-length 384` to regenerate JSONL.

**Step 4 — Disable gradient checkpointing fallback**
If OOM persists despite Steps 2-3, ensure gradient checkpointing is ON:
```yaml
gradient_checkpointing: true    # must be true — do NOT set false
```

**Step 5 — Force single GPU only**
Add to Cell 5 before the subprocess call:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 0
```
This bypasses the multi-GPU tensor placement issue entirely.

**Step 6 — Nuclear option: reduce LoRA rank**
Only if all above steps fail. In `configs/training_config.yaml`:
```yaml
lora:
  r: 8          # down from r=16
  lora_alpha: 16  # keep at 2x r
```
Note: r=8 reduces trainable parameters to ~21M (0.29%). Results may degrade slightly.

**Step 7 — Restart kernel**
After any config change, restart the Kaggle notebook kernel completely
and re-run from Cell 1. GPU memory is not freed until kernel restart.

---

## REFERENCE: Key file locations on Kaggle

| What                    | Path                                                    |
|-------------------------|---------------------------------------------------------|
| Source code             | /kaggle/input/meditune-codebase/                        |
| Training data output    | /kaggle/working/data/train.jsonl                        |
| Eval data output        | /kaggle/working/data/eval.jsonl                         |
| Model checkpoint        | /kaggle/working/outputs/meditune-checkpoint/            |
| Eval results JSON       | /kaggle/working/outputs/eval_results.json               |
| W&B run URL             | https://wandb.ai/YOUR_USERNAME/meditune-medical-qa      |

---
END OF CHECKLIST
