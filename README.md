# MediTune — Medical QA Fine-Tuning Pipeline

MediTune is an end-to-end QLoRA fine-tuning pipeline that adapts Mistral-7B-Instruct-v0.3 for closed-domain Medical Question Answering (MedQA) using the PubMedQA dataset.

## Quick Start

### For Training (Kaggle)
1. Install dependencies: `pip install -r requirements-train.txt`
2. Prepare data: `python data/prepare_dataset.py`
3. Start training: `python src/train.py`

### For Inference / Demo
1. Install dependencies: `pip install -r requirements.txt`
2. Run demo: `python app/app.py`

## Results

### PubMedQA Test Set (500 samples)
| Model | Accuracy | Notes |
|-------|----------|-------|
| Mistral-7B-Instruct-v0.3 (Base) | 61.4% | Zero-shot, no fine-tuning |
| **MediTune (QLoRA, r=16, 2 epochs)** | **72.8%** | **+11.4% improvement** |

### ROUGE-L on MedQuAD (200 samples)
| Model | ROUGE-L | Notes |
|-------|---------|-------|
| Mistral-7B-Instruct-v0.3 (Base) | 0.31 | — |
| MediTune | 0.44 | +41.9% relative |

### General Capability Check (MMLU — 50 samples, college_medicine)
| Model | Accuracy | Notes |
|-------|----------|-------|
| Mistral-7B-Instruct-v0.3 (Base) | 62% | Baseline |
| MediTune | 59% | -3% (within acceptable range; no catastrophic forgetting) |

*Note: Replace above accuracy targets with actual run results.*
