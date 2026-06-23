# MediTune — Full Project Specification Bundle

**Project:** QLoRA Fine-Tuning Pipeline for Medical Question Answering  
**Model:** Mistral-7B-Instruct-v0.3 + PEFT/QLoRA  
**Dataset:** PubMedQA (MIT License)  
**Hardware:** Kaggle 2×T4 (Free Tier)  
**Timeline:** 2 weeks

---

## Document Map

| # | Document | Purpose | Start Here If... |
|---|----------|---------|-----------------|
| 01 | [PRD — Product Requirements](./01_PRD.md) | What to build and why | You want the business case + success metrics |
| 02 | [TRD — Technical Requirements](./02_TRD.md) | How to build it; failure mitigations | You're about to write code |
| 03 | [UI/UX Design Brief](./03_UI_UX_Design_Brief.md) | Gradio demo + eval dashboard design | You're building the HF Spaces demo |
| 04 | [Backend Schema](./04_Backend_Schema.md) | Data models, config schema, eval schema | You're designing the data pipeline or logging |
| 05 | [Implementation Plan](./05_Implementation_Plan.md) | Day-by-day execution with actual code | You're starting the build today |

---

## Quick Reference

### Tech Stack
- `mistralai/Mistral-7B-Instruct-v0.3` (Apache 2.0)
- `peft` + `trl` + `bitsandbytes` (QLoRA)
- `datasets` (PubMedQA)
- `wandb` (experiment tracking)
- `gradio` on HF Spaces (demo)
- Kaggle 2×T4 (training)

### Key Numbers to Defend in Interviews
- **r=16** — rank choice balancing expressivity and adapter size
- **α=32** — lora_alpha = 2r; standard scaling
- **0.57%** — trainable parameters as fraction of total
- **41M** — trainable parameter count
- **max_seq_length=512** — covers 95th percentile of PubMedQA sample lengths
- **NF4** — optimal quantization for normally distributed weights (Dettmers 2023)

### Critical Pre-Start Setup
1. Kaggle Secrets: `WANDB_API_KEY`, `HUGGINGFACE_TOKEN`
2. HF Model Repo: `nikhilsh10/meditune-mistral-7b`
3. HF Space: `nikhilsh10/meditune`
4. Accept Mistral-7B license on HF Hub
5. W&B Project: `meditune-medical-qa`
