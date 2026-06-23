# Product Requirements Document (PRD)
## MediTune — QLoRA Fine-Tuning Pipeline for Medical Question Answering

**Version:** 1.0  
**Author:** Nikhil Sharma  
**Status:** Draft  
**Last Updated:** June 2026

---

## 1. Executive Summary

MediTune is an end-to-end QLoRA fine-tuning pipeline that adapts Mistral-7B-Instruct-v0.3 for closed-domain Medical Question Answering (MedQA). The project demonstrates mastery of Parameter-Efficient Fine-Tuning (PEFT) on consumer-grade hardware, reproducible ML experimentation, quantitative before/after evaluation, and model publishing — the four skills that distinguish a production ML engineer from someone who runs pre-built notebooks.

This is a **portfolio artifact**. Its primary consumers are technical interviewers and senior engineers evaluating depth. It must answer the implicit interview question: *"Can this person fine-tune an LLM with intent, not just by following a tutorial?"*

---

## 2. Problem Statement

### 2.1 The Real Problem (Justified)
General-purpose LLMs like Mistral-7B perform reasonably on broad knowledge tasks but degrade on narrow, evidence-grounded domains like clinical medicine. PubMedQA yes/no/maybe questions require the model to reason from provided abstracts — a specific retrieval-augmented reasoning behaviour that base instruction-following doesn't reliably exhibit.

### 2.2 The Portfolio Problem (Equally Important)
Fine-tuning is now a hiring signal. Interviewers asking "have you fine-tuned a model?" want to see:
- You understand the LoRA math (not just the API)
- You measured the effect (not just ran training)
- You know the failure modes (CUDA OOM, gradient overflow, catastrophic forgetting)
- You made deliberate hyperparameter choices and can defend them

MediTune is designed to give you a genuine, defensible answer to all four.

---

## 3. Goals

| # | Goal | Success Metric |
|---|------|---------------|
| G1 | Fine-tune Mistral-7B-Instruct-v0.3 on PubMedQA using QLoRA | Training run completes without OOM on Kaggle 2×T4 |
| G2 | Demonstrate measurable improvement over base model | ≥8% accuracy gain on PubMedQA test split (500 samples) |
| G3 | Publish model to Hugging Face Hub with full Model Card | Model publicly accessible at `nikhilsh10/meditune-mistral-7b` |
| G4 | Build an interactive Gradio demo on HF Spaces | Live demo URL in README |
| G5 | Log full training run to W&B with loss curves and eval metrics | Public W&B run link in README |
| G6 | Document architecture decisions and failure mitigations | Each key decision has a written rationale in the README |

---

## 4. Non-Goals

| # | Non-Goal | Reason |
|---|----------|--------|
| NG1 | Clinical deployment or real medical use | This is a portfolio project. Never imply it is production-safe medical software. |
| NG2 | Fine-tuning on proprietary medical records | No access, no IRB, unnecessary for the portfolio goal |
| NG3 | Matching GPT-4 or Claude performance | Not the benchmark. The benchmark is base Mistral-7B. |
| NG4 | Multi-GPU distributed training | Single-node QLoRA on 2×T4 is the constraint; distributed adds complexity without portfolio signal |
| NG5 | Real-time inference API (FastAPI/vLLM) | Out of scope for this project; the Fraud Detection project owns the MLOps/serving story |
| NG6 | Zero failures | Every ML training run encounters issues. The goal is pre-identified mitigations, not zero failures. |

---

## 5. Target Users

### Primary: Technical Interviewers / ML Hiring Managers
They will look at: the GitHub README, the W&B run, the HF Model Card, and then ask follow-up questions in the interview. They want to see deliberate choices, not copy-paste.

### Secondary: The Nikhil in 6 months
The implementation must be documented well enough that you can re-run, extend, or explain every part of it cold after 3 months away.

---

## 6. User Stories

| ID | As a... | I want to... | So that... |
|----|---------|-------------|-----------|
| US1 | Recruiter | See a live demo of the fine-tuned model answering medical questions | I can verify the project actually works |
| US2 | Technical Interviewer | See the W&B training curves and eval metrics | I can assess whether the author understands overfitting and hyperparameter tuning |
| US3 | ML Engineer reading the README | Understand why QLoRA was chosen over full fine-tuning | I can evaluate the author's depth of knowledge |
| US4 | Portfolio Visitor | Query the Gradio demo with a medical question | I can compare base model vs fine-tuned responses side by side |

---

## 7. Feature Requirements

### P0 — Must Have (MVP)

| ID | Feature | Acceptance Criteria |
|----|---------|-------------------|
| F1 | QLoRA Training Pipeline | Trains to completion on Kaggle free tier without OOM; final training loss < 1.5 |
| F2 | Dataset Preprocessing | PubMedQA long-form training split (211,269 samples) formatted to Alpaca-style instruction tuples |
| F3 | Before/After Evaluation | Accuracy on PubMedQA 500-sample test split measured for base AND fine-tuned model; delta reported |
| F4 | HF Hub Push | Merged LoRA adapter + base model pushed as a full model repo with Model Card |
| F5 | W&B Logging | Training loss, eval loss, learning rate logged per step; public run link |
| F6 | GitHub README | Architecture diagram, before/after eval table, W&B link, HF demo link, setup instructions |

### P1 — Should Have

| ID | Feature | Acceptance Criteria |
|----|---------|-------------------|
| F7 | Gradio Demo on HF Spaces | Side-by-side comparison: base Mistral-7B response vs MediTune response for same prompt |
| F8 | ROUGE-L Evaluation | ROUGE-L score on MedQuAD free-text answer subset (200 samples) |
| F9 | Training Config YAML | All hyperparameters in a version-controlled YAML; no hardcoded values in training script |

### P2 — Nice to Have

| ID | Feature | Acceptance Criteria |
|----|---------|-------------------|
| F10 | Perplexity Comparison | Perplexity on held-out medical text for base vs fine-tuned |
| F11 | LoRA Rank Ablation | Train with r=8 and r=32 in addition to r=16; compare eval accuracy |
| F12 | Quantized GGUF Export | Export fine-tuned model to GGUF Q4_K_M for local Ollama deployment |

---

## 8. Success Metrics

| Metric | Minimum Acceptable | Target |
|--------|-------------------|--------|
| PubMedQA Test Accuracy (base Mistral-7B) | Establish baseline | Expected ~55–65% |
| PubMedQA Test Accuracy (MediTune) | ≥ baseline + 8% | ≥ 70% |
| Training run wall-clock time | < 6 hours on 2×T4 | < 4 hours |
| HF Hub model downloads (30 days) | > 0 (public) | > 50 |
| W&B run is public and accessible | Required | Required |
| README completeness | Architecture, setup, results, links | + decision rationale |

---

## 9. Constraints

| Constraint | Detail |
|-----------|--------|
| Compute Budget | Free Kaggle GPU quota (30 hrs/week, 2×T4 16GB each) |
| Time Budget | 2 weeks end-to-end |
| Model License | Mistral-7B-v0.3 uses Apache 2.0; safe for public HF Hub push |
| Dataset License | PubMedQA: MIT license; MedQuAD: CC BY 4.0 |
| Hardware | Must run on a single Kaggle session (≤ 2×T4); no paid cloud |

---

## 10. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| CUDA OOM during training | High | High | Use gradient checkpointing + batch size 1 + 4-bit NF4; see TRD Section 5 |
| Base model accuracy higher than expected, delta too small | Medium | Medium | Evaluate on stricter MedMCQA if PubMedQA delta is < 5% |
| Catastrophic forgetting on general tasks | Low | Low | Evaluate on 50-sample MMLU subset post-fine-tuning; document in Model Card |
| HF Hub model too large to push | Medium | Low | Push only LoRA adapter weights (~300MB) if full merge is blocked |
| W&B API key management in Kaggle | Low | Medium | Use Kaggle Secrets for WANDB_API_KEY |

---

## 11. Out-of-Scope Interview Traps

When an interviewer asks about this project, these are the questions you must be ready for. Each is answered by a design decision in the TRD:

1. "Why QLoRA and not full fine-tuning?" → Memory budget; full FT needs ~160GB VRAM for 7B in bfloat16
2. "Why r=16 for LoRA rank?" → Balance between expressivity and parameter count; r=8 underfits on medical jargon, r=32 overfits on 211k samples
3. "How do you know the model didn't just memorize the training set?" → We evaluate on the held-out test split and measure ROUGE-L, not just accuracy
4. "What's catastrophic forgetting and does your model suffer from it?" → Covered in eval plan; we run MMLU sample post-training
5. "Why Mistral-7B and not LLaMA-3?" → Apache 2.0 license, better instruction-following at 7B scale per open evals
