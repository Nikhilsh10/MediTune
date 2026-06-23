# UI/UX Design Brief
## MediTune — Gradio Demo + Evaluation Dashboard

**Version:** 1.0  
**Author:** Nikhil Sharma  
**Target Deployment:** Hugging Face Spaces (Gradio)

---

## 1. Design Objective

MediTune's UI has one job: make the before/after improvement undeniable and memorable to a non-technical recruiter who spends 45 seconds on it. It is NOT a full product dashboard. It is a focused demo that shows the model works.

**Single Design Principle:** Every UI element should either show the comparison or explain why it matters. Nothing else.

---

## 2. Surfaces

### Surface 1: HF Spaces Gradio Demo (`app.py`)
The primary public face of the project. Hosted at `huggingface.co/spaces/nikhilsh10/meditune`.

### Surface 2: W&B Training Dashboard  
Managed by W&B; no custom UI needed. Ensure the run is public and the project name is `meditune-medical-qa`.

### Surface 3: GitHub README Eval Table  
Static markdown table showing before/after metrics. Not interactive but must be visually scannable.

---

## 3. Gradio Demo — Information Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  🩺 MediTune — Fine-tuned Medical QA                         │
│  Mistral-7B-Instruct + QLoRA on PubMedQA                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [CONTEXT]                                                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Paste or select a PubMed abstract here...             │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  [QUESTION]                                                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Type a yes/no/maybe medical question...               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  [▶ Run Comparison]    [🔄 Load Example]                     │
│                                                              │
├──────────────────┬───────────────────────────────────────────┤
│  BASE MODEL      │  MEDITUNE (Fine-tuned)                    │
│  Mistral-7B      │  Mistral-7B + QLoRA                       │
│  ─────────────── │  ──────────────────────────────────────   │
│  [Response text] │  [Response text]                          │
│                  │                                            │
│                  │  ✅ Decision: YES                          │
└──────────────────┴───────────────────────────────────────────┘

  📊 Model Stats:  r=16 | α=32 | Params: 41M / 7.24B (0.57%)
  📈 PubMedQA Acc: Base 61.4% → MediTune 72.8% (+11.4%)
```

---

## 4. Visual Design Spec

### 4.1 Color Palette

Use Gradio's dark theme as the base. Override with CSS for key elements only.

| Element | Color | Hex |
|---------|-------|-----|
| Page background | Dark graphite | `#0d1117` (matches GitHub dark) |
| Card background | Elevated surface | `#161b22` |
| Primary accent | Medical teal | `#0ea5e9` |
| Fine-tuned column highlight | Subtle green tint | `rgba(34, 197, 94, 0.08)` border |
| Base model column | Neutral | No tint |
| Text primary | Off-white | `#e6edf3` |
| Text secondary | Muted | `#8b949e` |
| Decision badge (yes) | Green | `#22c55e` |
| Decision badge (no) | Red | `#ef4444` |
| Decision badge (maybe) | Amber | `#f59e0b` |

**Rationale:** Medical teal is the standard color language for health tech. Dark theme matches HF Spaces default and the portfolio's Graphite & Ember design system.

### 4.2 Typography

| Element | Font | Size | Weight |
|---------|------|------|--------|
| Header (MediTune) | System sans-serif | 24px | 700 |
| Section labels | System sans-serif | 13px | 600, uppercase, letter-spacing 0.08em |
| Response text | Monospace (JetBrains Mono fallback) | 14px | 400 |
| Stats row | System sans-serif | 12px | 400 |

### 4.3 Layout Constraints

- **Max width:** 960px, centered
- **Column split:** 50/50 for base vs fine-tuned panel
- **Mobile:** Stack panels vertically; fine-tuned panel goes on top
- **No scrolling required** above the comparison: prompt inputs + Run button must be above the fold on 1080p

---

## 5. Component Specifications

### 5.1 Context Input (`gr.Textbox`)
```python
gr.Textbox(
    label="PubMed Abstract (Context)",
    placeholder="Paste a PubMed abstract here, or click 'Load Example'...",
    lines=6,
    max_lines=10,
    elem_id="context-input"
)
```

### 5.2 Question Input (`gr.Textbox`)
```python
gr.Textbox(
    label="Clinical Question",
    placeholder="e.g., Does the intervention significantly reduce mortality?",
    lines=2,
    elem_id="question-input"
)
```

### 5.3 Run Comparison Button
```python
gr.Button(
    "▶  Run Comparison",
    variant="primary",
    size="lg",
    elem_id="run-btn"
)
```
**State behavior:** Button must show loading spinner during inference. Use `gr.Button` with interactive=False during generation to prevent double-submission.

### 5.4 Response Panels (`gr.Textbox`, read-only)
Two side-by-side panels. Fine-tuned panel has a green left border (CSS):
```css
#finetuned-panel {
    border-left: 3px solid #22c55e;
}
```

### 5.5 Decision Badge
Extract `yes`/`no`/`maybe` from fine-tuned response and render as a colored `gr.HTML` badge:
```python
def get_decision_badge(text):
    decision = extract_decision(text)
    colors = {"yes": "#22c55e", "no": "#ef4444", "maybe": "#f59e0b"}
    color = colors.get(decision, "#6b7280")
    return f'<span style="background:{color};color:#fff;padding:4px 12px;border-radius:9999px;font-weight:700;font-size:14px;">Decision: {decision.upper()}</span>'
```

### 5.6 Example Loader
Pre-load 5 diverse PubMedQA examples (covering yes/no/maybe decisions). Use `gr.Examples`:
```python
examples = [
    [EXAMPLE_CONTEXTS[0], EXAMPLE_QUESTIONS[0]],
    [EXAMPLE_CONTEXTS[1], EXAMPLE_QUESTIONS[1]],
    # ... 5 total
]
gr.Examples(examples=examples, inputs=[context_input, question_input])
```

**Selection criteria for examples:**  
Pick examples where the base model gets it wrong (answers "maybe" when ground truth is "yes") and the fine-tuned model gets it right. This makes the demo compelling.

---

## 6. Stats Footer

Static HTML row below the comparison panels. Update values after your training run:
```html
<div style="display:flex;gap:24px;margin-top:16px;padding:12px 16px;
            background:#161b22;border-radius:8px;font-size:12px;color:#8b949e;">
  <span>🧠 <strong style="color:#e6edf3">LoRA Rank:</strong> 16 | α=32</span>
  <span>⚙️ <strong style="color:#e6edf3">Trainable:</strong> 41M / 7.24B params (0.57%)</span>
  <span>📊 <strong style="color:#e6edf3">PubMedQA Acc:</strong> 
        <span style="color:#8b949e">Base 61.4%</span> → 
        <span style="color:#22c55e;font-weight:700">MediTune 72.8%</span>
        <span style="color:#22c55e">(+11.4%)</span>
  </span>
  <span>🔗 <a href="https://wandb.ai/..." style="color:#0ea5e9">W&B Run</a></span>
</div>
```
*(Replace accuracy numbers with your actual results)*

---

## 7. Loading States & Error Handling

| State | Behavior |
|-------|----------|
| Button clicked, inference running | Button disabled + "Generating..." label; both output panels show "⏳ Generating..." |
| Inference complete | Both panels populated; decision badge appears |
| Empty inputs | Button click shows `gr.Warning("Please provide both a context and a question.")` |
| Model loading (cold start on HF Spaces) | Show `gr.Info("Model loading on first request (~30s). Subsequent requests are faster.")` |
| Inference timeout (>60s) | Catch exception; show `gr.Error("Inference timed out. Try a shorter context.")` |

---

## 8. README Eval Table (GitHub)

Place this table in the `## Results` section of the README. Update with your real numbers:

```markdown
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
```

*(All numbers above are illustrative targets. Replace with actual results.)*

---

## 9. HF Spaces Configuration (`README.md` header)

```yaml
---
title: MediTune Medical QA
emoji: 🩺
colorFrom: teal
colorTo: blue
sdk: gradio
sdk_version: 4.36.0
app_file: app.py
pinned: false
license: apache-2.0
models:
  - nikhilsh10/meditune-mistral-7b
datasets:
  - qiaojin/PubMedQA
tags:
  - medical
  - question-answering
  - qlora
  - fine-tuning
  - peft
  - mistral
---
```

---

## 10. What NOT to Build

- ❌ No authentication / user management
- ❌ No history / session persistence
- ❌ No streaming token output (adds complexity; the demo doesn't need it)
- ❌ No full-context RAG pipeline in the demo (that's a separate project)
- ❌ No mobile app
- ❌ No custom backend server; Gradio on HF Spaces is the server
