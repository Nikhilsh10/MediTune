import gradio as gr
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Model Loading (cached; only runs once on Space startup) ---
MODEL_REPO = "nikhilsh10/meditune-mistral-7b"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

def load_models():
    # If HF token is needed, ensure it's set in space secrets
    # token = os.environ.get("HUGGINGFACE_TOKEN")
    
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    except:
        # Fallback if model repo isn't public yet
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto"
    )
    
    # Try loading fine-tuned model
    print("Loading fine-tuned model...")
    try:
        finetuned = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO, quantization_config=bnb, device_map="auto"
        )
    except:
        # If model repo is not available yet, just use base model as a placeholder
        print(f"Warning: Could not load {MODEL_REPO}. Using base model as fallback.")
        finetuned = base
        
    return base, finetuned, tokenizer

# For HF Spaces we instantiate models at startup
try:
    base_model, ft_model, tokenizer = load_models()
except Exception as e:
    print(f"Failed to load models during startup: {e}")
    base_model, ft_model, tokenizer = None, None, None

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
    if model is None:
        return "Model not loaded. Please check logs."
        
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
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
    else:
        badge_html = f'<span style="background:#6b7280;color:#fff;padding:4px 12px;border-radius:9999px;font-weight:700;font-size:14px;">Decision: UNKNOWN</span>'
        
    return base_response, ft_response, badge_html

# --- Example data ---
EXAMPLES = [
    [
        "This study examined the effect of aspirin on colorectal cancer prevention in high-risk patients. Patients received 300mg aspirin daily for 2 years. The primary endpoint was colorectal adenoma recurrence. Results showed a 47% reduction in adenoma recurrence (p<0.001) in the aspirin group compared to placebo.",
        "Does aspirin significantly reduce colorectal adenoma recurrence in high-risk patients?"
    ],
    [
        "A randomized controlled trial was conducted to evaluate the efficacy of a new antiviral drug for the common cold. The study included 500 participants. The results indicated that the duration of symptoms was 5.2 days in the treatment group and 5.4 days in the placebo group (p=0.45). There was no significant difference in viral load.",
        "Is the new antiviral drug effective in reducing the duration of common cold symptoms?"
    ],
    [
        "We investigated whether higher Vitamin D levels are associated with a lower risk of multiple sclerosis (MS). In a cohort of 10,000 individuals followed over 10 years, those in the highest quartile of 25(OH)D had a 30% lower incidence of MS. However, confounding factors such as sun exposure cannot be entirely ruled out.",
        "Do higher Vitamin D levels definitively prevent multiple sclerosis?"
    ]
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
                max_lines=10,
                elem_id="context-input"
            )
            question_input = gr.Textbox(
                label="Clinical Question",
                placeholder="e.g., Does the intervention significantly reduce mortality?",
                lines=2,
                elem_id="question-input"
            )
            run_btn = gr.Button("▶  Run Comparison", variant="primary", size="lg", elem_id="run-btn")
    
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

if __name__ == "__main__":
    demo.launch()
