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

def extract_decision(text: str) -> str:
    """Extract yes/no/maybe from model output."""
    text = text.lower().strip()
    for label in ["yes", "no", "maybe"]:
        if text.startswith(label):
            return label
    # Fallback: find first occurrence
    for label in ["yes", "no", "maybe"]:
        if label in text[:100]:
            return label
    return "unknown"
