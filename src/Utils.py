"""
Shared utilities: prompt builders, decision extractor, label validator.

Imported by train.py, evaluate.py, inference.py, and prepare_dataset.py.
Keep this module free of heavy imports (no torch, no transformers) so it
loads instantly and is testable without a GPU environment.
"""

from __future__ import annotations

# Labels accepted by the model — used for validation and extraction
VALID_LABELS = ("yes", "no", "maybe")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_INSTRUCTION = (
    "You are a medical expert. Based ONLY on the provided context, answer the question "
    "with 'yes', 'no', or 'maybe'. Then briefly explain your reasoning in 1-2 sentences."
)


def build_instruction_prompt(sample: dict) -> str:
    """Format a PubMedQA sample into an Alpaca-style instruction string for SFT.

    The '### Response:' section includes both the decision label and the
    long_answer explanation so the model learns to generate structured output.

    Args:
        sample: A single PubMedQA dataset row with keys:
                'context' (dict with 'contexts' list), 'question',
                'final_decision', 'long_answer'.

    Returns:
        A fully formatted instruction string ready for tokenization.
    """
    abstract: str = " ".join(sample["context"]["contexts"])
    question: str = sample["question"].strip()
    decision: str = sample["final_decision"].strip().lower()
    long_answer: str = (sample.get("long_answer") or "").strip()

    # Capitalise first letter of explanation if present
    explanation = long_answer[0].upper() + long_answer[1:] if long_answer else ""
    response_text = f"{decision}. {explanation}" if explanation else decision

    return (
        f"<s>[INST] {_INSTRUCTION}\n\n"
        f"Context:\n{abstract}\n\n"
        f"Question:\n{question} [/INST] "
        f"{response_text}</s>"
    )


def build_inference_prompt(context: str, question: str) -> str:
    """Build a prompt for inference — no Response field.

    The model must generate the Response section from scratch.

    Args:
        context: PubMed abstract or any relevant passage.
        question: A yes/no/maybe clinical question.

    Returns:
        Prompt string ending with '### Response:\\n' for the model to continue.
    """
    context = context.strip()
    question = question.strip()

    return (
        f"<s>[INST] {_INSTRUCTION}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question} [/INST]"
    )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
import re


def extract_decision(text: str) -> str | None:
    """Extract 'yes', 'no', or 'maybe' from model output text.

    Strategy:
      1. Check if the stripped output starts with a valid label (most reliable).
      2. Scan the first 100 characters for any label occurrence.
      3. Return None if nothing is found (logged as 'unknown' by the caller).

    Args:
        text: Raw decoded model output (new tokens only, not the prompt).

    Returns:
        One of 'yes', 'no', 'maybe', or None.
    """
    normalised = text.lower().strip()

    # Pass 1: starts-with (highest confidence)
    for label in VALID_LABELS:
        if normalised.startswith(label):
            return label

    # Pass 2: first occurrence within first 100 chars (using regex word boundaries)
    window = normalised[:100]
    for label in VALID_LABELS:
        if re.search(rf"\b{label}\b", window):
            return label

    return None


# ---------------------------------------------------------------------------
# Data validation helper
# ---------------------------------------------------------------------------


def is_valid_sample(sample: dict, max_token_count: int | None = None) -> bool:
    """Return True if a PubMedQA sample passes all quality filters.

    Filters:
    - final_decision must be in VALID_LABELS
    - long_answer must be a non-empty string (> 10 chars)
    - context.contexts must be a non-empty list

    Args:
        sample:          Raw PubMedQA dataset row.
        max_token_count: If provided, sample['token_count'] must be <= this value.
                         Only applies after token_count has been computed.

    Returns:
        True if the sample should be kept, False otherwise.
    """
    if sample.get("final_decision") not in VALID_LABELS:
        return False

    long_answer = sample.get("long_answer") or ""
    if len(long_answer.strip()) < 10:
        return False

    contexts = (sample.get("context") or {}).get("contexts")
    if not contexts or len(contexts) == 0:
        return False

    if max_token_count is not None:
        if sample.get("token_count", 0) > max_token_count:
            return False

    return True