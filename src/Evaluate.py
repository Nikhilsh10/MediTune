"""
src/evaluate.py

Evaluation module for MediTune. Three evaluation functions:

  1. evaluate_pubmedqa()  — accuracy on PubMedQA yes/no/maybe classification
  2. evaluate_rouge_l()   — ROUGE-L on MedQuAD free-text answers (requires
                            the MedQuAD JSONL to be pre-downloaded)
  3. evaluate_mmlu()      — MMLU college_medicine accuracy (catastrophic-
                            forgetting sanity check)

Designed to be called from both notebooks and src/train.py post-training.

Usage:
    from src.evaluate import evaluate_pubmedqa
    results = evaluate_pubmedqa(model, tokenizer, n_samples=500)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Utils import build_inference_prompt, extract_decision

# ---------------------------------------------------------------------------
# 1. PubMedQA accuracy
# ---------------------------------------------------------------------------


def evaluate_pubmedqa(
    model,
    tokenizer,
    n_samples: int = 500,
    dataset_name: str = "qiaojin/PubMedQA",
    dataset_config: str = "pqa_labeled",
    split: str = "train",
    max_new_tokens: int = 100,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate model accuracy on PubMedQA yes/no/maybe classification.

    Args:
        model:          A loaded (and optionally PEFT-wrapped) causal LM.
        tokenizer:      Corresponding tokenizer with pad_token set.
        n_samples:      Number of samples to evaluate (≤ dataset size).
        dataset_name:   HF dataset identifier.
        dataset_config: 'pqa_labeled' for the 1k labeled set.
        split:          Dataset split to load.
        max_new_tokens: Max tokens to generate per sample.
        verbose:        Print running accuracy every 50 samples.

    Returns:
        dict with keys:
            accuracy, correct, total, unknown_predictions,
            per_label_accuracy, per_label_counts
    """
    import torch
    from datasets import load_dataset

    ds = load_dataset(dataset_name, dataset_config, split=split)
    # Filter to valid labels only before slicing
    ds = ds.filter(lambda s: s.get("final_decision") in ("yes", "no", "maybe"))
    n_samples = min(n_samples, len(ds))
    test_samples = ds.select(range(n_samples))

    per_label: dict[str, dict[str, int]] = {
        "yes":   {"correct": 0, "total": 0},
        "no":    {"correct": 0, "total": 0},
        "maybe": {"correct": 0, "total": 0},
    }
    correct = 0
    unknowns = 0

    model.eval()

    for i, sample in enumerate(test_samples):
        abstract: str = " ".join(sample["context"]["contexts"])
        prompt: str = build_inference_prompt(abstract, sample["question"])
        gt: str = sample["final_decision"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response: str = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        pred: str | None = extract_decision(response)

        per_label[gt]["total"] += 1

        if pred == gt:
            correct += 1
            per_label[gt]["correct"] += 1

        if pred is None:
            unknowns += 1

        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{n_samples}] running accuracy = {correct / (i + 1):.4f}")

    accuracy = correct / n_samples
    per_label_accuracy = {
        label: (v["correct"] / v["total"] if v["total"] > 0 else 0.0)
        for label, v in per_label.items()
    }

    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": n_samples,
        "unknown_predictions": unknowns,
        "per_label_accuracy": per_label_accuracy,
        "per_label_counts": per_label,
    }

    if verbose:
        print(f"\n  Final accuracy : {accuracy:.4f}  ({correct}/{n_samples})")
        print(f"  Per-label      : {per_label_accuracy}")
        print(f"  Unknowns       : {unknowns}")

    return results


# ---------------------------------------------------------------------------
# 2. ROUGE-L (MedQuAD)
# ---------------------------------------------------------------------------


def evaluate_rouge_l(
    model,
    tokenizer,
    medquad_jsonl_path: str,
    n_samples: int = 200,
    max_new_tokens: int = 150,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compute ROUGE-L on MedQuAD free-text QA pairs.

    MedQuAD must be pre-converted to a JSONL file where each line is:
        {"context": "...", "question": "...", "answer": "..."}

    Args:
        model:               Loaded causal LM.
        tokenizer:           Tokenizer with pad_token set.
        medquad_jsonl_path:  Path to the prepared MedQuAD JSONL.
        n_samples:           Number of samples to evaluate.
        max_new_tokens:      Max tokens to generate per sample.
        verbose:             Print progress every 50 samples.

    Returns:
        dict with keys: rouge_l_mean, rouge_l_scores, n_samples
    """
    import torch

    try:
        import evaluate as hf_evaluate
    except ImportError:
        raise ImportError(
            "The 'evaluate' library is required for ROUGE-L. "
            "Install it with: pip install evaluate rouge_score"
        )

    rouge = hf_evaluate.load("rouge")

    # Load MedQuAD JSONL
    records: list[dict] = []
    with open(medquad_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records = records[:n_samples]

    predictions: list[str] = []
    references: list[str] = []

    model.eval()

    for i, record in enumerate(records):
        prompt = build_inference_prompt(record["context"], record["question"])

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        predictions.append(prediction)
        references.append(record["answer"])

        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{n_samples}] samples processed")

    scores = rouge.compute(predictions=predictions, references=references)
    rouge_l_mean: float = scores["rougeL"]

    results = {
        "rouge_l_mean": rouge_l_mean,
        "rouge_l_scores": scores,
        "n_samples": len(records),
    }

    if verbose:
        print(f"\n  ROUGE-L mean: {rouge_l_mean:.4f}")

    return results


# ---------------------------------------------------------------------------
# 3. MMLU catastrophic-forgetting check
# ---------------------------------------------------------------------------


def evaluate_mmlu(
    model,
    tokenizer,
    n_samples: int = 50,
    subject: str = "college_medicine",
    verbose: bool = True,
) -> dict[str, Any]:
    """Run MMLU multiple-choice eval on a subject as a forgetting guard.

    Evaluates whether fine-tuning degraded general reasoning ability.
    A delta of > -10% vs the base model is flagged as catastrophic forgetting.

    Args:
        model:     Loaded causal LM.
        tokenizer: Tokenizer with pad_token set.
        n_samples: Number of MMLU questions to evaluate.
        subject:   MMLU subject (default: 'college_medicine').
        verbose:   Print progress.

    Returns:
        dict with keys: accuracy, correct, total
    """
    import torch
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", subject, split="test")
    n_samples = min(n_samples, len(ds))
    test_samples = ds.select(range(n_samples))

    choices_labels = ["A", "B", "C", "D"]
    correct = 0

    model.eval()

    for i, sample in enumerate(test_samples):
        choices_text = "\n".join(
            f"{label}. {text}"
            for label, text in zip(choices_labels, sample["choices"])
        )
        prompt = (
            f"The following is a multiple-choice question about {subject.replace('_', ' ')}.\n\n"
            f"Question: {sample['question']}\n\n"
            f"{choices_text}\n\n"
            f"Answer (A, B, C, or D):"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True).strip().upper()

        # Extract first letter
        pred_letter = next(
            (c for c in response if c in choices_labels), None
        )
        gt_letter = choices_labels[sample["answer"]]

        if pred_letter == gt_letter:
            correct += 1

        if verbose and (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{n_samples}] running accuracy = {correct / (i + 1):.4f}")

    accuracy = correct / n_samples

    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": n_samples,
    }

    if verbose:
        print(f"\n  MMLU {subject} accuracy: {accuracy:.4f}  ({correct}/{n_samples})")

    return results


# ---------------------------------------------------------------------------
# Save results helper
# ---------------------------------------------------------------------------


def save_eval_results(
    results: dict[str, Any],
    output_path: str = "outputs/eval_results.json",
) -> None:
    """Write the full evaluation results dict to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved eval results → {output_path}")