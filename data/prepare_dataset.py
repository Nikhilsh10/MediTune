"""
data/prepare_dataset.py

Downloads PubMedQA from HuggingFace Hub, applies Alpaca-style formatting,
deduplicates by pubid, filters by token length, and writes:

  data/train.jsonl  — from pqa_artificial (default 50 000 samples)
  data/eval.jsonl   — from pqa_labeled (default 500 samples)

Run from the repo root:
    python data/prepare_dataset.py
    python data/prepare_dataset.py --max-train 10000 --max-eval 200   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# Allow running as a script from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.Utils import build_instruction_prompt, is_valid_sample


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_NAME = "qiaojin/PubMedQA"
TRAIN_CONFIG = "pqa_artificial"
EVAL_CONFIG = "pqa_labeled"
DEFAULT_MAX_TRAIN = 50_000
DEFAULT_MAX_EVAL = 500
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

OUT_DIR = Path(__file__).resolve().parent  # data/
TRAIN_PATH = OUT_DIR / "train.jsonl"
EVAL_PATH = OUT_DIR / "eval.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_tokens(text: str, tokenizer) -> int:
    """Return the number of tokens in *text* without adding special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def assert_label_distribution(records: list[dict], split_name: str) -> None:
    """Warn if any label represents < 5 % of the split."""
    counts = Counter(r["decision"] for r in records)
    total = sum(counts.values())
    print(f"\n  Label distribution for '{split_name}':")
    for label in ("yes", "no", "maybe"):
        pct = counts.get(label, 0) / total * 100
        flag = " ⚠️  < 5%" if pct < 5 else ""
        print(f"    {label:6s}: {counts.get(label, 0):6d}  ({pct:.1f}%){flag}")


def write_jsonl(
    dataset,
    output_path: Path,
    tokenizer,
    max_samples: int | None,
    max_seq_length: int,
    split_name: str,
) -> list[dict]:
    """Format, filter, and write one split to a JSONL file.

    Returns the list of written records (for post-hoc distribution checks).
    """
    seen_ids: set[str] = set()
    written: list[dict] = []
    skipped_invalid = 0
    skipped_dup = 0
    skipped_long = 0

    iterable = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in iterable:
            # ── 1. Quality filter ──────────────────────────────────────────
            if not is_valid_sample(sample):
                skipped_invalid += 1
                continue

            # ── 2. Deduplication ───────────────────────────────────────────
            pubid = str(sample.get("pubid", ""))
            if pubid and pubid in seen_ids:
                skipped_dup += 1
                continue
            if pubid:
                seen_ids.add(pubid)

            # ── 3. Format ──────────────────────────────────────────────────
            text = build_instruction_prompt(sample)

            # ── 4. Token length filter ────────────────────────────────────
            token_count = count_tokens(text, tokenizer)
            if token_count > max_seq_length:
                skipped_long += 1
                continue

            # ── 5. Write ───────────────────────────────────────────────────
            record: dict = {
                "pubid": pubid,
                "text": text,
                "decision": sample["final_decision"],
                "token_count": token_count,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written.append(record)

    print(
        f"\n  {split_name}: written={len(written)}, "
        f"skipped_invalid={skipped_invalid}, "
        f"skipped_dup={skipped_dup}, "
        f"skipped_long={skipped_long}"
    )
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def prepare(
    max_train: int = DEFAULT_MAX_TRAIN,
    max_eval: int = DEFAULT_MAX_EVAL,
    max_seq_length: int = DEFAULT_MAX_SEQ_LEN,
    model_id: str = DEFAULT_MODEL_ID,
    train_out: Path = TRAIN_PATH,
    eval_out: Path = EVAL_PATH,
) -> None:
    # Lazy imports — not needed for syntax checks on other modules.
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Training split (pqa_artificial) ───────────────────────────────────
    print(f"\nLoading training data ({TRAIN_CONFIG})…")
    ds_train = load_dataset(DATASET_NAME, TRAIN_CONFIG, split="train")
    print(f"  Raw samples: {len(ds_train)}")

    train_records = write_jsonl(
        dataset=ds_train,
        output_path=train_out,
        tokenizer=tokenizer,
        max_samples=max_train,
        max_seq_length=max_seq_length,
        split_name="train",
    )
    assert_label_distribution(train_records, "train")

    # ── Eval split (pqa_labeled) ──────────────────────────────────────────
    print(f"\nLoading eval data ({EVAL_CONFIG})…")
    ds_eval = load_dataset(DATASET_NAME, EVAL_CONFIG, split="train")
    print(f"  Raw samples: {len(ds_eval)}")

    eval_records = write_jsonl(
        dataset=ds_eval,
        output_path=eval_out,
        tokenizer=tokenizer,
        max_samples=max_eval,
        max_seq_length=max_seq_length,
        split_name="eval",
    )
    assert_label_distribution(eval_records, "eval")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Done.")
    print(f"  Train JSONL : {train_out}  ({len(train_records)} samples)")
    print(f"  Eval  JSONL : {eval_out}  ({len(eval_records)} samples)")

    if len(train_records) < 1000:
        print(
            "\n⚠️  WARNING: fewer than 1 000 training samples written. "
            "Check filtering logic or raise --max-train."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PubMedQA training/eval JSONL files.")
    parser.add_argument("--max-train", type=int, default=DEFAULT_MAX_TRAIN,
                        help=f"Max training samples (default: {DEFAULT_MAX_TRAIN})")
    parser.add_argument("--max-eval", type=int, default=DEFAULT_MAX_EVAL,
                        help=f"Max eval samples (default: {DEFAULT_MAX_EVAL})")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LEN,
                        help=f"Token length ceiling (default: {DEFAULT_MAX_SEQ_LEN})")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                        help="HF tokenizer to measure lengths with")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare(
        max_train=args.max_train,
        max_eval=args.max_eval,
        max_seq_length=args.max_seq_length,
        model_id=args.model_id,
    )