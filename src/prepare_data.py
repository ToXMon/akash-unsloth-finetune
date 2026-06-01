"""
Dataset preparation for Akash Unsloth Fine-tune Demo.

Downloads dataset from HuggingFace, subsets to configured size,
converts to Alpaca format, validates, and saves as JSONL.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("finetune.prepare")

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


def download_dataset(
    dataset_name: str,
    split: str = "train",
    max_retries: int = MAX_RETRIES,
) -> Any:
    """Download dataset from HuggingFace with retry logic."""
    from datasets import load_dataset

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Downloading dataset '%s' (split=%s), attempt %d/%d", dataset_name, split, attempt, max_retries)
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
            logger.info("Download complete: %d samples", len(dataset))
            return dataset
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = RETRY_DELAY_SECONDS * attempt
                logger.warning("Download failed (attempt %d): %s. Retrying in %ds...", attempt, e, wait)
                time.sleep(wait)
            else:
                logger.error("Download failed after %d attempts: %s", max_retries, e)

    raise RuntimeError(f"Failed to download dataset '{dataset_name}' after {max_retries} attempts: {last_error}")


def subset_dataset(dataset: Any, size: int, seed: int = 42) -> List[Dict]:
    """Select a random subset of the dataset."""
    total = len(dataset)
    if size >= total:
        logger.info("Requested size %d >= dataset size %d, using full dataset", size, total)
        return [dict(row) for row in dataset]

    logger.info("Subsetting from %d to %d samples (seed=%d)", total, size, seed)
    shuffled = dataset.shuffle(seed=seed)
    return [dict(shuffled[i]) for i in range(size)]


def convert_to_alpaca(sample: Dict, domain: str = "medical_qa") -> Optional[Dict]:
    """
    Convert a raw dataset sample to Alpaca format.
    Alpaca format: {"instruction": str, "input": str, "output": str}

    OlympiadBench has fields like 'question', 'answer', 'solution', etc.
    This handles common field name variations.
    """
    # Try to extract the question/instruction
    instruction = (
        sample.get("question")
        or sample.get("prompt")
        or sample.get("instruction")
        or sample.get("problem")
        or sample.get("input")
        or ""
    )

    # Try to extract context/input
    input_text = (
        sample.get("context")
        or sample.get("background")
        or ""
    )

    # Try to extract the answer/output
    output = (
        sample.get("answer")
        or sample.get("response")
        or sample.get("output")
        or sample.get("solution")
        or ""
    )

    # Handle nested answer fields (OlympiadBench specific)
    if not output and "final_answer" in sample:
        final = sample["final_answer"]
        if isinstance(final, list):
            output = "\n".join(str(f) for f in final)
        else:
            output = str(final)

    if not output and "answer_units" in sample:
        output = str(sample.get("answer_units", ""))

    # Clean up text
    if isinstance(instruction, list):
        instruction = "\n".join(str(i) for i in instruction)
    if isinstance(output, list):
        output = "\n".join(str(o) for o in output)

    instruction = str(instruction).strip()
    output = str(output).strip()

    return {
        "instruction": instruction,
        "input": str(input_text).strip() if input_text else "",
        "output": output,
    }


def validate_sample(sample: Dict) -> bool:
    """Validate that a sample has required fields with non-empty text."""
    if not all(k in sample for k in ("instruction", "input", "output")):
        return False
    if not sample["instruction"] or len(sample["instruction"].strip()) < 5:
        return False
    if not sample["output"] or len(sample["output"].strip()) < 2:
        return False
    return True


def save_jsonl(samples: List[Dict], path: str) -> None:
    """Save samples as JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(output_path), "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info("Saved %d samples to %s", len(samples), path)


def log_stats(total: int, valid: int, filtered: int, samples: List[Dict]) -> None:
    """Log dataset statistics."""
    logger.info("=" * 50)
    logger.info("Dataset Statistics")
    logger.info("=" * 50)
    logger.info("Total raw samples:    %d", total)
    logger.info("Valid samples:        %d", valid)
    logger.info("Filtered out:         %d", filtered)
    logger.info("Filter rate:          %.1f%%", (filtered / total * 100) if total > 0 else 0)

    if samples:
        avg_instr = sum(len(s["instruction"]) for s in samples) / len(samples)
        avg_output = sum(len(s["output"]) for s in samples) / len(samples)
        logger.info("Avg instruction len:  %.0f chars", avg_instr)
        logger.info("Avg output len:       %.0f chars", avg_output)

        # Rough token estimate (~4 chars per token)
        avg_tokens = (avg_instr + avg_output) / 4
        logger.info("Est. avg tokens:      %.0f", avg_tokens)

    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning dataset")
    parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name")
    parser.add_argument("--subset-size", type=int, default=None, help="Number of samples to use")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--domain", type=str, default=None, help="Domain label")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to use (e.g. train, test_en, test_cn)")
    args = parser.parse_args()

    # Load config
    config_path = os.environ.get("CONFIG_PATH", "configs/train_config.yaml")
    from src.utils import load_config
    config = load_config(config_path)

    dataset_name = args.dataset or os.environ.get("DATASET_NAME", config["dataset"]["source"])
    subset_size = args.subset_size or int(os.environ.get("SUBSET_SIZE", config["dataset"]["subset_size"]))
    output_path = args.output or os.environ.get("OUTPUT_PATH", config["paths"]["train_file"])
    seed = args.seed or config["dataset"].get("seed", 42)
    domain = args.domain or config["dataset"].get("domain", "medical_qa")
    split = args.split or os.environ.get("DATA_SPLIT", config["dataset"].get("train_split", "train"))

    logger.info("Preparing dataset: %s (split=%s, subset=%d, domain=%s)", dataset_name, split, subset_size, domain)

    # Download
    dataset = download_dataset(dataset_name, split=split)

    # Subset
    raw_samples = subset_dataset(dataset, subset_size, seed=seed)

    # Convert and validate
    alpaca_samples = []
    filtered = 0
    for sample in raw_samples:
        converted = convert_to_alpaca(sample, domain=domain)
        if converted and validate_sample(converted):
            alpaca_samples.append(converted)
        else:
            filtered += 1

    total = len(raw_samples)
    valid = len(alpaca_samples)

    # Log stats
    log_stats(total, valid, filtered, alpaca_samples)

    if valid == 0:
        logger.error("No valid samples after conversion! Check dataset format.")
        sys.exit(1)

    # Save
    save_jsonl(alpaca_samples, output_path)
    logger.info("Dataset preparation complete: %d samples saved to %s", valid, output_path)


if __name__ == "__main__":
    main()
