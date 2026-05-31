"""
Core training script for Akash Unsloth Fine-tune Demo.

Loads base model with Unsloth FastLanguageModel, applies QLoRA config,
trains on prepared dataset, saves LoRA adapters and optionally merged model.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("finetune.train")


def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    """Load training configuration from YAML."""
    from src.utils import load_config as _load
    return _load(config_path)


def load_model_and_tokenizer(config: dict):
    """Load base model and tokenizer using Unsloth's FastLanguageModel."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("Unsloth not installed. Install with: pip install unsloth")
        sys.exit(1)

    model_cfg = config["model"]
    logger.info("Loading base model: %s", model_cfg["base_model"])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model"],
        max_seq_length=model_cfg.get("max_seq_length", 2048),
        dtype=model_cfg.get("dtype"),
        load_in_4bit=model_cfg.get("load_in_4bit", True),
    )

    logger.info("Model loaded: %s params", f"{sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def apply_lora(model, config: dict):
    """Apply LoRA adapters to the model."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        from peft import LoraConfig, get_peft_model

    lora_cfg = config["lora"]
    logger.info(
        "Applying LoRA: rank=%d, alpha=%d, dropout=%.3f",
        lora_cfg["rank"], lora_cfg["alpha"], lora_cfg["dropout"],
    )

    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            use_rslora=lora_cfg.get("use_rslora", False),
            use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", True),
        )
    except (NameError, AttributeError):
        # Fallback to PEFT directly
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %s / %s (%.2f%%)",
                f"{trainable:,}", f"{total:,}", 100 * trainable / total)
    return model


def load_dataset(config: dict):
    """Load and format the prepared training dataset."""
    from datasets import Dataset
    from src.utils import load_config as _load

    data_path = config["paths"]["train_file"]
    logger.info("Loading training data from %s", data_path)

    if not os.path.exists(data_path):
        logger.error("Training data not found: %s. Run prepare_data.py first.", data_path)
        sys.exit(1)

    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    logger.info("Loaded %d samples", len(samples))

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(samples)

    # Apply Alpaca prompt formatting
    def format_alpaca(example):
        if example.get("input") and example["input"].strip():
            text = (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                f"### Response:\n{example['output']}"
            )
        else:
            text = (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Response:\n{example['output']}"
            )
        return {"text": text}

    dataset = dataset.map(format_alpaca)
    logger.info("Dataset formatted with %d samples", len(dataset))

    return dataset


def train(model, tokenizer, dataset, config: dict, cost_tracker=None):
    """Run the training loop using SFTTrainer."""
    from trl import SFTTrainer
    from transformers import TrainingArguments

    train_cfg = config["training"]
    paths_cfg = config["paths"]

    output_dir = paths_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting training: %d epochs, lr=%.1e, batch=%d, grad_accum=%d",
                train_cfg["num_epochs"], train_cfg["learning_rate"],
                train_cfg["batch_size"], train_cfg["gradient_accumulation_steps"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_steps=train_cfg["warmup_steps"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg.get("save_total_limit", 3),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        optim=train_cfg.get("optim", "adamw_8bit"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        seed=train_cfg.get("seed", 42),
        logging_dir=paths_cfg.get("log_dir", "/app/logs"),
        report_to="none",
        max_steps=int(os.environ.get("MAX_STEPS", 0)) or -1,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config["model"].get("max_seq_length", 2048),
        args=training_args,
    )

    # Check for checkpoint resumption
    checkpoint = os.environ.get("RESUME_CHECKPOINT")
    if checkpoint and os.path.isdir(checkpoint):
        logger.info("Resuming from checkpoint: %s", checkpoint)
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Log final metrics
    train_result = trainer.state.log_history[-1] if trainer.state.log_history else {}
    logger.info("Training complete. Final loss: %.4f", train_result.get("loss", 0))
    logger.info("Training samples/sec: %.2f", train_result.get("train_samples_per_second", 0))

    return trainer


def save_training_history(trainer, output_dir: str = "/app/output") -> str:
    """Export training loss curve data as JSON and CSV."""
    history_path = Path(output_dir)
    history_path.mkdir(parents=True, exist_ok=True)

    log_history = trainer.state.log_history
    records = []
    for entry in log_history:
        if "loss" in entry:
            records.append({
                "step": entry.get("step", 0),
                "loss": entry.get("loss"),
                "learning_rate": entry.get("learning_rate"),
                "epoch": entry.get("epoch"),
            })

    # Save JSON
    json_path = history_path / "training_history.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    logger.info("Training history saved to %s (%d entries)", json_path, len(records))

    # Save CSV
    csv_path = history_path / "training_history.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "learning_rate", "epoch"])
        writer.writeheader()
        writer.writerows(records)
    logger.info("Training history CSV saved to %s", csv_path)

    return str(json_path)


def run_inference_comparison(
    config: dict,
    lora_dir: str,
    output_dir: str = "/app/output",
    sample_prompts: list = None,
) -> str:
    """Run inference comparison between base and fine-tuned model."""
    if sample_prompts is None:
        sample_prompts = [
            "Explain the concept of gradient descent in simple terms.",
            "What is the difference between supervised and unsupervised learning?",
            "Describe how a transformer attention mechanism works.",
            "What are the advantages of QLoRA over full fine-tuning?",
            "Explain batch normalization and why it helps training.",
        ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_cfg = config["model"]
    base_model_name = model_cfg["base_model"]
    max_seq_length = model_cfg.get("max_seq_length", 2048)

    results = []
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.warning("Unsloth not available, skipping inference comparison")
        return None

    # --- Base model inference ---
    logger.info("Loading base model for inference comparison: %s", base_model_name)
    try:
        base_model, base_tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=max_seq_length,
            dtype=model_cfg.get("dtype"),
            load_in_4bit=model_cfg.get("load_in_4bit", True),
        )
        FastLanguageModel.for_inference(base_model)

        base_outputs = []
        for prompt in sample_prompts:
            inputs = base_tokenizer(prompt, return_tensors="pt").to(base_model.device)
            output_ids = base_model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True)
            response = base_tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            base_outputs.append(response)

        # Unload base model to free GPU memory
        del base_model
        del base_tokenizer
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Base model inference complete, memory freed")
    except Exception as e:
        logger.warning("Base model inference failed: %s", e)
        base_outputs = ["[base inference failed]"] * len(sample_prompts)

    # --- Fine-tuned model inference ---
    logger.info("Loading fine-tuned model from %s", lora_dir)
    try:
        ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_dir,
            max_seq_length=max_seq_length,
            dtype=model_cfg.get("dtype"),
            load_in_4bit=model_cfg.get("load_in_4bit", True),
        )
        FastLanguageModel.for_inference(ft_model)

        ft_outputs = []
        for prompt in sample_prompts:
            inputs = ft_tokenizer(prompt, return_tensors="pt").to(ft_model.device)
            output_ids = ft_model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True)
            response = ft_tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            ft_outputs.append(response)

        del ft_model
        del ft_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Fine-tuned model inference complete")
    except Exception as e:
        logger.warning("Fine-tuned model inference failed: %s", e)
        ft_outputs = ["[finetuned inference failed]"] * len(sample_prompts)

    # Build comparison results
    for i, prompt in enumerate(sample_prompts):
        results.append({
            "prompt": prompt,
            "base_output": base_outputs[i] if i < len(base_outputs) else "[missing]",
            "finetuned_output": ft_outputs[i] if i < len(ft_outputs) else "[missing]",
            "timestamp": timestamp,
        })

    # Save comparison results
    comparison_path = output_path / "inference_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Inference comparison saved to %s (%d prompts)", comparison_path, len(results))

    # Log results to console
    for r in results:
        logger.info("---\nPrompt: %s\nBase: %s\nFine-tuned: %s", r["prompt"][:80], r["base_output"][:100], r["finetuned_output"][:100])

    return str(comparison_path)


def save_model(model, tokenizer, config: dict):
    """Save LoRA adapters and optionally merged model."""
    paths_cfg = config["paths"]
    upload_cfg = config["upload"]

    # Save LoRA adapters
    lora_dir = paths_cfg["lora_dir"]
    Path(lora_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Saving LoRA adapters to %s", lora_dir)
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    logger.info("LoRA adapters saved")

    # Optionally merge and save full model
    if upload_cfg.get("push_merged", True):
        merged_dir = paths_cfg["merged_dir"]
        Path(merged_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Merging and saving model to %s", merged_dir)

        try:
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
            logger.info("Merged model saved (16-bit)")
        except Exception as e:
            logger.warning("Merge save failed (%s), saving unmerged", e)
            model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)

    return lora_dir


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model with Unsloth QLoRA")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    from src.utils import setup_logging, set_seed, CostTracker, check_gpu

    # Setup
    setup_logging()
    config = load_config(args.config)
    set_seed(config["training"].get("seed", 42))

    # GPU check
    gpu_info = check_gpu()
    logger.info("GPU: %s", gpu_info)

    # Cost tracking
    cost_cfg = config["cost"]
    cost_tracker = CostTracker(
        hourly_rate=cost_cfg["hourly_rate"],
        max_budget=cost_cfg["max_budget"],
        warn_at_percent=cost_cfg.get("warn_at_percent", 80),
        stop_at_percent=cost_cfg.get("stop_at_percent", 100),
    )
    cost_tracker.start()

    # Override max_steps from CLI
    if args.max_steps:
        os.environ["MAX_STEPS"] = str(args.max_steps)

    # Load model
    model, tokenizer = load_model_and_tokenizer(config)
    model = apply_lora(model, config)

    # Load dataset
    dataset = load_dataset(config)

    # Train
    logger.info("=" * 60)
    logger.info("TRAINING START")
    logger.info("=" * 60)

    start = time.time()
    trainer = train(model, tokenizer, dataset, config, cost_tracker)
    elapsed = time.time() - start

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Elapsed: %.1f seconds (%.2f hours)", elapsed, elapsed / 3600)
    logger.info("=" * 60)

    # Check cost
    cost_status = cost_tracker.check()
    logger.info("Cost status: %s", json.dumps(cost_status, indent=2))

    # Export training history (AC-09)
    output_dir = config["paths"].get("output_dir", "/app/output")
    save_training_history(trainer, output_dir)

    # Save
    lora_dir = save_model(model, tokenizer, config)

    # Inference comparison (AC-10)
    run_inference_comparison(config, lora_dir, output_dir)

    # Final summary
    summary = cost_tracker.summary()
    logger.info("Total cost: $%.4f (%.2f hours)", summary["total_cost"], summary["elapsed_hours"])
    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    main()
