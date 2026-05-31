"""Integration tests for the Akash Unsloth Fine-tune pipeline.

Tests end-to-end data preparation and training configuration validation.
Full training requires GPU; these tests validate the pipeline structure.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def temp_dirs():
    """Create temporary directories for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "data")
        output_dir = os.path.join(tmpdir, "output")
        lora_dir = os.path.join(output_dir, "lora_adapters")
        merged_dir = os.path.join(output_dir, "merged_model")
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(data_dir)
        os.makedirs(lora_dir)
        os.makedirs(merged_dir)
        os.makedirs(log_dir)
        yield {
            "tmpdir": tmpdir,
            "data_dir": data_dir,
            "output_dir": output_dir,
            "lora_dir": lora_dir,
            "merged_dir": merged_dir,
            "log_dir": log_dir,
            "train_file": os.path.join(data_dir, "train.jsonl"),
        }


@pytest.fixture
def sample_jsonl(temp_dirs):
    """Create a small sample JSONL training file."""
    samples = []
    for i in range(10):
        samples.append({
            "instruction": "Solve the equation x + {} = {}".format(i, i * 2),
            "input": "",
            "output": "x = {}".format(i),
        })
    with open(temp_dirs["train_file"], "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return temp_dirs["train_file"]


class TestEndToEndDataPipeline:
    """Test the data preparation to training file pipeline."""

    def test_prepare_and_validate(self, temp_dirs):
        """Data prep produces valid JSONL that can be loaded."""
        from src.prepare_data import convert_to_alpaca, validate_sample, save_jsonl

        raw_data = [
            {"question": "What is {} plus {}?".format(i + 5, i + 5), "answer": str((i + 5) * 2)}
            for i in range(10)
        ]

        samples = []
        for raw in raw_data:
            alpaca = convert_to_alpaca(raw)
            if validate_sample(alpaca):
                samples.append(alpaca)

        assert len(samples) == 10
        save_jsonl(samples, temp_dirs["train_file"])

        loaded = []
        with open(temp_dirs["train_file"]) as f:
            for line in f:
                loaded.append(json.loads(line.strip()))

        assert len(loaded) == 10
        assert all("instruction" in s for s in loaded)
        assert all("output" in s for s in loaded)

    def test_config_loads_and_validates(self):
        """Config file loads and has required sections."""
        from src.utils import load_config

        config = load_config("configs/train_config.yaml")

        required_sections = ["model", "dataset", "lora", "training", "upload", "cost", "paths"]
        for section in required_sections:
            assert section in config, "Missing config section: {}".format(section)

        assert config["model"]["base_model"] == "unsloth/qwen3.5-4b-instruct-bnb-4bit"
        assert config["lora"]["rank"] == 16
        assert config["training"]["num_epochs"] == 3
        assert config["cost"]["hourly_rate"] == 0.75

    def test_training_args_from_config(self):
        """Training arguments can be constructed from config."""
        from src.utils import load_config

        config = load_config("configs/train_config.yaml")
        train_cfg = config["training"]

        assert train_cfg["learning_rate"] > 0
        assert train_cfg["batch_size"] > 0
        assert train_cfg["num_epochs"] > 0
        assert train_cfg["gradient_accumulation_steps"] > 0
        assert train_cfg["warmup_steps"] >= 0

    def test_dataset_loading_from_jsonl(self, sample_jsonl):
        """Prepared JSONL can be loaded and formatted for training."""
        from datasets import Dataset

        samples = []
        with open(sample_jsonl) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line.strip()))

        assert len(samples) == 10
        dataset = Dataset.from_list(samples)
        assert len(dataset) == 10

        def format_alpaca(example):
            if example.get("input") and example["input"].strip():
                text = (
                    "### Instruction:\n{}\n\n"
                    "### Input:\n{}\n\n"
                    "### Response:\n{}"
                ).format(example["instruction"], example["input"], example["output"])
            else:
                text = (
                    "### Instruction:\n{}\n\n"
                    "### Response:\n{}"
                ).format(example["instruction"], example["output"])
            return {"text": text}

        dataset = dataset.map(format_alpaca)
        assert len(dataset) == 10
        assert all("text" in row for row in dataset)

    def test_cost_tracker_integration(self):
        """Cost tracker works with config values."""
        from src.utils import load_config, CostTracker

        config = load_config("configs/train_config.yaml")
        cost_cfg = config["cost"]
        tracker = CostTracker(
            hourly_rate=cost_cfg["hourly_rate"],
            max_budget=cost_cfg["max_budget"],
        )
        tracker.start()
        summary = tracker.summary()
        assert summary["hourly_rate"] == 0.75
        assert summary["budget"] == 3.00
        assert summary["total_cost"] >= 0


class TestAlpacaFormatting:
    """Test the Alpaca prompt formatting pipeline."""

    def test_format_with_input(self):
        """Format sample with input context."""
        sample = {
            "instruction": "Translate to French",
            "input": "Hello world",
            "output": "Bonjour le monde",
        }
        text = (
            "### Instruction:\n{}\n\n"
            "### Input:\n{}\n\n"
            "### Response:\n{}"
        ).format(sample["instruction"], sample["input"], sample["output"])

        assert "### Instruction:" in text
        assert "### Input:" in text
        assert "### Response:" in text
        assert "Bonjour" in text

    def test_format_without_input(self):
        """Format sample without input context."""
        sample = {
            "instruction": "What is AI?",
            "input": "",
            "output": "Artificial intelligence is...",
        }
        text = (
            "### Instruction:\n{}\n\n"
            "### Response:\n{}"
        ).format(sample["instruction"], sample["output"])

        assert "### Instruction:" in text
        assert "### Response:" in text
        assert "### Input:" not in text


class TestDockerfileVerification:
    """Verify Dockerfile meets requirements."""

    def test_no_latest_tags(self):
        """Dockerfile has no :latest image tags."""
        with open("Dockerfile") as f:
            content = f.read()
        assert ":latest" not in content

    def test_multi_stage(self):
        """Dockerfile uses multi-stage build."""
        with open("Dockerfile") as f:
            content = f.read()
        assert "AS base" in content
        assert "AS app" in content

    def test_pinned_versions(self):
        """Python packages are pinned."""
        with open("Dockerfile") as f:
            content = f.read()
        # Check for == version pins
        assert "unsloth==" in content
        assert "torch==" in content
        assert "transformers==" in content

    def test_exposes_port(self):
        """Port 8080 is exposed."""
        with open("Dockerfile") as f:
            content = f.read()
        assert "EXPOSE 8080" in content


class TestDeployYamlVerification:
    """Verify deploy.yaml meets requirements."""

    def test_has_h200_gpu(self):
        """GPU is configured as H200."""
        import yaml
        with open("deploy.yaml") as f:
            config = yaml.safe_load(f)
        gpu = config["profiles"]["compute"]["finetune"]["resources"]["gpu"]
        assert gpu["units"] == 1
        assert "h200" in str(gpu["attributes"]["vendor"]["nvidia"])

    def test_required_env_vars_present(self):
        """HF_TOKEN and HF_REPO_ID are in env vars."""
        with open("deploy.yaml") as f:
            content = f.read()
        assert "HF_TOKEN" in content
        assert "HF_REPO_ID" in content

    def test_port_exposed(self):
        """Port 8080 is exposed globally."""
        import yaml
        with open("deploy.yaml") as f:
            config = yaml.safe_load(f)
        expose = config["services"]["finetune"]["expose"]
        assert expose[0]["port"] == 8080
        assert expose[0]["to"][0]["global"] is True

    def test_pricing_in_uact(self):
        """Pricing uses uact denomination."""
        import yaml
        with open("deploy.yaml") as f:
            config = yaml.safe_load(f)
        pricing = config["profiles"]["placement"]["akash"]["pricing"]["finetune"]
        assert pricing["denom"] == "uact"
