"""Tests for src/prepare_data.py."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prepare_data import convert_to_alpaca, validate_sample, save_jsonl, subset_dataset


class TestConvertToAlpaca:
    """Tests for convert_to_alpaca function."""

    def test_standard_fields(self):
        """Convert question/answer fields to Alpaca format."""
        sample = {"question": "What is 2+2?", "answer": "4"}
        result = convert_to_alpaca(sample)
        assert result["instruction"] == "What is 2+2?"
        assert result["output"] == "4"
        assert result["input"] == ""

    def test_with_context(self):
        """Context field maps to input."""
        sample = {"question": "Solve", "context": "Given x=3", "answer": "3"}
        result = convert_to_alpaca(sample)
        assert result["input"] == "Given x=3"

    def test_alternative_field_names(self):
        """Handle prompt/response, instruction/output, problem/solution."""
        # prompt/response
        result = convert_to_alpaca({"prompt": "Q1", "response": "A1"})
        assert result["instruction"] == "Q1"
        assert result["output"] == "A1"

        # instruction/output
        result = convert_to_alpaca({"instruction": "Do X", "output": "Done"})
        assert result["instruction"] == "Do X"
        assert result["output"] == "Done"

    def test_list_fields_joined(self):
        """List values are joined with newlines."""
        sample = {"question": ["Part 1", "Part 2"], "answer": ["A1", "A2"]}
        result = convert_to_alpaca(sample)
        assert "Part 1" in result["instruction"]
        assert "Part 2" in result["instruction"]

    def test_final_answer_fallback(self):
        """Use final_answer when answer field is empty."""
        sample = {"question": "Compute x", "final_answer": 42}
        result = convert_to_alpaca(sample)
        assert result["output"] == "42"

    def test_empty_sample(self):
        """Empty sample produces empty instruction/output."""
        result = convert_to_alpaca({})
        assert result["instruction"] == ""
        assert result["output"] == ""


class TestValidateSample:
    """Tests for validate_sample function."""

    def test_valid_sample(self):
        """Valid Alpaca sample passes validation."""
        sample = {"instruction": "Explain quantum computing", "input": "", "output": "Quantum computing uses qubits..."}
        assert validate_sample(sample) is True

    def test_missing_instruction(self):
        """Missing instruction fails validation."""
        sample = {"input": "", "output": "Some answer"}
        assert validate_sample(sample) is False

    def test_missing_output(self):
        """Missing output fails validation."""
        sample = {"instruction": "A question", "input": ""}
        assert validate_sample(sample) is False

    def test_short_instruction(self):
        """Instruction shorter than 5 chars fails."""
        sample = {"instruction": "Hi", "input": "", "output": "Hello"}
        assert validate_sample(sample) is False

    def test_short_output(self):
        """Output shorter than 2 chars fails."""
        sample = {"instruction": "Explain this", "input": "", "output": "A"}
        assert validate_sample(sample) is False

    def test_empty_strings(self):
        """Empty string values fail validation."""
        sample = {"instruction": "", "input": "", "output": ""}
        assert validate_sample(sample) is False


class TestSaveJsonl:
    """Tests for save_jsonl function."""

    def test_creates_file(self):
        """JSONL file is created with correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data", "test.jsonl")
            samples = [
                {"instruction": "Q1", "input": "", "output": "A1"},
                {"instruction": "Q2", "input": "ctx", "output": "A2"},
            ]
            save_jsonl(samples, path)

            assert os.path.exists(path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0])["instruction"] == "Q1"
            assert json.loads(lines[1])["input"] == "ctx"

    def test_empty_samples(self):
        """Empty list creates empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.jsonl")
            save_jsonl([], path)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert content == ""


class TestSubsetDataset:
    """Tests for subset_dataset function."""

    def test_subset_smaller(self):
        """Subset to smaller size works correctly."""
        from datasets import Dataset

        data = [{"text": f"item_{i}"} for i in range(100)]
        dataset = Dataset.from_list(data)
        result = subset_dataset(dataset, size=10, seed=42)
        assert len(result) == 10

    def test_subset_larger_than_dataset(self):
        """Requesting more than available returns full dataset."""
        from datasets import Dataset

        data = [{"text": f"item_{i}"} for i in range(5)]
        dataset = Dataset.from_list(data)
        result = subset_dataset(dataset, size=100, seed=42)
        assert len(result) == 5

    def test_subset_deterministic(self):
        """Same seed produces same subset."""
        from datasets import Dataset

        data = [{"text": f"item_{i}"} for i in range(50)]
        dataset = Dataset.from_list(data)
        r1 = subset_dataset(dataset, size=10, seed=42)
        r2 = subset_dataset(dataset, size=10, seed=42)
        assert r1 == r2
