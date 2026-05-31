"""Unit tests for src/utils.py."""

import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import CostTracker, check_gpu, set_seed, setup_logging, validate_env_vars


class TestCheckGPU:
    """Tests for check_gpu function."""

    @patch("src.utils.subprocess.run")
    def test_single_gpu_detected(self, mock_run):
        """Single GPU returns correct info."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA H200 141GiB, 147456, 550.54.15",
        )
        result = check_gpu()
        assert result["gpu_count"] == 1
        assert result["gpus"][0]["name"] == "NVIDIA H200 141GiB"
        assert result["gpus"][0]["memory_total_mb"] == 147456

    @patch("src.utils.subprocess.run")
    def test_multiple_gpus_detected(self, mock_run):
        """Multiple GPUs all parsed."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA A100 80GiB, 81920, 550.54.15\nNVIDIA A100 80GiB, 81920, 550.54.15",
        )
        result = check_gpu()
        assert result["gpu_count"] == 2

    @patch("src.utils.subprocess.run")
    def test_no_gpu_exits(self, mock_run):
        """Missing nvidia-smi causes SystemExit."""
        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(SystemExit):
            check_gpu()

    @patch("src.utils.subprocess.run")
    def test_nvidia_smi_failure_exits(self, mock_run):
        """nvidia-smi non-zero return causes SystemExit."""
        mock_run.return_value = MagicMock(returncode=1, stderr="driver error")
        with pytest.raises(SystemExit):
            check_gpu()

    @patch("src.utils.subprocess.run")
    def test_empty_output_exits(self, mock_run):
        """Empty nvidia-smi output causes SystemExit."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        with pytest.raises(SystemExit):
            check_gpu()


class TestCostTracker:
    """Tests for CostTracker class."""

    def test_initial_state(self):
        """Tracker starts with zero cost."""
        tracker = CostTracker(hourly_rate=0.75, max_budget=3.00)
        assert tracker.current_cost() == 0.0
        assert tracker.elapsed_hours() == 0.0
        assert tracker.budget_used_percent() == 0.0

    def test_cost_accumulation(self):
        """Cost increases with time."""
        tracker = CostTracker(hourly_rate=0.75, max_budget=3.00)
        tracker.start_time = time.time() - 3600  # 1 hour ago
        cost = tracker.current_cost()
        assert 0.74 < cost < 0.76  # ~$0.75 for 1 hour

    def test_budget_warning(self, caplog):
        """Warning logged at 80% budget without hard stop."""
        import logging
        with caplog.at_level(logging.WARNING, logger="finetune.cost"):
            tracker = CostTracker(hourly_rate=0.75, max_budget=1.00, warn_at_percent=80.0, stop_at_percent=200.0)
            tracker.start_time = time.time() - 4000  # >80% of 1hr budget
            status = tracker.check()
            assert status["budget_used_percent"] >= 80.0
            assert "BUDGET WARNING" in caplog.text

    def test_budget_exceeded_exits(self):
        """Hard stop at 100% budget."""
        tracker = CostTracker(hourly_rate=0.75, max_budget=0.01, stop_at_percent=100.0)
        tracker.start_time = time.time() - 3600  # 1 hour ago = $0.75 >> $0.01 budget
        with pytest.raises(SystemExit):
            tracker.check()

    def test_summary_returns_dict(self):
        """Summary returns all expected keys."""
        tracker = CostTracker(hourly_rate=0.75, max_budget=3.00)
        tracker.start()
        summary = tracker.summary()
        assert "elapsed_hours" in summary
        assert "total_cost" in summary
        assert "hourly_rate" in summary
        assert "budget" in summary
        assert "budget_used_percent" in summary

    def test_custom_rates(self):
        """Custom hourly rate and budget work correctly."""
        tracker = CostTracker(hourly_rate=2.00, max_budget=10.00)
        tracker.start_time = time.time() - 1800  # 30 min
        cost = tracker.current_cost()
        assert 0.99 < cost < 1.01  # $2/hr * 0.5hr = $1.00


class TestSetSeed:
    """Tests for set_seed function."""

    def test_python_seed_set(self):
        """Python random seed is set."""
        import random
        set_seed(123)
        val1 = random.random()
        set_seed(123)
        val2 = random.random()
        assert val1 == val2

    def test_pythonhashseed_env(self):
        """PYTHONHASHSEED env var is set."""
        set_seed(42)
        assert os.environ.get("PYTHONHASHSEED") == "42"

    def test_numpy_seed(self):
        """NumPy seed is set when available."""
        import numpy as np
        set_seed(99)
        val1 = np.random.random()
        set_seed(99)
        val2 = np.random.random()
        assert val1 == val2


class TestValidateEnvVars:
    """Tests for validate_env_vars function."""

    def test_all_present(self):
        """Returns dict when all vars present."""
        os.environ["TEST_VAR_A"] = "value_a"
        os.environ["TEST_VAR_B"] = "value_b"
        result = validate_env_vars(["TEST_VAR_A", "TEST_VAR_B"])
        assert result == {"TEST_VAR_A": "value_a", "TEST_VAR_B": "value_b"}
        del os.environ["TEST_VAR_A"]
        del os.environ["TEST_VAR_B"]

    def test_missing_var_exits(self):
        """SystemExit when required var missing."""
        # Ensure the var doesn't exist
        os.environ.pop("TEST_MISSING_VAR", None)
        with pytest.raises(SystemExit):
            validate_env_vars(["TEST_MISSING_VAR"])

    def test_empty_list_passes(self):
        """Empty required list returns empty dict."""
        result = validate_env_vars([])
        assert result == {}

    def test_partial_missing_exits(self):
        """SystemExit when some vars missing."""
        os.environ["TEST_PRESENT"] = "yes"
        os.environ.pop("TEST_ABSENT", None)
        with pytest.raises(SystemExit):
            validate_env_vars(["TEST_PRESENT", "TEST_ABSENT"])
        del os.environ["TEST_PRESENT"]


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_creates_log_dir(self):
        """Log directory and file are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs")
            logger = setup_logging(log_dir=log_dir)
            assert os.path.exists(os.path.join(log_dir, "training.log"))
            assert logger.name == "finetune"
