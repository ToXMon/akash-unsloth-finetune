"""
Utility functions for Akash Unsloth Fine-tune Demo.

Provides GPU checking, logging, cost tracking, seed setting, and env validation.
"""

import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def check_gpu() -> Dict[str, str]:
    """Run nvidia-smi and return GPU info. Exit with clear error if no GPU found."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"ERROR: nvidia-smi failed: {result.stderr.strip()}", file=sys.stderr)
            sys.exit(1)

        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        if not lines:
            print("ERROR: No GPU detected. nvidia-smi returned empty output.", file=sys.stderr)
            sys.exit(1)

        gpus = []
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "index": i,
                    "name": parts[0],
                    "memory_total_mb": int(parts[1]),
                    "driver_version": parts[2],
                })

        if not gpus:
            print("ERROR: Could not parse GPU info from nvidia-smi output.", file=sys.stderr)
            sys.exit(1)

        gpu_info = {"gpu_count": len(gpus), "gpus": gpus}
        logging.info("GPU check passed: %d GPU(s) detected", len(gpus))
        for g in gpus:
            logging.info("  GPU %d: %s (%d MB), driver %s", g["index"], g["name"], g["memory_total_mb"], g["driver_version"])

        return gpu_info

    except FileNotFoundError:
        print("ERROR: nvidia-smi not found. No NVIDIA GPU or driver installed.", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("ERROR: nvidia-smi timed out after 30 seconds.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: GPU check failed: {e}", file=sys.stderr)
        sys.exit(1)


def setup_logging(log_dir: str = "/app/logs", level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging to console and file."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / "training.log"

    # Create formatter with structured output
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(str(log_file), mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger("finetune")
    logger.setLevel(level)

    logger.info("Logging initialized — console + %s", log_file)
    return logger


class CostTracker:
    """
    Track training cost based on hourly rate and elapsed time.
    Warns at configurable threshold, hard-stops at budget limit.
    """

    def __init__(
        self,
        hourly_rate: float = 0.75,
        max_budget: float = 3.00,
        warn_at_percent: float = 80.0,
        stop_at_percent: float = 100.0,
    ):
        self.hourly_rate = hourly_rate
        self.max_budget = max_budget
        self.warn_at_percent = warn_at_percent
        self.stop_at_percent = stop_at_percent
        self.start_time: Optional[float] = None
        self._warned = False
        self.logger = logging.getLogger("finetune.cost")

    def start(self) -> None:
        """Start the cost timer."""
        self.start_time = time.time()
        self.logger.info(
            "Cost tracking started: $%.2f/hr, budget $%.2f (est %.1f hrs)",
            self.hourly_rate, self.max_budget, self.max_budget / self.hourly_rate,
        )

    def elapsed_hours(self) -> float:
        """Return hours elapsed since start()."""
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) / 3600.0

    def current_cost(self) -> float:
        """Calculate current cost based on elapsed time."""
        return self.elapsed_hours() * self.hourly_rate

    def budget_used_percent(self) -> float:
        """Return percentage of budget used."""
        if self.max_budget <= 0:
            return 100.0
        return (self.current_cost() / self.max_budget) * 100.0

    def check(self) -> Dict[str, float]:
        """
        Check budget status. Returns status dict.
        Raises SystemExit if budget exceeded.
        """
        cost = self.current_cost()
        pct = self.budget_used_percent()
        remaining = max(0.0, self.max_budget - cost)
        remaining_hours = remaining / self.hourly_rate if self.hourly_rate > 0 else 0.0

        status = {
            "elapsed_hours": self.elapsed_hours(),
            "current_cost": round(cost, 4),
            "budget_used_percent": round(pct, 2),
            "remaining_budget": round(remaining, 4),
            "remaining_hours": round(remaining_hours, 2),
        }

        # Warn threshold
        if pct >= self.warn_at_percent and not self._warned:
            self._warned = True
            self.logger.warning(
                "BUDGET WARNING: %.1f%% used ($%.2f of $%.2f). %.1f hours remaining.",
                pct, cost, self.max_budget, remaining_hours,
            )

        # Hard stop threshold
        if pct >= self.stop_at_percent:
            self.logger.error(
                "BUDGET EXCEEDED: %.1f%% used ($%.2f of $%.2f). Stopping to prevent overruns.",
                pct, cost, self.max_budget,
            )
            print(
                f"FATAL: Budget exceeded (${cost:.2f} / ${self.max_budget:.2f}). "
                f"Training stopped.",
                file=sys.stderr,
            )
            sys.exit(2)

        return status

    def summary(self) -> Dict[str, float]:
        """Return final cost summary without exit checks."""
        cost = self.current_cost()
        return {
            "elapsed_hours": round(self.elapsed_hours(), 4),
            "total_cost": round(cost, 4),
            "hourly_rate": self.hourly_rate,
            "budget": self.max_budget,
            "budget_used_percent": round(self.budget_used_percent(), 2),
        }


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
        logging.getLogger("finetune").debug("NumPy seed set to %d", seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logging.getLogger("finetune").debug("PyTorch seed set to %d (cuda + cudnn deterministic)", seed)
    except ImportError:
        pass

    logging.getLogger("finetune").info("Random seed set to %d", seed)


def validate_env_vars(required: List[str]) -> Dict[str, str]:
    """
    Check that required environment variables are set.
    Exits with error listing all missing variables.
    Returns dict of found variables.
    """
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        print(
            f"ERROR: Missing required environment variables:\n"
            + "\n".join(f"  - {var}" for var in missing),
            file=sys.stderr,
        )
        sys.exit(1)

    found = {var: os.environ[var] for var in required}
    logging.getLogger("finetune").info("All %d required env vars present: %s", len(required), ", ".join(required))
    return found


def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    """Load and return training config from YAML file."""
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logging.getLogger("finetune").info("Config loaded from %s", config_path)
    return config
