"""
JSON Logger utility for Customer Churn Analysis Pipeline.
Provides structured JSONL logging with log rotation handler.
"""

import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler


class JSONStepFormatter(logging.Formatter):
    """Custom logging formatter outputting single-line JSON objects."""

    def format(self, record):
        step_name = getattr(record, "step", record.getMessage())
        exec_time = getattr(record, "execution_time_seconds", 0.0)
        n_samples = getattr(record, "n_samples", 0)

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "step": step_name,
            "execution_time_seconds": round(float(exec_time), 4),
            "n_samples": int(n_samples),
        }
        return json.dumps(log_entry)


def get_json_logger(name="churn_pipeline", log_file="logs/pipeline.jsonl"):
    """Initialize and return a logger with RotatingFileHandler writing JSONL format."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # RotatingFileHandler: 5MB max size per log file, 3 backup files
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(JSONStepFormatter())
        logger.addHandler(file_handler)

        # Also attach console handler for CLI output
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(JSONStepFormatter())
        logger.addHandler(console_handler)

    return logger


def log_pipeline_step(
    logger, step_name, exec_time_seconds, n_samples, level=logging.INFO
):
    """Log a structured pipeline step with execution time and sample count."""
    extra = {
        "step": step_name,
        "execution_time_seconds": exec_time_seconds,
        "n_samples": n_samples,
    }
    logger.log(level, f"Pipeline step '{step_name}' completed", extra=extra)
