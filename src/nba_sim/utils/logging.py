"""Centralized logging setup.

Call :func:`setup_logging` once at process start (done by the CLI and by
the training entry points). Uses ``rich`` handlers for human-readable
terminal output and structured key=value lines for file sinks.
"""

from __future__ import annotations

import logging


def setup_logging(level: str | int = "INFO", *, json_file: str | None = None) -> None:
    """Configure root logger. Safe to call multiple times — idempotent."""
    raise NotImplementedError


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger. Thin wrapper around :func:`logging.getLogger`."""
    return logging.getLogger(name)
