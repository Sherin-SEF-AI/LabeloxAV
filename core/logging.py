"""Structured JSON logging via structlog. Call setup_logging() once at process start."""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(level: str = "INFO", stream=None) -> None:
    """Configure stdlib logging and structlog. `stream` defaults to stdout; a command whose stdout is its
    output (scripts/sign_in_link.py prints a link a shell script captures) passes stderr instead."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    out = stream or sys.stdout

    logging.basicConfig(format="%(message)s", stream=out, level=log_level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=out),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
