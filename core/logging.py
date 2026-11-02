"""Structured JSON logging via structlog. Call setup_logging() once at process start."""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(level: str = "INFO", stream=None) -> None:
    """Configure stdlib logging and structlog. `stream` defaults to stdout; a command whose stdout is its
    output (scripts/sign_in_link.py prints a link a shell script captures) passes stderr instead."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=stream or sys.stdout, level=log_level)

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
        # Without an explicit stream the factory resolves stdout when each logger is made, not now. Binding the
        # stdout of this moment instead broke every later log line wherever stdout is swapped out, as pytest
        # does per test: 57 tests failed on "I/O operation on closed file" once the API's startup ran.
        logger_factory=structlog.PrintLoggerFactory(file=stream) if stream else structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
