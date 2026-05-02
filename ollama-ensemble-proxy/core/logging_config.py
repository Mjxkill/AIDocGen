"""Structured logging setup for the proxy.

One JSON line per log event on stdout — parseable by journalctl + Loki/ELK
later, but still readable as-is. Add per-call context via `extra={}`.

Usage:
    from core.logging_config import get_logger
    log = get_logger(__name__)
    log.info("run started", extra={"run_id": run_id, "user": user["id"]})
"""
import json
import logging
import os
import sys
import time
from typing import Any

# Fields produced by the stdlib LogRecord that we don't want in our JSON output
# (we replace them with our own normalized keys).
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach any extra={} fields the caller passed
        for key, val in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            try:
                json.dumps(val)        # filter unserializable values
                payload[key] = val
            except Exception:
                payload[key] = repr(val)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger once. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(lvl)
    # Tame noisy libraries
    for noisy in ("httpx", "httpcore", "uvicorn.access", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
