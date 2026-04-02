"""
logger.py
---------
Centralised logging for the AI Travel Planner.

All log records are written as JSON to logs/travel_planner.log (rotating,
10 MB per file, 5 backups) so they can be parsed/queried later.
Human-readable INFO+ records also go to the console.

Usage
-----
    from src.logger import get_logger, log_api_call, log_agent_run

    # Module-level logger (use __name__ so the hierarchy is automatic)
    logger = get_logger(__name__)
    logger.info("Something happened")
    logger.debug("Full payload: %s", data)

    # Macro-style decorator — wraps any function that calls an external API
    @log_api_call("serpapi.flights")
    def _fetch_outbound(self, ...):
        ...

    # Macro-style decorator — wraps an agent entry-point method
    @log_agent_run("FlightAgent")
    def search_and_recommend(self, ...):
        ...
"""

from __future__ import annotations

import functools
import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable

# ── Log directory / file ─────────────────────────────────────────────────────

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = _LOG_DIR / "travel_planner.log"


# ── JSON formatter ────────────────────────────────────────────────────────────

class _JSONFormatter(logging.Formatter):
    """Emit one compact JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts":     self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }
        # Attach any structured payload passed as extra={"payload": ...}
        payload = getattr(record, "payload", None)
        if payload is not None:
            entry["data"] = payload
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


# ── Root logger setup (runs once) ─────────────────────────────────────────────

_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return

    root = logging.getLogger("travel_planner")
    root.setLevel(logging.DEBUG)

    # File handler — full JSON, rotating
    fh = RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_JSONFormatter())
    root.addHandler(fh)

    # Console handler — human-readable, INFO and above only
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(ch)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "langchain", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


# ── Public factory ─────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'travel_planner' hierarchy.

    Args:
        name: Typically ``__name__`` from the calling module, or a short
              label such as ``"agent.flight"``.
    """
    _configure_root()
    short = name.removeprefix("src.")
    return logging.getLogger(f"travel_planner.{short}")


# ── Structured log helpers ─────────────────────────────────────────────────────

def _emit(logger: logging.Logger, level: int, msg: str, payload: Any = None) -> None:
    """Emit a log record, attaching an optional structured payload for the file."""
    if logger.isEnabledFor(level):
        extra = {"payload": payload} if payload is not None else {}
        logger.log(level, msg, extra=extra)


def log_api_request(logger: logging.Logger, endpoint: str, params: dict) -> None:
    """Log an outgoing API request (SerpAPI, OpenAI …)."""
    _emit(logger, logging.DEBUG, f"API_REQUEST  {endpoint}", {
        "event": "api_request", "endpoint": endpoint, "params": params,
    })


def log_api_response(
    logger: logging.Logger, endpoint: str, result: Any, duration_ms: float
) -> None:
    """Log an API response with full payload (for later debug replay)."""
    _emit(logger, logging.DEBUG, f"API_RESPONSE {endpoint}  ({duration_ms:.0f} ms)", {
        "event": "api_response",
        "endpoint": endpoint,
        "duration_ms": round(duration_ms, 1),
        "response": result,
    })


def log_llm_call(
    logger: logging.Logger,
    model: str,
    prompt: str,
    response: Any,
    duration_ms: float,
) -> None:
    """Log an LLM invocation with truncated prompt/response."""
    _emit(logger, logging.INFO, f"LLM_CALL  model={model}  ({duration_ms:.0f} ms)", {
        "event": "llm_call",
        "model": model,
        "prompt_preview": prompt[:500],
        "response_preview": str(response)[:500],
        "duration_ms": round(duration_ms, 1),
    })


def log_user_message(logger: logging.Logger, message: str) -> None:
    """Log an incoming user message (conversation audit trail)."""
    _emit(logger, logging.INFO, f"USER_MESSAGE: {message[:200]}", {
        "event": "user_message", "message": message,
    })


def log_agent_start(logger: logging.Logger, agent: str, kwargs: dict) -> None:
    """Log the start of an agent run with its input parameters."""
    _emit(logger, logging.INFO, f"AGENT_START  {agent}", {
        "event": "agent_start", "agent": agent, "input": kwargs,
    })


def log_agent_end(
    logger: logging.Logger, agent: str, status: str, duration_ms: float
) -> None:
    """Log the successful completion of an agent run."""
    _emit(
        logger,
        logging.INFO,
        f"AGENT_END    {agent}  status={status}  ({duration_ms:.0f} ms)",
        {"event": "agent_end", "agent": agent, "status": status,
         "duration_ms": round(duration_ms, 1)},
    )


# ── Decorator macros ───────────────────────────────────────────────────────────

def log_api_call(endpoint: str) -> Callable:
    """Decorator macro — logs request params and response for any API-calling function.

    Usage::

        @log_api_call("serpapi.flights")
        def _fetch_outbound(self, departure_airport, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(f"api.{endpoint}")
            log_api_request(logger, endpoint, kwargs)
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                log_api_response(logger, endpoint, result,
                                 (time.perf_counter() - t0) * 1000)
                return result
            except Exception as exc:
                logger.error(
                    "API_ERROR %s after %.0f ms: %s",
                    endpoint,
                    (time.perf_counter() - t0) * 1000,
                    exc,
                    exc_info=True,
                )
                raise
        return wrapper
    return decorator


def log_agent_run(agent_name: str) -> Callable:
    """Decorator macro — logs input kwargs, timing, and status for an agent method.

    Usage::

        @log_agent_run("FlightAgent")
        def search_and_recommend(self, departure_airport, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            logger = get_logger(f"agent.{agent_name.lower()}")
            log_agent_start(logger, agent_name, kwargs)
            t0 = time.perf_counter()
            try:
                result = func(self, *args, **kwargs)
                status = getattr(result, "status", "done")
                log_agent_end(logger, agent_name, status,
                              (time.perf_counter() - t0) * 1000)
                return result
            except Exception as exc:
                logger.error(
                    "AGENT_ERROR %s after %.0f ms: %s",
                    agent_name,
                    (time.perf_counter() - t0) * 1000,
                    exc,
                    exc_info=True,
                )
                raise
        return wrapper
    return decorator
