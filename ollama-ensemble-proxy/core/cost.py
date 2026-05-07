"""Per-call cost tracker.

One JSON line per chargeable API call in `<DATA_DIR>/cost_log.jsonl`:
    {ts, user_id, user_name, job_id?, provider, kind, model?, units, cost_usd}

`record()` is best-effort and never raises — it must not break callers.
"""
import json
import time
from pathlib import Path
from typing import Any

from .logging_config import get_logger

log = get_logger("aidocgen.cost")

# Tariffs in USD. Approximate; tweak as providers update.
# Format: (provider, model, kind) -> $/unit
TARIFFS: dict[tuple[str, str, str], float] = {
    # DeepSeek API (May 2026 published rates)
    ("deepseek", "deepseek-v4-flash", "input_token"):  0.27e-6,
    ("deepseek", "deepseek-v4-flash", "output_token"): 1.10e-6,
    ("deepseek", "deepseek-v4-pro",   "input_token"):  0.55e-6,
    ("deepseek", "deepseek-v4-pro",   "output_token"): 2.19e-6,

    # Firecrawl: ~$0.001 per credit
    ("firecrawl", "scrape", "credit"): 0.001,
    ("firecrawl", "crawl",  "credit"): 0.001,
    ("firecrawl", "agent",  "credit"): 0.005,
    ("firecrawl", "parse",  "credit"): 0.001,
    ("firecrawl", "search", "credit"): 0.005,
    ("firecrawl", "map",    "credit"): 0.001,
    ("firecrawl", "extract","credit"): 0.001,

    # Tavily: $0.005 per advanced search query
    ("tavily", "search", "credit"): 0.005,

    # OpenAI TTS
    ("openai", "gpt-4o-mini-tts", "char"): 15.0e-6,  # $15 per 1M chars
    ("openai", "gpt-image-1",     "image"): 0.04,    # $0.04/image at 1024x1024
}


def estimate(provider: str, model: str, kind: str, units: float) -> float:
    """Look up the tariff and compute cost in USD; 0.0 if unknown."""
    rate = TARIFFS.get((provider, model, kind))
    if rate is None:
        # Try without model (some providers have flat rate per kind)
        rate = TARIFFS.get((provider, "*", kind), 0.0)
    return float(rate) * float(units)


def _log_path(data_dir: Path) -> Path:
    return Path(data_dir) / "cost_log.jsonl"


def record(data_dir: Path, *, provider: str, kind: str, units: float,
           user_id: str | None = None, user_name: str | None = None,
           job_id: str | None = None, model: str | None = None,
           cost_usd: float | None = None, **extra) -> None:
    """Append one cost entry. If cost_usd is not given, look it up from TARIFFS."""
    try:
        if cost_usd is None:
            cost_usd = estimate(provider, model or "*", kind, units)
        entry: dict[str, Any] = {
            "ts": int(time.time()),
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provider": provider,
            "kind": kind,
            "model": model,
            "units": units,
            "cost_usd": round(float(cost_usd), 6),
            "user_id": user_id,
            "user_name": user_name,
            "job_id": job_id,
        }
        if extra:
            entry["details"] = extra
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        with _log_path(Path(data_dir)).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("cost record failed", extra={"error": str(e), "provider": provider})


def _iter_entries(data_dir: Path):
    p = _log_path(Path(data_dir))
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def summary(data_dir: Path, *, user_id: str | None = None,
            since_ts: int | None = None) -> dict[str, Any]:
    """Aggregate cost entries.

    Returns:
        {
          "total_usd": float,
          "by_provider": {"deepseek": {"cost_usd": ..., "calls": N}, ...},
          "by_user": {user_id: {"cost_usd": ..., "calls": N, "user_name": ...}, ...},
          "calls": N,
        }
    `user_id` filters to a single user. `since_ts` filters by timestamp.
    """
    by_provider: dict[str, dict[str, Any]] = {}
    by_user: dict[str, dict[str, Any]] = {}
    total = 0.0
    calls = 0
    for e in _iter_entries(Path(data_dir)):
        if since_ts and e.get("ts", 0) < since_ts:
            continue
        if user_id and e.get("user_id") != user_id:
            continue
        cost = float(e.get("cost_usd") or 0.0)
        prov = e.get("provider", "?")
        by_provider.setdefault(prov, {"cost_usd": 0.0, "calls": 0})
        by_provider[prov]["cost_usd"] += cost
        by_provider[prov]["calls"] += 1
        if not user_id:
            uid = e.get("user_id") or "(anon)"
            by_user.setdefault(uid, {"cost_usd": 0.0, "calls": 0,
                                     "user_name": e.get("user_name")})
            by_user[uid]["cost_usd"] += cost
            by_user[uid]["calls"] += 1
        total += cost
        calls += 1
    # Round for nicer JSON
    for d in by_provider.values():
        d["cost_usd"] = round(d["cost_usd"], 4)
    for d in by_user.values():
        d["cost_usd"] = round(d["cost_usd"], 4)
    return {
        "total_usd": round(total, 4),
        "by_provider": by_provider,
        "by_user": by_user if not user_id else {},
        "calls": calls,
    }
