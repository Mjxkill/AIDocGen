"""Append-only audit log for user actions.

One JSON line per event in `<DATA_DIR>/audit.log`. Read-only for everyone
except admins (via `/v1/audit/log`).
"""
import json
import time
from pathlib import Path

from .logging_config import get_logger

log = get_logger("aidocgen.audit")


def _audit_path(data_dir: Path) -> Path:
    return Path(data_dir) / "audit.log"


def record(data_dir: Path, *, action: str,
           actor_id: str | None = None, actor_name: str | None = None,
           target: str | None = None, ip: str | None = None,
           **details) -> None:
    """Append one event to the audit log. Never raises — audit must not break callers."""
    try:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": int(time.time()),
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": action,
            "actor_id": actor_id,
            "actor_name": actor_name,
            "target": target,
            "ip": ip,
        }
        if details:
            entry["details"] = details
        with _audit_path(Path(data_dir)).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        # Audit must never break the caller; surface the error in main logs only.
        log.warning("audit write failed", extra={"error": str(e), "action": action})


def tail(data_dir: Path, n: int = 200, action_filter: str | None = None) -> list[dict]:
    p = _audit_path(Path(data_dir))
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if action_filter and e.get("action") != action_filter:
                continue
            out.append(e)
    return out[-n:]
