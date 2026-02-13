import re
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

def canonicalize_url(url: str) -> str:
    if not url: return ""
    url = re.sub(r"[?#].*$", "", url)
    return url.lower().rstrip("/")

def simple_tokens(text: str) -> list[str]:
    return re.findall(r"\w{4,}", text.lower())

def save_json(path: Path, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(path: Path) -> Any:
    if not path.exists(): return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except: return None

def markdown_anchor(text: str) -> str:
    # Accurate anchor generation for GitHub/Markdown compatibility
    anchor = text.lower().strip()
    anchor = re.sub(r"[^\w\s-]", "", anchor)
    anchor = re.sub(r"\s+", "-", anchor)
    return anchor

async def emit_progress(
    progress_cb: Callable[[dict[str, Any]], None] | None,
    run_dir: Path | None,
    stage: str,
    message: str
):
    event = {
        "timestamp": int(time.time()),
        "stage": stage,
        "message": message,
    }
    if run_dir:
        s_path = run_dir / "status.json"
        try:
            data = load_json(s_path) or {}
            data["stage"] = stage
            data["updated_at"] = int(time.time())
            if "events" not in data: data["events"] = []
            data["events"].append(event)
            # Limit events log to last 50 to avoid file bloat
            data["events"] = data["events"][-50:]
            save_json(s_path, data)
        except: pass
    if progress_cb:
        try: progress_cb(event)
        except: pass