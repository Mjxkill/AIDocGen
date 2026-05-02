#!/usr/bin/env python3
"""Appel propre à Ollama via HTTP, sans pollution TTY."""
import json
import sys
import urllib.request
from pathlib import Path

PROMPT_SYSTEM = (Path(__file__).parent / "prompt_md.txt").read_text(encoding="utf-8")
RAW_TEXT = Path(sys.argv[1]).read_text(encoding="utf-8")
OUTPUT_PATH = Path(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else "glm-5.1:cloud"

payload = {
    "model": MODEL,
    "prompt": PROMPT_SYSTEM + RAW_TEXT,
    "stream": False,
    "think": False,
    "options": {"temperature": 0.1, "num_ctx": 32768},
}

req = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(req, timeout=900) as resp:
    result = json.loads(resp.read().decode("utf-8"))

OUTPUT_PATH.write_text(result["response"], encoding="utf-8")
print(f"OK → {OUTPUT_PATH} ({len(result['response'])} chars)")
print(f"eval_count={result.get('eval_count')} total_duration={result.get('total_duration', 0)/1e9:.1f}s")
