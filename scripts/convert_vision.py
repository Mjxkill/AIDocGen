#!/usr/bin/env python3
"""Appel vision à Ollama Cloud pour transcrire une page PDF rendue en PNG."""
import base64
import json
import sys
import urllib.request
from pathlib import Path

IMG_PATH = Path(sys.argv[1])
OUTPUT_PATH = Path(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwen3-vl:235b-cloud"

img_b64 = base64.b64encode(IMG_PATH.read_bytes()).decode("ascii")

PROMPT = (
    "Это скриншот с телефона или страница документа. "
    "Полностью транскрибируй весь видимый текст, сохраняя порядок и структуру "
    "(абзацы, списки, заголовки). Если это интерфейс приложения — "
    "игнорируй элементы UI (время, батарея, кнопки), но включи весь содержательный текст. "
    "Если есть изображения без текста — кратко опиши их одной строкой в формате "
    "[изображение: описание]. "
    "Выведи ТОЛЬКО Markdown, без преамбулы, без пояснений. "
    "Сохраняй русский язык если текст на русском."
)

payload = {
    "model": MODEL,
    "prompt": PROMPT,
    "images": [img_b64],
    "stream": False,
    "options": {"temperature": 0.1},
}

req = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(req, timeout=600) as resp:
    result = json.loads(resp.read().decode("utf-8"))

OUTPUT_PATH.write_text(result["response"], encoding="utf-8")
print(f"OK → {OUTPUT_PATH} ({len(result['response'])} chars, {result.get('total_duration', 0)/1e9:.1f}s)")
