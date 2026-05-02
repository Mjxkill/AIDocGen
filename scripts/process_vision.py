#!/usr/bin/env python3
"""PDF (scan ou screenshots) → rendu pages PNG → transcription vision → Markdown."""
import base64
import concurrent.futures as cf
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

if len(sys.argv) < 3:
    print("usage: process_vision.py <input.pdf> <output.md> [model] [parallelism]")
    sys.exit(1)

PDF = sys.argv[1]
OUT_MD = Path(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwen3-vl:235b-cloud"
PAR = int(sys.argv[4]) if len(sys.argv) > 4 else 5

PROMPT = (
    "Это страница PDF-документа или скриншот из приложения. "
    "Полностью транскрибируй весь видимый содержательный текст, сохраняя порядок и структуру "
    "(абзацы, списки, заголовки). Если это UI приложения — игнорируй элементы интерфейса "
    "(время, батарея, название приложения, кнопки навигации), но включи весь содержательный текст. "
    "Для изображений без текста дай краткое описание одной строкой: [изображение: описание]. "
    "Выведи ТОЛЬКО Markdown, без преамбулы. "
    "Сохраняй язык оригинала."
)


def render_all(pdf: str, tmp: Path) -> list[Path]:
    print(f"rendering {pdf} ...", flush=True)
    subprocess.run(
        ["pdftoppm", "-r", "120", "-png", pdf, str(tmp / "p")],
        check=True, capture_output=True,
    )
    pages = sorted(tmp.glob("p-*.png"))
    print(f"{len(pages)} pages rendered", flush=True)
    return pages


def vision_page(img_path: Path) -> tuple[int, str]:
    page_num = int(img_path.stem.split("-")[-1])
    img_b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
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
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            text = result["response"].strip()
            print(f"  page {page_num:3d}: {len(text)} chars", flush=True)
            return page_num, text
        except Exception as e:
            print(f"  page {page_num:3d} attempt {attempt+1} failed: {e}", flush=True)
    return page_num, f"[ERROR: could not process page {page_num}]"


def main():
    title = Path(PDF).stem
    with tempfile.TemporaryDirectory(prefix="vision_") as tmp_str:
        tmp = Path(tmp_str)
        pages = render_all(PDF, tmp)
        results: dict[int, str] = {}
        with cf.ThreadPoolExecutor(max_workers=PAR) as ex:
            futures = [ex.submit(vision_page, p) for p in pages]
            for fut in cf.as_completed(futures):
                num, text = fut.result()
                results[num] = text

    parts = [f"# {title}\n"]
    for num in sorted(results):
        parts.append(f"\n## Страница {num}\n")
        parts.append(results[num])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(parts), encoding="utf-8")
    print(f"\nDONE → {OUT_MD} ({OUT_MD.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
