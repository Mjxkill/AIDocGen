#!/usr/bin/env python3
"""Fichiers énormes : pdftotext → chunks → LLM cleanup en parallèle → concat."""
import concurrent.futures as cf
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

SRC_PDF = sys.argv[1]
OUT_MD = Path(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else "glm-5.1:cloud"
TARGET_CHUNK_BYTES = 40000  # ~10-15K tokens
PAR = 5

SYSTEM_PROMPT = (
    "Ты получаешь фрагмент сырого текста, извлечённого из PDF (заметки Apple Notes на русском).\n"
    "Задача:\n"
    "1. Привести к чистому Markdown: заголовки (##, ###), списки (-)\n"
    "2. Исправить артефакты извлечения из Apple Notes: 'ĸ' (U+0138) → 'к' (кириллица), "
    "другие подобные ошибки кодировки шрифта\n"
    "3. Убрать номера страниц, повторяющиеся колонтитулы, лишние пустые строки\n"
    "4. Склеить абзацы, разорванные переносом\n\n"
    "СТРОГО:\n"
    "- НЕ перефразируй, НЕ пересказывай, НЕ сокращай\n"
    "- НЕ добавляй комментариев\n"
    "- Сохраняй русский язык, ВСЁ содержание дословно\n"
    "- Верни ТОЛЬКО Markdown\n\n"
    "Вот фрагмент:\n\n---\n"
)


def extract_text(pdf: str) -> str:
    r = subprocess.run(
        ["pdftotext", "-layout", pdf, "-"],
        check=True, capture_output=True, text=True,
    )
    return r.stdout


def chunk_by_pages(raw: str, target: int) -> list[str]:
    """Split text by form feeds (pages), then greedy-pack into chunks of ≤target bytes."""
    pages = raw.split("\f")
    chunks = []
    cur = []
    cur_size = 0
    for page in pages:
        psize = len(page.encode("utf-8"))
        if cur and cur_size + psize > target:
            chunks.append("\n\n".join(cur))
            cur = [page]
            cur_size = psize
        else:
            cur.append(page)
            cur_size += psize
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def process_chunk(idx: int, chunk: str) -> tuple[int, str]:
    payload = {
        "model": MODEL,
        "prompt": SYSTEM_PROMPT + chunk,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1, "num_ctx": 32768},
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            text = result["response"].strip()
            print(f"  chunk {idx:3d}: in={len(chunk)}B out={len(text)}B ({result.get('total_duration',0)/1e9:.0f}s)")
            return idx, text
        except Exception as e:
            print(f"  chunk {idx:3d} attempt {attempt+1}: {e}")
    return idx, f"[ERROR chunk {idx}]\n{chunk}"


def main():
    print(f"extracting {SRC_PDF}...")
    raw = extract_text(SRC_PDF)
    print(f"  {len(raw)} chars / {len(raw.encode('utf-8'))} bytes")

    chunks = chunk_by_pages(raw, TARGET_CHUNK_BYTES)
    print(f"  split into {len(chunks)} chunks")

    results: dict[int, str] = {}
    with cf.ThreadPoolExecutor(max_workers=PAR) as ex:
        futures = [ex.submit(process_chunk, i, c) for i, c in enumerate(chunks)]
        for fut in cf.as_completed(futures):
            i, text = fut.result()
            results[i] = text

    title = Path(SRC_PDF).stem
    parts = [f"# {title}\n"]
    for i in sorted(results):
        parts.append(results[i])
    OUT_MD.write_text("\n\n".join(parts), encoding="utf-8")
    print(f"\nDONE → {OUT_MD} ({OUT_MD.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
