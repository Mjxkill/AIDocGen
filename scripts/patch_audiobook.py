#!/usr/bin/env python3
"""Re-synthesize chapters of an existing audiobook whose source contained
audio-hostile patterns (HTML anchors, hashtag tags, code blocks,
"Section :" prefixes, etc.). Reuses chapters whose content didn't change.

Workflow:
  1. Read summary.md
  2. Split into chapters using the CURRENT (fixed) clean_markdown
  3. For each chapter, detect if its source contained problematic patterns
     that would have leaked into the old audio
  4. For dirty chapters: synthesize new MP3 from the cleaned text
  5. Re-mux M4B + ZIP from the (possibly updated) chapter MP3s

Existing chapter MP3s on disk are preserved when content didn't change.
The ones being re-synthesized get .old backups before being overwritten.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

# Reuse the build_audiobook helpers
sys.path.insert(0, str(Path(__file__).parent))
import build_audiobook as bb

# Patterns whose presence in the RAW summary.md (before cleaning) means a
# chapter's audio was poisoned and must be re-synthesised.
DIRTY_PATTERNS = [
    re.compile(r"<[a-zA-Z/!][^>]*>"),       # HTML tag
    re.compile(r"(?<![#\w])#[A-Za-zÀ-ÿ]"),  # hashtag-style topic marker
    re.compile(r"```"),                     # code block fence
    re.compile(r"\[code\]"),                # legacy code placeholder
    re.compile(r"^#{1,6}\s+Section\s*:", re.M),  # "Section :" prefix in heading
]


def split_raw_chapters(md: str) -> list[tuple[str, str]]:
    """Split summary.md the same way the synthesis pipeline does, but keep
    the body content RAW (HTML tags, hashtags, etc.) so we can detect
    dirtiness patterns. Mirrors split_into_chapters' chapter selection so
    indices align 1:1 with chapters_new from clean_markdown(summary)."""
    # Match the cleaning steps that affect chapter boundaries / titles:
    #  - code blocks must be stripped first (they contain stray '## ' lines
    #    that would create phantom chapters)
    #  - "Section :" prefix in headings is stripped so titles align
    md = re.sub(r"```[\s\S]*?```", "", md)
    md = re.sub(r"^(#{1,6}\s+)Section\s*:\s*", r"\1", md, flags=re.M)
    lines = md.splitlines()
    chapters: list[tuple[str, list[str]]] = []
    current_title = "Introduction"
    current_body: list[str] = []
    for ln in lines:
        m = re.match(r"^(#{1,3})\s+(.+)$", ln)
        if m and len(m.group(1)) == 2:
            if current_body:
                chapters.append((current_title, current_body))
            current_title = m.group(2).strip().rstrip(":") or "Chapitre"
            current_body = []
        else:
            current_body.append(ln)
    if current_body:
        chapters.append((current_title, current_body))
    out: list[tuple[str, str]] = []
    for t, body_lines in chapters:
        if bb.is_toc_title(t):
            continue
        body = "\n".join(body_lines).strip()
        if body or out:
            out.append((t, body))
    return out


def is_dirty(title: str, body: str) -> list[str]:
    """Return a list of pattern names matched, empty if clean."""
    full = f"## {title}\n{body}"
    hits = []
    for pat in DIRTY_PATTERNS:
        if pat.search(full):
            hits.append(pat.pattern[:30])
    return hits


def find_chapter_mp3(chapters_dir: Path, idx_1based: int) -> Path | None:
    matches = sorted(chapters_dir.glob(f"{idx_1based:02d}_*.mp3"))
    if matches:
        return matches[0]
    matches = sorted(chapters_dir.glob(f"{idx_1based:03d}_*.mp3"))
    if matches:
        return matches[0]
    for p in sorted(chapters_dir.iterdir()):
        m = re.match(r"^(\d+)_", p.name)
        if m and int(m.group(1)) == idx_1based:
            return p
    return None


def synth_chapter(body: str, voice: str, speed: float, host: str,
                  out_mp3: Path, language: str = "fr") -> None:
    """Synthesise a chapter body to a single MP3 by chunking + TTS."""
    chunks = bb.chunk_text(body, target=3000)
    parts: list[bytes] = []
    for j, c in enumerate(chunks):
        if not c.strip():
            continue
        audio = bb.tts_xtts(c, voice, speed, host, language)
        parts.append(audio)
        print(f"    chunk {j+1}/{len(chunks)}: {len(audio)} bytes",
              flush=True)
    out_mp3.write_bytes(b"".join(parts))


def ffprobe_duration(p: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
        check=True, capture_output=True, text=True,
    )
    return float(r.stdout.strip() or 0)


def slugify(s: str, max_len: int = 40) -> str:
    return bb.slugify(s, max_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--xtts-host", default="http://127.0.0.1:8021")
    ap.add_argument("--dry-run", action="store_true",
                    help="List dirty chapters, don't re-synthesise")
    args = ap.parse_args()

    job = Path(args.job_dir).resolve()
    status_path = job / "status.json"
    status = json.loads(status_path.read_text())
    summary = (job / "summary.md").read_text(encoding="utf-8")
    voice = status["voice"]
    speed = float(status.get("speed", 1.0))
    basename = status["basename"]
    title_doc = status["title"]
    chapters_dir = job / "chapters"

    # Compute new chapters using the FIXED clean_markdown
    cleaned = bb.clean_markdown(summary)
    chapters_new = bb.split_into_chapters(cleaned)
    # Pre-cleaning split for dirty-detection (so we see HTML tags etc.)
    chapters_raw_all = split_raw_chapters(summary)

    def _norm_title(t: str) -> str:
        t = re.sub(r"(?<![#\w])#(?=[A-Za-zÀ-ÿ])", "", t)
        for src, dst in bb._AUDIO_SYMBOL_REPLACEMENTS:
            t = t.replace(src, dst)
        return re.sub(r"\s+", " ", t).strip().lower()

    raw_lookup: dict[str, str] = {}
    for t_raw, body_raw in chapters_raw_all:
        raw_lookup.setdefault(_norm_title(t_raw), body_raw)

    pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []
    missing_match = 0
    for t_new, body_new in chapters_new:
        body_raw = raw_lookup.get(_norm_title(t_new))
        if body_raw is None:
            missing_match += 1
            body_raw = ""  # treat as clean if we can't find raw
        pairs.append(((t_new, body_new), (t_new, body_raw)))

    if missing_match:
        print(f"[patch] WARN: {missing_match} chapter(s) have no matching "
              f"raw body — assuming clean", file=sys.stderr)
    print(f"[patch] {len(pairs)} aligned chapters to inspect")

    dirty_indices: list[int] = []
    for i, ((title_new, body_new), (title_raw, body_raw)) in enumerate(pairs):
        hits = is_dirty(title_raw, body_raw)
        if hits:
            dirty_indices.append(i)

    print(f"[patch] dirty chapters: {len(dirty_indices)} / {len(pairs)}")
    print(f"[patch] first 10 dirty:")
    for i in dirty_indices[:10]:
        ti, _ = pairs[i][0]
        print(f"  #{i+1:>3}: {ti[:80]}")

    if args.dry_run:
        print("[patch] dry-run; no synthesis or remux")
        return

    if not dirty_indices:
        print("[patch] nothing to do; aborting")
        return

    # Re-synth dirty chapters
    started = time.time()
    for k, i in enumerate(dirty_indices):
        title_new, body_new = pairs[i][0]
        idx1 = i + 1
        out_mp3 = find_chapter_mp3(chapters_dir, idx1)
        if out_mp3 is None:
            stem = slugify(title_new)
            out_mp3 = chapters_dir / f"{idx1:02d}_{stem}.mp3"
        else:
            backup = out_mp3.with_suffix(out_mp3.suffix + ".old_dirty")
            if not backup.exists():
                shutil.move(str(out_mp3), str(backup))
        if not body_new.strip():
            print(f"[patch] {k+1}/{len(dirty_indices)} #{idx1} {title_new!r}: "
                  f"new body empty after cleaning, skipping (chapter dropped)",
                  flush=True)
            continue
        print(f"[patch] {k+1}/{len(dirty_indices)} #{idx1} {title_new[:60]!r} "
              f"({len(body_new)} chars)…", flush=True)
        synth_chapter(body_new, voice, speed, args.xtts_host, out_mp3)
        elapsed = time.time() - started
        rate = (k + 1) / max(0.001, elapsed)
        eta = (len(dirty_indices) - (k + 1)) / max(rate, 0.001) / 60
        print(f"  → done. avg {rate*60:.1f} ch/min, ETA {eta:.0f} min",
              flush=True)

    # Re-mux M4B + ZIP
    print(f"[patch] re-muxing M4B…", flush=True)
    chapter_files = []
    for i, (title_new, body_new) in enumerate(p[0] for p in pairs):
        if not body_new.strip():
            continue  # skip empty after clean
        idx1 = i + 1
        mp3 = find_chapter_mp3(chapters_dir, idx1)
        if mp3 is None or not mp3.exists():
            print(f"  WARN: chapter #{idx1} mp3 missing, skipping")
            continue
        chapter_files.append((title_new, mp3, ffprobe_duration(mp3)))

    out_m4b = job / status["m4b_name"]
    out_zip = job / status["zip_name"]
    if out_m4b.exists():
        shutil.move(str(out_m4b), str(out_m4b) + ".pre_patch")
    if out_zip.exists():
        shutil.move(str(out_zip), str(out_zip) + ".pre_patch")
    bb.make_m4b(chapter_files, out_m4b, title_doc)

    print(f"[patch] writing ZIP…", flush=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as zf:
        for _, p, _ in chapter_files:
            zf.write(p, arcname=f"chapters/{p.name}")
        zf.write(out_m4b, arcname=out_m4b.name)

    duration = sum(d for _, _, d in chapter_files)
    status.update({
        "m4b_size": out_m4b.stat().st_size,
        "zip_size": out_zip.stat().st_size,
        "chapters_count": len(chapter_files),
        "duration_seconds": round(duration, 1),
    })
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2))
    print(f"[patch] done. {len(chapter_files)} chapters, "
          f"{duration/60:.1f} min, "
          f"{out_m4b.stat().st_size/1e9:.2f} GB M4B")


if __name__ == "__main__":
    main()
