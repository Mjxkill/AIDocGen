#!/usr/bin/env python3
"""Re-mux an existing audiobook M4B/ZIP, dropping chapters whose title
matches a TOC pattern (or a custom skip-list).

Use this when an existing audiobook contains a Table-of-Contents chapter
read aloud — re-running the full pipeline would lose the synthesised audio
(~hours of GPU work). This rebuilds the M4B/ZIP from the per-chapter MP3s
already on disk.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


TOC_PATTERNS = [
    r"^table\s+des\s+mati[èe]res\s*$",
    r"^sommaire\s*$",
    r"^table\s+of\s+contents\s*$",
    r"^contents?\s*$",
    r"^toc\s*$",
    r"^index\s*$",
]
TOC_RE = re.compile("|".join(TOC_PATTERNS), re.IGNORECASE)


def is_toc_title(title: str) -> bool:
    return bool(TOC_RE.match(title.strip()))


def ffprobe_chapters(m4b: Path) -> list[dict]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_chapters", "-of", "json", str(m4b)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(r.stdout).get("chapters", [])


def find_chapter_mp3(chapters_dir: Path, idx_1based: int) -> Path:
    """Per-chapter MP3 filename starts with `NN_` (zero-padded). Returns
    the file matching the given 1-based chapter index."""
    prefix = f"{idx_1based:02d}_" if idx_1based < 100 else f"{idx_1based}_"
    matches = sorted(chapters_dir.glob(f"{prefix}*.mp3"))
    if matches:
        return matches[0]
    # Fall back: scan for any file whose numeric prefix matches.
    for p in sorted(chapters_dir.iterdir()):
        m = re.match(r"^(\d+)_", p.name)
        if m and int(m.group(1)) == idx_1based:
            return p
    raise FileNotFoundError(f"chapter {idx_1based} mp3 not found in {chapters_dir}")


def ffprobe_duration(p: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
        check=True, capture_output=True, text=True,
    )
    return float(r.stdout.strip() or 0)


def build_m4b(chapter_files: list[tuple[str, Path, float]],
              out_m4b: Path, title: str):
    work = out_m4b.parent / "_remux_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    list_file = work / "list.txt"
    list_file.write_text(
        "\n".join(f"file '{p}'" for _, p, _ in chapter_files), encoding="utf-8"
    )
    combined = work / "combined.m4a"
    print(f"[remux] encoding {len(chapter_files)} chapters to AAC…", flush=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c:a", "aac", "-b:a", "96k", str(combined)],
        check=True, capture_output=True, text=True,
    )
    meta_lines = [";FFMETADATA1", f"title={title}", "encoder=AIDocGen"]
    cursor = 0.0
    for ch_title, _, dur in chapter_files:
        start_ms = int(cursor * 1000)
        end_ms = int((cursor + dur) * 1000)
        meta_lines += ["[CHAPTER]", "TIMEBASE=1/1000",
                       f"START={start_ms}", f"END={end_ms}",
                       f"title={ch_title}"]
        cursor += dur
    meta_path = work / "meta.txt"
    meta_path.write_text("\n".join(meta_lines), encoding="utf-8")
    print(f"[remux] muxing M4B → {out_m4b}", flush=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(combined), "-i", str(meta_path),
         "-map", "0:a", "-map_metadata", "1", "-c", "copy", str(out_m4b)],
        check=True, capture_output=True, text=True,
    )
    shutil.rmtree(work)
    return cursor  # total duration


def build_zip(chapter_files: list[tuple[str, Path, float]], out_zip: Path,
              m4b_path: Path):
    """ZIP: chapter MP3s + the M4B."""
    import zipfile
    print(f"[remux] writing ZIP → {out_zip}", flush=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as zf:
        for _, p, _ in chapter_files:
            zf.write(p, arcname=f"chapters/{p.name}")
        zf.write(m4b_path, arcname=m4b_path.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir", help="path to data/audiobook_jobs/<id>/")
    ap.add_argument("--drop-titles", nargs="*", default=[],
                    help="Extra chapter titles to drop (case-insensitive)")
    ap.add_argument("--keep-toc", action="store_true",
                    help="Don't auto-drop chapters matching TOC patterns")
    args = ap.parse_args()

    job = Path(args.job_dir).resolve()
    if not job.exists():
        sys.exit(f"job dir not found: {job}")
    status_path = job / "status.json"
    if not status_path.exists():
        sys.exit(f"status.json missing: {status_path}")
    status = json.loads(status_path.read_text())

    title = status.get("title", "Audiobook")
    basename = status.get("basename", "audiobook")
    m4b_old = job / status.get("m4b_name", f"{basename}.m4b")
    if not m4b_old.exists():
        sys.exit(f"M4B not found: {m4b_old}")
    chapters_dir = job / "chapters"
    if not chapters_dir.is_dir():
        sys.exit(f"chapters/ dir missing: {chapters_dir}")

    chapters = ffprobe_chapters(m4b_old)
    print(f"[remux] {len(chapters)} chapters in source")
    extra_drop = {t.lower() for t in args.drop_titles}

    keep: list[tuple[str, Path, float]] = []
    dropped: list[str] = []
    for i, ch in enumerate(chapters):
        ch_title = ch.get("tags", {}).get("title", f"Chapter {i+1}")
        is_toc = (not args.keep_toc) and is_toc_title(ch_title)
        if is_toc or ch_title.lower() in extra_drop:
            dropped.append(f"  - #{i+1:>3}: {ch_title}")
            continue
        mp3 = find_chapter_mp3(chapters_dir, i + 1)
        dur = ffprobe_duration(mp3)
        keep.append((ch_title, mp3, dur))

    if not dropped:
        print("[remux] nothing to drop; aborting.")
        sys.exit(0)

    print(f"[remux] dropping {len(dropped)} chapter(s):")
    for d in dropped:
        print(d)
    print(f"[remux] keeping {len(keep)} chapter(s)")

    backup = m4b_old.with_suffix(m4b_old.suffix + ".old")
    shutil.move(str(m4b_old), str(backup))
    print(f"[remux] backup → {backup}")

    duration = build_m4b(keep, m4b_old, title)
    zip_old = job / status.get("zip_name", f"{basename}.zip")
    if zip_old.exists():
        shutil.move(str(zip_old), str(zip_old) + ".old")
    build_zip(keep, zip_old, m4b_old)

    status.update({
        "m4b_size": m4b_old.stat().st_size,
        "zip_size": zip_old.stat().st_size,
        "chapters_count": len(keep),
        "duration_seconds": round(duration, 1),
    })
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2))
    print(f"[remux] done. New duration: {duration/60:.1f} min — "
          f"{m4b_old.stat().st_size/1e9:.2f} GB M4B")


if __name__ == "__main__":
    main()
