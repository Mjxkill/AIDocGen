#!/usr/bin/env python3
"""Build EPUB3 books (text + audio + SMIL Media Overlays) from an existing
audiobook job, using the source dossier's planner.json as the canonical
chapter outline.

One EPUB3 per planner.master_outline entry. Audio is the concatenation of
the corresponding per-section MP3s already on disk (no re-TTS). A SMIL
media overlay maps each text section to its audio time range.

Output:
  <job_dir>/epub3/<basename>_part_NN_<title-slug>.epub
  <job_dir>/<basename>_epub3.zip   (bundle of all parts)
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile
from html import escape
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

sys.path.insert(0, str(Path(__file__).parent))
import build_audiobook as bb


# ---------- planner ↔ summary.md mapping ----------

def _norm_chapter_title(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip().lower()


def assign_audio_chapters_to_parts(summary_md: str, planner_chapter_titles: list[str]
                                   ) -> tuple[list[tuple[str, str]], list[int]]:
    """Returns (audio_chapters, part_index_per_audio_chapter) where:
      - audio_chapters: [(title, body)] from clean_markdown + split_into_chapters
      - part_index: list parallel to audio_chapters; each entry is the
        0-based index into planner_chapter_titles, or -1 if unassigned.
    The mapping is computed by walking the SUMMARY (after code-block strip)
    line by line, tracking the most recent `# Chapter : Sub` heading whose
    prefix matches a planner chapter title.
    """
    # Strip code blocks first (they contain fake '# ' / '## ' lines)
    md = re.sub(r"```[\s\S]*?```", "", summary_md)

    norm_planner = [_norm_chapter_title(t) for t in planner_chapter_titles]

    # Find each `## ` heading and what part it falls under
    current_part = -1  # unassigned by default
    part_for_h2: list[int] = []  # one entry per ## heading encountered
    for line in md.splitlines():
        m1 = re.match(r"^#\s+(.+)$", line)
        m2 = re.match(r"^##\s+(.+)$", line)
        if m1:
            head = m1.group(1).strip()
            head_norm = _norm_chapter_title(head)
            # Strip trailing ":" or " :", then check prefix against planner
            for j, pn in enumerate(norm_planner):
                if head_norm == pn or head_norm.startswith(pn + " "):
                    current_part = j
                    break
        elif m2:
            part_for_h2.append(current_part)

    # Now align with split_into_chapters output. split_into_chapters drops
    # TOC chapters and chapters whose body is empty after cleaning, so we
    # must apply the same filtering here.
    chapters = bb.split_into_chapters(bb.clean_markdown(summary_md))
    # Re-walk to know which ## indices ARE kept (matched 1:1 with chapters)
    # by replaying split_into_chapters' logic — easier to count h2s in
    # cleaned text and zip with chapters.
    cleaned = bb.clean_markdown(summary_md)
    cleaned_h2 = []
    for line in cleaned.splitlines():
        if re.match(r"^##\s+(.+)$", line):
            cleaned_h2.append(True)

    # Filter part_for_h2 down to what survives clean_markdown.
    # clean_markdown does NOT add or remove ## headings (it only strips
    # code blocks, which we already did, and Section: prefix which keeps
    # the header line). So part_for_h2 should already align with
    # cleaned_h2 length.
    if len(part_for_h2) != len(cleaned_h2):
        print(f"WARN: h2 count mismatch raw={len(part_for_h2)} "
              f"cleaned={len(cleaned_h2)}", file=sys.stderr)

    # split_into_chapters then drops TOC + empty-body. We re-derive chapters
    # carefully: same iteration, but also tracking part_idx for each chapter.
    lines_cleaned = cleaned.splitlines()
    chapters_with_part: list[tuple[str, str, int]] = []
    current_title = "Introduction"
    current_body: list[str] = []
    h2_seen = -1
    for ln in lines_cleaned:
        m = re.match(r"^(#{1,3})\s+(.+)$", ln)
        if m and len(m.group(1)) == 2:
            if current_body:
                chapters_with_part.append(
                    (current_title, "\n".join(current_body),
                     part_for_h2[h2_seen] if 0 <= h2_seen < len(part_for_h2) else -1))
            h2_seen += 1
            current_title = m.group(2).strip().rstrip(":") or "Chapitre"
            current_body = []
        else:
            current_body.append(ln)
    if current_body:
        chapters_with_part.append(
            (current_title, "\n".join(current_body),
             part_for_h2[h2_seen] if 0 <= h2_seen < len(part_for_h2) else -1))

    # Apply the body cleanup + TOC + empty filter that split_into_chapters does
    out: list[tuple[str, str]] = []
    parts: list[int] = []
    for title, body, part_idx in chapters_with_part:
        if bb.is_toc_title(title):
            continue
        body = re.sub(
            r"^(#{1,6})\s+(.+)$",
            lambda m: m.group(2).strip().rstrip(":") + ".",
            body, flags=re.M,
        )
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if body or out:
            out.append((title, body))
            parts.append(part_idx)
    return out, parts


# ---------- audio ----------

def ffprobe_duration(p: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
        check=True, capture_output=True, text=True,
    )
    return float(r.stdout.strip() or 0)


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


def concat_mp3(parts: list[Path], out: Path):
    list_file = out.parent / f"_cat_{out.stem}.txt"
    list_file.write_text("\n".join(f"file '{p}'" for p in parts), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c", "copy", str(out)],
        check=True, capture_output=True, text=True,
    )
    list_file.unlink(missing_ok=True)


# ---------- EPUB3 building ----------

CONTAINER_XML = '''<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/package.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
'''

STYLES_CSS = '''body { font-family: Georgia, serif; font-size: 1em; line-height: 1.55;
    margin: 1em; color: #222; }
h1 { font-size: 1.6em; margin-top: 1.2em; }
h2 { font-size: 1.25em; color: #555; margin-top: 1em;
    border-bottom: 1px solid #ddd; padding-bottom: .2em; }
p { margin: .5em 0; text-align: justify; }
section.audiobook-chapter { margin-bottom: 2em; }
.smil-active { background-color: #fffdd0; }
'''


def md_section_to_xhtml(title: str, body: str, section_id: str) -> str:
    """Convert a single audio chapter's title + body into XHTML markup.
    Each paragraph gets an id (for SMIL fine-grain sync if we ever do that;
    for v1 the SMIL syncs at section level only)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    parts = [f'<section id="{section_id}" class="audiobook-chapter" '
             f'epub:type="chapter">',
             f'  <h2>{escape(title)}</h2>']
    for i, p in enumerate(paragraphs):
        pid = f"{section_id}_p{i+1}"
        parts.append(f'  <p id="{pid}">{escape(p)}</p>')
    parts.append('</section>')
    return "\n".join(parts)


def build_xhtml(part_title: str, sections_xhtml: list[str], lang: str = "fr") -> str:
    body_inner = "\n".join(sections_xhtml)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="{lang}">
<head>
  <meta charset="UTF-8"/>
  <title>{escape(part_title)}</title>
  <link rel="stylesheet" type="text/css" href="styles.css"/>
</head>
<body>
  <h1>{escape(part_title)}</h1>
  {body_inner}
</body>
</html>
'''


def build_smil(audio_filename: str, sections: list[tuple[str, float, float]]) -> str:
    """sections: list of (section_id, start_seconds, end_seconds)."""
    pars = []
    for i, (sid, t0, t1) in enumerate(sections):
        pars.append(f'''    <par id="par{i+1}">
      <text src="content.xhtml#{sid}"/>
      <audio src="{audio_filename}" clipBegin="{t0:.3f}s" clipEnd="{t1:.3f}s"/>
    </par>''')
    pars_xml = "\n".join(pars)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<smil xmlns="http://www.w3.org/ns/SMIL"
      xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">
  <body>
{pars_xml}
  </body>
</smil>
'''


def build_nav(part_title: str, sections: list[tuple[str, str]]) -> str:
    """sections: [(section_id, title)]."""
    items = "\n".join(
        f'      <li><a href="content.xhtml#{sid}">{escape(t)}</a></li>'
        for sid, t in sections
    )
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="fr">
<head>
  <meta charset="UTF-8"/>
  <title>{escape(part_title)}</title>
</head>
<body>
  <nav epub:type="toc" id="toc">
    <h1>{escape(part_title)}</h1>
    <ol>
{items}
    </ol>
  </nav>
</body>
</html>
'''


def build_package_opf(part_title: str, identifier: str, lang: str,
                      total_duration: float) -> str:
    h = int(total_duration // 3600)
    m = int((total_duration % 3600) // 60)
    s = int(total_duration % 60)
    duration_str = f"{h:02d}:{m:02d}:{s:02d}.000"
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0"
         unique-identifier="bookid" prefix="media: http://www.idpf.org/epub/vocab/overlays/#">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="bookid">{xml_escape(identifier)}</dc:identifier>
    <dc:title>{xml_escape(part_title)}</dc:title>
    <dc:language>{lang}</dc:language>
    <dc:creator>AIDocGen</dc:creator>
    <meta property="dcterms:modified">2026-05-03T00:00:00Z</meta>
    <meta property="media:duration">{duration_str}</meta>
    <meta property="media:active-class">smil-active</meta>
    <meta property="media:narrator">XTTS-v2</meta>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="content" href="content.xhtml" media-type="application/xhtml+xml" media-overlay="overlay"/>
    <item id="overlay" href="overlay.smil" media-type="application/smil+xml"/>
    <item id="audio" href="audio.mp3" media-type="audio/mpeg"/>
    <item id="css" href="styles.css" media-type="text/css"/>
  </manifest>
  <spine>
    <itemref idref="content"/>
  </spine>
</package>
'''


def write_epub3(out_path: Path, part_title: str, identifier: str,
                content_xhtml: str, nav_xhtml: str, smil_xml: str,
                package_opf: str, audio_path: Path, styles_css: str = STYLES_CSS):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    with zipfile.ZipFile(out_path, "w") as zf:
        # mimetype must be first and uncompressed
        zi = zipfile.ZipInfo("mimetype")
        zi.compress_type = zipfile.ZIP_STORED
        zf.writestr(zi, "application/epub+zip")
        zf.writestr("META-INF/container.xml", CONTAINER_XML,
                    zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/package.opf", package_opf, zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/nav.xhtml", nav_xhtml, zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/content.xhtml", content_xhtml, zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/overlay.smil", smil_xml, zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/styles.css", styles_css, zipfile.ZIP_DEFLATED)
        zf.write(audio_path, "OEBPS/audio.mp3", zipfile.ZIP_STORED)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("planner_path", nargs="?", default=None,
                    help="Path to planner.json (defaults to "
                         "data/dossiers/<source_run_id>/planner.json)")
    ap.add_argument("--lang", default="fr")
    args = ap.parse_args()

    job = Path(args.job_dir).resolve()
    status = json.loads((job / "status.json").read_text())
    summary = (job / "summary.md").read_text(encoding="utf-8")
    chapters_dir = job / "chapters"
    basename = status.get("basename", "audiobook")
    source_run = status.get("source_run_id", "")

    if args.planner_path:
        planner_path = Path(args.planner_path)
    elif source_run:
        # job_dir is data/audiobook_jobs/<id>/, planner is data/dossiers/<run>/
        planner_path = (job.parent.parent / "dossiers" / source_run
                        / "planner.json")
    else:
        sys.exit("no source_run_id in status.json and no planner path given")
    if not planner_path.exists():
        sys.exit(f"planner not found: {planner_path}")
    planner = json.loads(planner_path.read_text())
    planner_chapters = [c["chapter_title"] for c in planner["master_outline"]]
    print(f"[epub3] planner has {len(planner_chapters)} parts")

    audio_chapters, parts_idx = assign_audio_chapters_to_parts(
        summary, planner_chapters
    )
    print(f"[epub3] {len(audio_chapters)} audio chapters mapped, "
          f"{sum(1 for p in parts_idx if p == -1)} unassigned")

    # Group audio chapters by part
    by_part: dict[int, list[int]] = {}
    for i, p in enumerate(parts_idx):
        by_part.setdefault(p, []).append(i)

    out_dir = job / "epub3"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    work_dir = job / "_epub3_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()

    epub_files: list[tuple[Path, str]] = []  # (path, human title)
    # -1 (unassigned) goes first as "Préambule"
    ordered_parts = []
    if -1 in by_part:
        ordered_parts.append(-1)
    ordered_parts.extend(sorted(p for p in by_part if p >= 0))

    for order_i, p in enumerate(ordered_parts):
        chapter_indices = by_part[p]
        if p == -1:
            part_title = "Préambule"
        else:
            part_title = planner_chapters[p]
        print(f"[epub3] part {order_i+1}/{len(ordered_parts)}: "
              f"{part_title!r} ({len(chapter_indices)} sections)")

        # Build XHTML and gather audio for this part
        section_xhtml_blocks: list[str] = []
        section_for_smil: list[tuple[str, float, float]] = []  # (id, t0, t1)
        nav_entries: list[tuple[str, str]] = []
        audio_parts: list[Path] = []
        cumulative = 0.0

        for sec_i in chapter_indices:
            title, body = audio_chapters[sec_i]
            section_id = f"sec_{sec_i+1:04d}"
            mp3 = find_chapter_mp3(chapters_dir, sec_i + 1)
            if mp3 is None or not mp3.exists():
                print(f"  WARN: missing mp3 for chapter #{sec_i+1}, "
                      f"skipping section in audio")
                continue
            dur = ffprobe_duration(mp3)
            section_xhtml_blocks.append(
                md_section_to_xhtml(title, body, section_id)
            )
            section_for_smil.append((section_id, cumulative, cumulative + dur))
            nav_entries.append((section_id, title))
            audio_parts.append(mp3)
            cumulative += dur

        if not section_xhtml_blocks:
            print(f"  → no audio, skipping part")
            continue

        # Concat audio for this part
        part_work = work_dir / f"part_{order_i+1:02d}"
        part_work.mkdir()
        audio_path = part_work / "audio.mp3"
        concat_mp3(audio_parts, audio_path)

        # Build files
        content_xhtml = build_xhtml(part_title, section_xhtml_blocks, args.lang)
        nav_xhtml = build_nav(part_title, nav_entries)
        smil_xml = build_smil("audio.mp3", section_for_smil)
        identifier = f"aidocgen:{basename}:part{order_i+1:02d}"
        package_opf = build_package_opf(part_title, identifier, args.lang,
                                         cumulative)

        slug = bb.slugify(part_title, max_len=40)
        epub_name = f"{basename}_part_{order_i+1:02d}_{slug}.epub"
        epub_path = out_dir / epub_name
        write_epub3(epub_path, part_title, identifier, content_xhtml,
                    nav_xhtml, smil_xml, package_opf, audio_path)
        epub_files.append((epub_path, part_title))
        print(f"  → {epub_path.name} ({epub_path.stat().st_size/1e6:.1f} MB, "
              f"{cumulative/60:.1f} min)")

    # Cleanup work dir
    shutil.rmtree(work_dir)

    # Bundle ZIP
    zip_path = job / f"{basename}_epub3.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for ep, _ in epub_files:
            zf.write(ep, arcname=ep.name)
    print(f"[epub3] bundle ZIP: {zip_path.name} "
          f"({zip_path.stat().st_size/1e9:.2f} GB)")

    # Update status.json
    status["epubs"] = [
        {"name": p.name, "size": p.stat().st_size, "title": title}
        for p, title in epub_files
    ]
    status["epubs_zip_name"] = zip_path.name
    status["epubs_zip_size"] = zip_path.stat().st_size
    (job / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2)
    )
    print(f"[epub3] done. {len(epub_files)} EPUB3(s) in {out_dir}")


if __name__ == "__main__":
    main()
