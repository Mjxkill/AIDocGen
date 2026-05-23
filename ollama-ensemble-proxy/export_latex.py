"""
Export dossier to PDF via Pandoc + LuaLaTeX.

Reads sections.json to build a clean Markdown with proper heading hierarchy,
generates a BibTeX bibliography from corpus sources,
then delegates all Markdown->LaTeX conversion to Pandoc.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

TEMPLATE = Path(__file__).parent / "pandoc-template.tex"
UNICODE_STY = Path(__file__).parent / "unicode-support.sty"
PUPPETEER_CONFIG = Path(__file__).parent / "puppeteer-config.json"


def _render_mermaid_blocks(markdown_text: str, out_dir: Path) -> str:
    """Extract ```mermaid blocks, render to PNG via mmdc, replace with image refs."""
    import re
    import shutil as _shutil
    if not _shutil.which("mmdc"):
        return markdown_text

    pattern = re.compile(r'```mermaid\n(.*?)\n```', re.DOTALL)
    counter = [0]
    images_dir = out_dir / "diagrams"
    images_dir.mkdir(exist_ok=True)

    def replace(match: re.Match) -> str:
        counter[0] += 1
        idx = counter[0]
        mmd_code = match.group(1)
        # Force vertical layout to avoid text rotation issues with chromium
        mmd_code = re.sub(r'^\s*graph\s+(LR|RL)\b', 'graph TD', mmd_code, flags=re.MULTILINE)
        mmd_code = re.sub(r'^\s*flowchart\s+(LR|RL)\b', 'flowchart TD', mmd_code, flags=re.MULTILINE)
        # Remove \textbackslash{}  artifacts from LaTeX escaping
        mmd_code = mmd_code.replace(r'\textbackslash{}', '\\')
        # Quote node labels containing parentheses so mermaid doesn't
        # interpret them as shape delimiters:  A[foo (bar)] -> A["foo (bar)"]
        mmd_code = re.sub(
            r'(\w)\[([^\]"]*\([^\]]*)\]',
            lambda m: f'{m.group(1)}["{m.group(2)}"]',
            mmd_code,
        )
        mmd_file = images_dir / f"diagram_{idx}.mmd"
        png_file = images_dir / f"diagram_{idx}.png"
        mmd_file.write_text(mmd_code, encoding="utf-8")
        try:
            # Prepend config to force proper fonts (DejaVu available on host)
            mmd_with_config = "%%{init: {'theme': 'default', 'themeVariables': {'fontFamily': 'DejaVu Sans, sans-serif', 'fontSize': '18px'}}}%%\n" + mmd_code
            mmd_file.write_text(mmd_with_config, encoding="utf-8")
            result = subprocess.run(
                ["mmdc", "-i", str(mmd_file.resolve()), "-o", str(png_file.resolve()),
                 "-p", str(PUPPETEER_CONFIG), "-b", "white", "-w", "1400", "-H", "900"],
                capture_output=True, timeout=60, cwd=str(out_dir),
            )
            if png_file.exists():
                # Force in-place placement, limit height to avoid full-page figures
                return (
                    f"\n\n```{{=latex}}\n"
                    f"\\begin{{figure}}[H]\n"
                    f"\\centering\n"
                    f"\\includegraphics[width=0.55\\textwidth,height=0.35\\textheight,keepaspectratio]"
                    f"{{diagrams/diagram_{idx}.png}}\n"
                    f"\\caption*{{Diagramme {idx}}}\n"
                    f"\\end{{figure}}\n"
                    f"```\n\n"
                )
            else:
                print(f"Mermaid render failed for diagram {idx}: {result.stderr.decode()[:200]}")
                return f"\n\n```\n{mmd_code}\n```\n\n"
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"Mermaid timeout/error: {e}")
            return f"\n\n```\n{mmd_code}\n```\n\n"

    return pattern.sub(replace, markdown_text)


# Lazy `(.*?)` + backtracking handles `]` inside the alt-text (eg "[News] ...").
# `[^\]]*` would refuse the bracket and silently skip these images.
_IMG_MD_RE = re.compile(r'!\[(.*?)\]\((https?://[^\s)]+)\)')

# Match: ![alt](path){...?}   optionally followed by an italic caption block
_IMG_BLOCK_RE = re.compile(
    r'!\[(.*?)\]\(([^)\s]+)\)(?:\{[^}]*\})?'  # image markdown
    r'(?:[ \t]*\n+[ \t]*\*[^*\n][^*]*\*[ \t]*)?',  # optional italic caption block right after
)


def _dedupe_section_images(md: str) -> str:
    """Supprime les images dupliquées (même path) au sein d'une même section H2.

    Le writer peut injecter la même image deux fois dans une section
    (fallbacks multiples convergent vers la même source). On ne garde
    que la première occurrence par path dans chaque section ``## ...``.
    """
    parts = re.split(r'(?m)^(## .*)$', md)
    # parts = [preamble, heading, content, heading, content, ...]
    out = [parts[0]]
    removed = 0
    for i in range(1, len(parts), 2):
        heading = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        seen: set[str] = set()

        def repl(m: re.Match) -> str:
            nonlocal removed
            path = m.group(2)
            if path in seen:
                removed += 1
                return ""
            seen.add(path)
            return m.group(0)

        new_content = _IMG_BLOCK_RE.sub(repl, content)
        out.append(heading)
        out.append(new_content)
    if removed:
        print(f"Dedupe: {removed} doublons d'image supprimés (section-level)")
    return "".join(out)


def _download_remote_images(markdown_text: str, out_dir: Path) -> str:
    """Download ![alt](http(s)://...) image URLs to local files and rewrite refs.

    pandoc + pdflatex cannot fetch remote images. Without a local copy the URL
    leaks into the PDF as plain text. We cache by URL hash so re-runs are cheap.
    On any download/HTTP error we drop the image entirely (keeping nothing) so
    the PDF doesn't show a stray URL.
    """
    try:
        import requests
    except Exception:
        return markdown_text

    images_dir = out_dir / "images_dl"
    images_dir.mkdir(exist_ok=True)
    headers = {
        "User-Agent": "AIDocGen/1.0 (+https://electrosens.fr)",
        "Accept": "image/*,*/*;q=0.8",
    }
    cache: dict[str, str | None] = {}

    def _ext_from(url: str, content_type: str) -> str:
        ct = (content_type or "").lower().split(";")[0].strip()
        mapping = {
            "image/jpeg": ".jpg", "image/jpg": ".jpg",
            "image/png": ".png", "image/gif": ".gif",
            "image/webp": ".webp", "image/svg+xml": ".svg",
            "image/bmp": ".bmp", "image/tiff": ".tiff",
        }
        if ct in mapping:
            return mapping[ct]
        # fallback: try extension from URL path
        m = re.search(r'\.(jpe?g|png|gif|webp|svg|bmp|tiff)(?:\?|$)', url, re.IGNORECASE)
        if m:
            ext = m.group(1).lower()
            return ".jpg" if ext == "jpeg" else f".{ext}"
        return ".img"

    def _convert_webp_to_png(webp_path: Path) -> Path | None:
        """pdflatex ne supporte pas .webp; on convertit en PNG via dwebp."""
        png_path = webp_path.with_suffix(".png")
        if png_path.exists() and png_path.stat().st_size > 256:
            try:
                webp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return png_path
        try:
            r = subprocess.run(
                ["dwebp", str(webp_path), "-o", str(png_path)],
                capture_output=True, timeout=30,
            )
            if r.returncode == 0 and png_path.exists() and png_path.stat().st_size > 256:
                try:
                    webp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return png_path
        except Exception:
            pass
        return None

    def fetch(url: str) -> str | None:
        if url in cache:
            return cache[url]
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]
        # check if already cached on disk
        for existing in images_dir.glob(f"{h}.*"):
            cache[url] = f"images_dl/{existing.name}"
            return cache[url]
        try:
            r = requests.get(url, headers=headers, timeout=20, stream=True)
            if r.status_code != 200:
                cache[url] = None
                return None
            ct = r.headers.get("Content-Type", "")
            if "image" not in ct.lower():
                # not an image (e.g. HTML error page) — drop
                cache[url] = None
                return None
            ext = _ext_from(url, ct)
            # skip SVG (pdflatex doesn't render it without extra tooling)
            if ext == ".svg":
                cache[url] = None
                return None
            fname = f"{h}{ext}"
            fpath = images_dir / fname
            with open(fpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            if fpath.stat().st_size < 256:
                # too small / probably broken
                try:
                    fpath.unlink()
                except Exception:
                    pass
                cache[url] = None
                return None
            # pdflatex ne supporte pas .webp -> on convertit en PNG
            if ext == ".webp":
                png = _convert_webp_to_png(fpath)
                if png is not None:
                    cache[url] = f"images_dl/{png.name}"
                    return cache[url]
                cache[url] = None
                return None
            cache[url] = f"images_dl/{fname}"
            return cache[url]
        except Exception:
            cache[url] = None
            return None

    def repl(m: re.Match) -> str:
        alt = m.group(1)
        url = m.group(2)
        local = fetch(url)
        if local:
            # Raw LaTeX bloc figure[H] : place forcée, pas de flottement.
            # (pandoc par défaut omet [H] et la figure dérive loin de son ancre.)
            # Le caption italique du writer reste en paragraphe en-dessous.
            return (
                "\n\n```{=latex}\n"
                "\\begin{figure}[H]\n"
                "\\centering\n"
                "\\includegraphics[width=0.85\\textwidth,height=0.55\\textheight,keepaspectratio]"
                f"{{{local}}}\n"
                "\\end{figure}\n"
                "```\n\n"
            )
        # drop image entirely — keep alt text as plain italic for context
        return f"*{alt}*" if alt.strip() else ""

    new_md = _IMG_MD_RE.sub(repl, markdown_text)
    ok = sum(1 for v in cache.values() if v)
    fail = sum(1 for v in cache.values() if not v)
    print(f"Remote images: {ok} downloaded, {fail} dropped (total {len(cache)})")
    return new_md


def _clean_heading_text(text: str) -> str:
    """Strip markdown bold/hash from a heading used in structure."""
    if not text:
        return ""
    text = re.sub(r"#+\s*", "", text)
    text = text.replace("**", "").strip()
    return text


def _strip_emojis(text: str) -> str:
    """Remove emoji characters that may cause issues."""
    return re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        r"\U0001F900-\U0001F9FF\U00002702-\U000027B0"
        r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\uFE0F]",
        "",
        text,
    )


def _clean_section_content(content: str) -> str:
    """Clean section content for pandoc consumption."""
    if not content:
        return ""

    # Fix stray Windows-1252 bytes
    content = content.replace("\u0080", "").replace("\u0099", "")
    content = content.replace("\u0093", '"').replace("\u0094", '"')
    content = _strip_emojis(content)

    # Replace emojis that look like checkmarks/crosses with text
    content = content.replace("\u2705", "\u2713").replace("\u274C", "\u2717")
    content = content.replace("\u26A0", "[!]").replace("\u2B50", "*")

    # Escape literal \n, \t etc. in prose
    content = re.sub(r'(?<!\\)\\n(?![a-zA-Z])', r'\\textbackslash{}n', content)
    content = re.sub(r'(?<!\\)\\t(?![a-zA-Z])', r'\\textbackslash{}t', content)
    content = re.sub(r'(?<!\\)\\r(?![a-zA-Z])', r'\\textbackslash{}r', content)

    # Convert key insight markers to tcolorbox LaTeX
    # Pattern: > **Point cle:** or > **Key Insight:** etc.
    content = _convert_insight_boxes(content)

    lines = content.split("\n")
    cleaned = []
    skip_leading_blanks = True

    for line in lines:
        stripped = line.strip()

        # Skip HTML anchors
        if stripped.startswith("<a name=") or stripped.startswith("<a id="):
            continue

        # Skip headings
        if re.match(r"^\s*#{1,6}\s", line):
            continue

        # Skip horizontal rules
        if re.match(r"^\s*-{3,}\s*$", stripped) or re.match(r"^\s*\*{3,}\s*$", stripped):
            cleaned.append("")
            continue

        if skip_leading_blanks and stripped == "":
            continue
        skip_leading_blanks = False
        cleaned.append(line)

    result = "\n".join(cleaned).rstrip()

    # Ensure blank lines before lists and code blocks
    result = re.sub(r'([^\n])\n(```)', r'\1\n\n\2', result)
    result = re.sub(r'(```)\n([^\n`])', r'\1\n\n\2', result)
    result = re.sub(r'([^\n])\n(\d+\.\s)', r'\1\n\n\2', result)
    result = re.sub(r'([^\n])\n(- )', r'\1\n\n- ', result)
    result = re.sub(r'([^\n|])\n(\|[^\n]*\|)', r'\1\n\n\2', result)

    return result


def _md_to_latex_inline(text: str) -> str:
    """Convert markdown inline formatting to LaTeX for raw blocks."""
    # Escape LaTeX special chars (but not backslash from our own escapes)
    for ch in ['&', '%', '#']:
        text = text.replace(ch, f'\\{ch}')
    # Underscores: only escape those NOT inside \texttt{}
    text = re.sub(r'(?<!\\)_', r'\\_', text)
    # Convert `code` to \texttt{code}
    text = re.sub(r'`([^`]+)`', r'\\texttt{\1}', text)
    # Convert **bold** to \textbf{bold}
    text = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', text)
    # Convert *italic* to \textit{italic}
    text = re.sub(r'\*([^*]+)\*', r'\\textit{\1}', text)
    return text


def _convert_insight_boxes(content: str) -> str:
    """Convert markdown blockquote patterns to LaTeX tcolorbox environments."""
    # Pattern: > **Point cle / Key Insight / Important / Attention**
    # Convert to raw LaTeX tcolorbox blocks

    def replace_insight(match):
        box_type = match.group(1).strip().lower()
        box_content = match.group(2).strip()
        # Remove > prefix from each line
        box_content = re.sub(r'^>\s?', '', box_content, flags=re.MULTILINE)
        # Convert markdown inline to LaTeX (since this goes in a raw block)
        box_content = _md_to_latex_inline(box_content)

        if any(w in box_type for w in ["attention", "warning", "danger", "prudence"]):
            env = "warningbox"
            title = "Attention"
        elif any(w in box_type for w in ["succes", "success", "resultat", "result", "conclusion"]):
            env = "successbox"
            title = match.group(1).strip()
        else:
            env = "keyinsightbox"
            title = match.group(1).strip()

        return (
            f"\n\n```{{=latex}}\n"
            f"\\begin{{{env}}}[{title}]\n"
            f"{box_content}\n"
            f"\\end{{{env}}}\n"
            f"```\n\n"
        )

    # Match: > **Title:** content (possibly multi-line with >)
    content = re.sub(
        r'>\s*\*\*([^*]+)\*\*[:\s]*\n((?:>.*\n?)+)',
        replace_insight,
        content
    )

    return content


def _escape_yaml(text: str) -> str:
    """Escape a string for YAML double-quoted value."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _escape_bibtex(text: str) -> str:
    """Escape special characters for BibTeX."""
    text = text.replace("&", "\\&")
    text = text.replace("%", "\\%")
    text = text.replace("#", "\\#")
    text = text.replace("_", "\\_")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    return text


def _generate_bibtex(base_path: Path) -> None:
    """Generate references.bib from corpus.json sources."""
    corpus_path = base_path / "corpus.json"
    if not corpus_path.exists():
        # Create empty bib file
        (base_path / "references.bib").write_text("", encoding="utf-8")
        return

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    sources = corpus.get("sources", [])

    bib_entries = []
    for idx, src in enumerate(sources, 1):
        sid = src.get("source_id", f"src{idx}")
        title = _escape_bibtex(src.get("title", "Sans titre"))
        url = src.get("canonical_url", src.get("url", ""))
        domain = src.get("domain", "")

        bib_entry = (
            f"@online{{{sid},\n"
            f"  title = {{{title}}},\n"
            f"  author = {{{domain}}},\n"
            f"  url = {{{url}}},\n"
            f"  urldate = {{2026-03-31}},\n"
            f"  year = {{2026}},\n"
            f"}}\n"
        )
        bib_entries.append(bib_entry)

    (base_path / "references.bib").write_text(
        "\n".join(bib_entries), encoding="utf-8"
    )


def generate_latex(run_id: str, data_dir: str = "data/dossiers") -> None:
    """Generate PDF from a dossier run using Pandoc + LuaLaTeX."""
    base_path = Path(data_dir) / run_id
    planner = json.loads((base_path / "planner.json").read_text(encoding="utf-8"))
    sections = json.loads((base_path / "sections.json").read_text(encoding="utf-8"))["sections"]

    doc_title = _clean_heading_text(
        planner.get("question_reformulated", "Dossier de Recherche")
    )

    # ── Generate BibTeX bibliography ──
    _generate_bibtex(base_path)

    # ── Build clean Markdown with YAML frontmatter ──
    md = []
    md.append("---")
    md.append(f'title: "{_escape_yaml(doc_title)}"')
    md.append("author: AIDocGen")
    from datetime import date
    md.append(f"date: \"{date.today().strftime('%d/%m/%Y')}\"")
    md.append("lang: fr")
    md.append("toc: true")
    md.append("toc-depth: 3")
    md.append("---")
    md.append("")

    # ── Insert executive summary if exists ──
    exec_summary_sections = [s for s in sections if "resume" in s.get("s_title", "").lower() or "executive" in s.get("s_title", "").lower()]
    report_md_path = base_path / "report.md"
    if report_md_path.exists():
        report_content = report_md_path.read_text(encoding="utf-8")
        # Extract executive summary from report
        exec_match = re.search(r'## Resume Executif\s*\n(.*?)(?=\n---|\n## )', report_content, re.DOTALL)
        if exec_match:
            exec_text = _clean_section_content(exec_match.group(1).strip())
            md.append("# Resume Executif {.unnumbered}")
            md.append("")
            md.append("```{=latex}")
            md.append("\\vspace{-0.5em}\\noindent\\rule{\\textwidth}{0.4pt}\\vspace{0.5em}")
            md.append("```")
            md.append("")
            md.append(exec_text)
            md.append("")
            md.append("```{=latex}")
            md.append("\\vspace{0.5em}\\noindent\\rule{\\textwidth}{0.4pt}")
            md.append("\\clearpage")
            md.append("```")
            md.append("")

    current_part = ""
    current_chapter = ""

    for s in sections:
        p_title = _clean_heading_text(s.get("p_title", ""))
        c_title = _clean_heading_text(s.get("c_title", ""))
        s_title = _clean_heading_text(s.get("s_title", ""))
        content = _clean_section_content(s.get("content", ""))

        # ── Part (raw LaTeX block) ──
        if p_title and p_title != current_part:
            md.append("")
            md.append("```{=latex}")
            md.append(f"\\part{{{p_title}}}")
            md.append("```")
            md.append("")
            current_part = p_title
            current_chapter = ""

        # ── Chapter (H1) ──
        if c_title and c_title != current_chapter:
            md.append(f"# {c_title}")
            md.append("")
            current_chapter = c_title

        # ── Section (H2) ──
        md.append(f"## {s_title}")
        md.append("")

        if content:
            md.append(content)
            md.append("")

    pandoc_md = "\n".join(md)

    # ── PREPROCESS MERMAID BLOCKS -> PNG IMAGES ──
    pandoc_md = _render_mermaid_blocks(pandoc_md, base_path)

    # ── INJECT TIKZ REGISTER DIAGRAMS ──
    try:
        from export_tikz import detect_and_convert_register_tables
        pandoc_md = detect_and_convert_register_tables(pandoc_md)
    except Exception as e:
        print(f"TikZ injection failed: {e}")

    # ── DEDUP images dupliquées dans une même section H2 ──
    # (fait avant le download pour matcher la forme ![alt](url))
    pandoc_md = _dedupe_section_images(pandoc_md)

    # ── DOWNLOAD REMOTE IMAGES so pandoc/pdflatex can embed them ──
    pandoc_md = _download_remote_images(pandoc_md, base_path)

    # Write the clean markdown
    pandoc_input = base_path / "report_pandoc.md"
    pandoc_input.write_text(pandoc_md, encoding="utf-8")

    # ── Copy unicode-support.sty to build dir ──
    import shutil
    if UNICODE_STY.exists():
        shutil.copy2(UNICODE_STY, base_path / "unicode-support.sty")

    # ── Step 1: Pandoc -> .tex ──
    template = str(TEMPLATE.resolve()) if TEMPLATE.exists() else None
    pandoc_cmd = [
        "pandoc",
        "report_pandoc.md",
        "-o", "report.tex",
        "--from=markdown+tex_math_single_backslash+raw_tex",
        "--top-level-division=chapter",
        "--toc", "--toc-depth=3",
        "--variable=documentclass:scrreprt",
        "--variable=classoption:11pt,a4paper",
        "--variable=lang:fr",
        "--pdf-engine=pdflatex",
    ]
    if template:
        pandoc_cmd.extend(["--template", template])

    # Add bibliography if exists
    bib_path = base_path / "references.bib"
    if bib_path.exists() and bib_path.stat().st_size > 0:
        pandoc_cmd.extend(["--bibliography", "references.bib", "--citeproc"])

    try:
        result = subprocess.run(
            pandoc_cmd, cwd=base_path,
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            (base_path / "pandoc_error.log").write_text(
                f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
                encoding="utf-8",
            )
            print(f"Pandoc markdown->tex failed (code {result.returncode})")
            return
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Pandoc failed: {e}")
        return

    # ── Step 2: LuaLaTeX x 3 passes ──
    for i in range(3):
        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "report.tex"],
                cwd=base_path, capture_output=True, timeout=300,
            )
        except subprocess.TimeoutExpired:
            print(f"lualatex pass {i+1} timed out")
            break

    # ── Step 3: Biber for bibliography (if .bib exists and biber is available) ──
    if bib_path.exists() and bib_path.stat().st_size > 0:
        import shutil
        if shutil.which("biber"):
            try:
                subprocess.run(
                    ["biber", "report"],
                    cwd=base_path, capture_output=True, timeout=120,
                )
                # Re-run lualatex twice to resolve references
                for i in range(2):
                    subprocess.run(
                        ["pdflatex", "-interaction=nonstopmode", "report.tex"],
                        cwd=base_path, capture_output=True, timeout=300,
                    )
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"Biber/final lualatex failed: {e}")
        else:
            print("Biber not found -- bibliography will show as [?]. Install texlive-biber.")

    pdf_path = base_path / "report.pdf"
    if pdf_path.exists():
        size_kb = pdf_path.stat().st_size / 1024
        print(f"PDF generated: {pdf_path} ({size_kb:.0f} KB)")
    else:
        print("PDF generation failed -- check report.log")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        generate_latex(sys.argv[1])
