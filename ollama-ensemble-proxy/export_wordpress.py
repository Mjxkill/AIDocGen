"""
AiDocGen → WordPress Publisher
Crée un dossier multi-pages sur WordPress :
  - 1 page sommaire (parent) avec table des matières
  - 1 page par chapitre (enfant) avec navigation prev/next
  - Parse directement report.md (le même contenu que le PDF)
  - Styles inline pour compatibilité Gutenberg/Elementor
"""

import json
import re
import requests
import sys
from html import escape
from pathlib import Path
from markdown_it import MarkdownIt

# ── Config ──
WP_URL = "https://worddev.electrosens.fr/wp-json/wp/v2"
WP_USER = "michael"
WP_APP_PASSWORD = "dRUrAREJaSPlo63WUl9D5w1z"

md_renderer = MarkdownIt().enable("table")


# ── Inline styles ──
S = {
    "wrapper": "max-width:900px;margin:0 auto;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;line-height:1.8;color:#2d3748;padding:20px;",
    "header": "background:linear-gradient(135deg,#1a365d 0%,#2b6cb0 100%);color:#fff;padding:40px 40px 30px;border-radius:16px;margin-bottom:2em;",
    "header_h1": "color:#fff;font-size:1.8em;margin:0 0 8px 0;font-weight:700;line-height:1.3;",
    "header_meta": "opacity:0.85;font-size:0.95em;",
    "toc_section": "margin:1em 0;",
    "toc_chapter": "display:flex;align-items:flex-start;gap:12px;padding:10px 14px;margin:4px 0;background:#fff;border:1px solid #e2e8f0;border-radius:8px;text-decoration:none;color:inherit;",
    "toc_num": "display:flex;align-items:center;justify-content:center;min-width:28px;height:28px;background:#3182ce;color:#fff;border-radius:50%;font-weight:700;font-size:0.8em;flex-shrink:0;margin-top:2px;",
    "toc_text": "flex:1;",
    "toc_title": "font-size:1.05em;font-weight:600;color:#2b6cb0;margin:0 0 4px 0;line-height:1.4;",
    "toc_desc": "font-size:0.88em;color:#718096;margin:0;line-height:1.5;",
    "toc_subs": "font-size:0.82em;color:#a0aec0;margin:4px 0 0;line-height:1.4;",
    "breadcrumb": "margin-bottom:1.5em;",
    "breadcrumb_a": "color:#3182ce;text-decoration:none;font-size:0.95em;",
    "ch_title": "font-size:1.6em;color:#1a202c;border-bottom:3px solid #3182ce;padding-bottom:12px;margin:0 0 1.5em;",
    "nav": "display:flex;justify-content:space-between;align-items:center;margin:3em 0 1em;padding:1.5em 0;border-top:2px solid #e2e8f0;gap:1em;",
    "nav_btn": "display:inline-flex;align-items:center;gap:8px;padding:10px 20px;background:#ebf4ff;color:#2b6cb0;border-radius:8px;text-decoration:none;font-weight:500;font-size:0.95em;",
    "nav_sommaire": "display:inline-flex;align-items:center;gap:8px;padding:10px 20px;background:#f7fafc;color:#4a5568;border-radius:8px;text-decoration:none;font-weight:500;font-size:0.95em;",
    "footer": "text-align:center;padding:2em 0 1em;color:#a0aec0;font-size:0.8em;border-top:1px solid #e2e8f0;margin-top:3em;",
}


KATEX_HEAD = (
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">'
    '<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>'
    '<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" '
    'onload="renderMathInElement(document.body,{delimiters:['
    '{left:\'$$\',right:\'$$\',display:true},'
    '{left:\'$\',right:\'$\',display:false}'
    ']});"></script>'
)


def _latex_to_katex(text: str) -> str:
    r"""Convert LaTeX \[...\] and \(...\) delimiters to $$...$$ and $...$ for KaTeX."""
    # Display math: \[ ... \]  →  $$ ... $$
    text = re.sub(r"\\\[\s*\n?(.*?)\n?\s*\\\]", r"$$\1$$", text, flags=re.DOTALL)
    # Inline math: \( ... \)  →  $ ... $
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text)
    return text


def _md_to_html(text: str) -> str:
    """Convert markdown to HTML with inline styles for tables and code."""
    text = _latex_to_katex(text)
    html = md_renderer.render(text)
    # Tables
    html = html.replace("<table>", '<table style="border-collapse:collapse;width:100%;margin:1.5em 0;font-size:0.95em;">')
    html = html.replace("<th>", '<th style="background:#ebf4ff;color:#2b6cb0;text-align:left;padding:10px 14px;border:1px solid #bee3f8;">')
    html = html.replace("<td>", '<td style="padding:10px 14px;border:1px solid #e2e8f0;">')
    # Code blocks
    html = html.replace("<pre>", '<pre style="background:#1a202c;color:#e2e8f0;padding:1.2em;border-radius:8px;overflow-x:auto;font-size:0.9em;line-height:1.5;">')
    html = html.replace("<code>", '<code style="background:#edf2f7;padding:2px 6px;border-radius:4px;font-size:0.9em;color:#e53e3e;">')
    html = re.sub(
        r'(<pre[^>]*>)\s*<code[^>]*>',
        r'\1<code style="background:none;color:inherit;padding:0;">',
        html,
    )
    # Blockquotes
    html = html.replace(
        "<blockquote>",
        '<blockquote style="border-left:4px solid #3182ce;background:#ebf8ff;margin:1.5em 0;padding:1em 1.5em;border-radius:0 8px 8px 0;font-style:italic;color:#2c5282;">',
    )
    # Headings
    html = re.sub(r"<h1>", '<h1 style="font-size:1.5em;color:#1a202c;margin-top:2.5em;border-bottom:2px solid #3182ce;padding-bottom:8px;">', html)
    html = re.sub(r"<h2>", '<h2 style="font-size:1.3em;color:#2b6cb0;margin-top:2em;border-bottom:1px solid #e2e8f0;padding-bottom:6px;">', html)
    html = re.sub(r"<h3>", '<h3 style="font-size:1.15em;color:#2c5282;margin-top:1.5em;">', html)
    html = re.sub(r"<h4>", '<h4 style="font-size:1.05em;color:#4a5568;margin-top:1.2em;">', html)
    return html


def _parse_report_md(run_dir: Path) -> dict:
    """
    Parse report.md to extract title, TOC, and chapters.
    Returns: {"title": str, "chapters": [{"title", "sub_titles", "content", "summary"}]}

    Structure of report.md:
      # Main Title
      ## Table des Matières
      - [Chapter 1](#...)
        - [Sub 1.1](#...)
      ...
      ## Chapter 1 Title        <-- chapter boundary (## with <a name> anchor)
      <a name='...'></a>
      ### Sub-section title
      <a name='...'></a>
      # **Writer section**      <-- actual content (h1 from writer)
      ## subsection content
      ...
      ## Chapter 2 Title        <-- next chapter boundary
    """
    report_file = run_dir / "report.md"
    if not report_file.exists():
        return {"title": "", "chapters": []}

    text = report_file.read_text(encoding="utf-8")
    lines = text.split("\n")

    # ── 1. Extract main title (first # line) ──
    main_title = "Dossier"
    for line in lines[:5]:
        if line.startswith("# ") and not line.startswith("## "):
            main_title = line.lstrip("# ").strip()
            break

    # ── 2. Parse TOC to get chapter names and sub-sections ──
    toc_chapters = []
    toc_start = toc_end = 0
    in_toc = False
    current_toc = None

    for i, line in enumerate(lines):
        if line.strip() == "## Table des Matières":
            in_toc = True
            toc_start = i
            continue
        if in_toc:
            if line.startswith("- ["):
                # Top-level chapter
                match = re.match(r"^- \[(.+?)\]\(", line)
                if match:
                    current_toc = {"title": match.group(1), "subs": []}
                    toc_chapters.append(current_toc)
            elif line.startswith("  - [") and current_toc:
                match = re.match(r"^\s+- \[(.+?)\]\(", line)
                if match:
                    current_toc["subs"].append(match.group(1))
            elif line.strip() and not line.strip().startswith("-") and not line.strip().startswith(" "):
                toc_end = i
                break

    # ── 3. Find chapter boundaries in body ──
    # Chapters are marked by ## with <a name> anchor on next line
    chapter_boundaries = []
    for i in range(toc_end, len(lines)):
        line = lines[i]
        if line.startswith("## ") and not line.startswith("### "):
            # Check if next line has an anchor tag
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if "<a name=" in next_line:
                title = line.lstrip("## ").strip()
                chapter_boundaries.append({"title": title, "line": i})

    # ── 4. Extract chapter content between boundaries ──
    chapters = []
    for idx, boundary in enumerate(chapter_boundaries):
        start = boundary["line"]
        end = chapter_boundaries[idx + 1]["line"] if idx + 1 < len(chapter_boundaries) else len(lines)

        # Get content lines (skip the ## title and <a name> anchor)
        content_lines = lines[start + 2 : end]

        # Clean: remove <a name> anchor tags, they're just noise in HTML
        cleaned = []
        for cl in content_lines:
            if cl.strip().startswith("<a name="):
                continue
            cleaned.append(cl)

        content_md = "\n".join(cleaned).strip()

        # Find matching TOC entry for sub_titles
        sub_titles = []
        for toc_ch in toc_chapters:
            if toc_ch["title"] == boundary["title"]:
                sub_titles = toc_ch["subs"]
                break

        # Extract summary from first paragraph
        summary = ""
        for cl in cleaned:
            s = cl.strip()
            if s and not s.startswith("#") and not s.startswith("-") and not s.startswith("|") and not s.startswith("*") and not s.startswith("```"):
                summary = re.sub(r"[*_`]", "", s)
                if len(summary) > 180:
                    summary = summary[:180] + "..."
                break

        chapters.append({
            "title": boundary["title"],
            "sub_titles": sub_titles,
            "content": content_md,
            "summary": summary,
        })

    return {"title": main_title, "chapters": chapters}


def _wp_compress(html: str) -> str:
    """Remove newlines between tags to prevent wpautop from adding empty <p> tags."""
    html = re.sub(r">\s+<", "><", html)
    return html.strip()


def _sommaire_html(title: str, nb_chapters: int, chapter_pages: list) -> str:
    """Build the sommaire page HTML. Uses <span> everywhere to avoid wpautop corruption."""
    parts = [
        f'<div style="{S["wrapper"]}">',
        f'<div style="{S["header"]}">',
        f'<h1 style="{S["header_h1"]}">{escape(title)}</h1>',
        f'<span style="{S["header_meta"]}display:block;">{nb_chapters} chapitres &bull; G&eacute;n&eacute;r&eacute; par AiDocGen</span>',
        '</div>',
        f'<div style="{S["toc_section"]}">',
    ]

    for i, cp in enumerate(chapter_pages):
        sub_text = ""
        if cp.get("sub_titles"):
            subs = [escape(s) for s in cp["sub_titles"][:6]]
            sub_text = f'<span style="{S["toc_subs"]}display:block;">{" &bull; ".join(subs)}</span>'

        summary = escape(cp.get("summary", "")[:160])
        desc_part = f'<span style="{S["toc_desc"]}display:block;">{summary}</span>' if summary else ""

        parts.append(
            f'<a href="{cp["link"]}" style="{S["toc_chapter"]}">'
            f'<span style="{S["toc_num"]}">{i + 1}</span>'
            f'<span style="{S["toc_text"]}display:block;">'
            f'<span style="{S["toc_title"]}display:block;">{escape(cp["title"])}</span>'
            f'{desc_part}'
            f'{sub_text}'
            f'</span></a>'
        )

    parts.append('</div>')
    parts.append(f'<span style="{S["footer"]}display:block;">Dossier g&eacute;n&eacute;r&eacute; par AiDocGen &mdash; IA Research Engine</span>')
    parts.append('</div>')
    return _wp_compress("".join(parts))


def _chapter_html(index: int, total: int, title: str, body_html: str,
                  sommaire_url: str, prev_url: str | None, next_url: str | None,
                  prev_title: str | None, next_title: str | None) -> str:
    """Build a chapter page HTML with inline styles."""
    parts = [
        KATEX_HEAD,
        f'<div style="{S["wrapper"]}">',
        f'<div style="{S["breadcrumb"]}">',
        f'<a href="{sommaire_url}" style="{S["breadcrumb_a"]}">&larr; Retour au sommaire</a>',
        '</div>',
        f'<h1 style="{S["ch_title"]}">Chapitre {index} / {total} &mdash; {escape(title)}</h1>',
        body_html,
        f'<div style="{S["nav"]}">',
    ]

    if prev_url:
        parts.append(f'<a href="{prev_url}" style="{S["nav_btn"]}">&larr; {escape(prev_title or "Precedent")[:40]}</a>')
    else:
        parts.append("<span></span>")

    parts.append(f'<a href="{sommaire_url}" style="{S["nav_sommaire"]}">Sommaire</a>')

    if next_url:
        parts.append(f'<a href="{next_url}" style="{S["nav_btn"]}">{escape(next_title or "Suivant")[:40]} &rarr;</a>')
    else:
        parts.append("<span></span>")

    parts.append('</div>')
    parts.append(f'<span style="{S["footer"]}display:block;">Dossier g&eacute;n&eacute;r&eacute; par AiDocGen &mdash; IA Research Engine</span>')
    parts.append('</div>')
    return _wp_compress("".join(parts))


def _create_page(title: str, content: str, status: str = "draft", parent: int = 0, menu_order: int = 0) -> dict:
    """Create a WordPress page via REST API."""
    resp = requests.post(
        f"{WP_URL}/pages",
        auth=(WP_USER, WP_APP_PASSWORD),
        json={
            "title": title,
            "content": content,
            "status": status,
            "parent": parent,
            "menu_order": menu_order,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def _update_page(page_id: int, content: str) -> None:
    """Update a WordPress page content."""
    requests.post(
        f"{WP_URL}/pages/{page_id}",
        auth=(WP_USER, WP_APP_PASSWORD),
        json={"content": content},
        timeout=60,
    )


def publish_dossier(run_dir: str, status: str = "draft") -> dict:
    """
    Publish a dossier to WordPress as hierarchical pages.
    Reads directly from report.md (same content as PDF).
    Returns dict with sommaire URL and chapter URLs.
    """
    run_path = Path(run_dir)

    # Parse report.md
    parsed = _parse_report_md(run_path)
    if not parsed["chapters"]:
        raise ValueError("No chapters found in report.md")

    dossier_title = parsed["title"]
    if len(dossier_title) > 100:
        dossier_title = dossier_title[:100] + "..."

    chapters = parsed["chapters"]

    print(f"Publishing: {dossier_title}")
    print(f"  {len(chapters)} chapters from report.md")

    # ── 1. Create sommaire placeholder ──
    placeholder = _wp_compress(
        f'<div style="{S["wrapper"]}">'
        f'<div style="{S["header"]}">'
        f'<h1 style="{S["header_h1"]}">{escape(dossier_title)}</h1>'
        f'<span style="{S["header_meta"]}display:block;">Publication en cours...</span>'
        f'</div></div>'
    )

    sommaire_page = _create_page(title=dossier_title, content=placeholder, status=status)
    sommaire_id = sommaire_page["id"]
    sommaire_url = sommaire_page["link"]
    print(f"  Sommaire: {sommaire_url} (ID: {sommaire_id})")

    # ── 2. Create chapter pages ──
    chapter_pages = []
    for i, ch in enumerate(chapters):
        body_html = _md_to_html(ch["content"])

        tmp_content = _chapter_html(
            index=i + 1, total=len(chapters),
            title=ch["title"], body_html=body_html,
            sommaire_url=sommaire_url,
            prev_url=None, next_url=None,
            prev_title=None, next_title=None,
        )

        page = _create_page(
            title=f"Ch. {i + 1} – {ch['title'][:70]}",
            content=tmp_content,
            status=status,
            parent=sommaire_id,
            menu_order=i + 1,
        )
        chapter_pages.append({
            "id": page["id"],
            "link": page["link"],
            "title": ch["title"],
            "summary": ch.get("summary", ""),
            "sub_titles": ch.get("sub_titles", []),
        })
        print(f"  [{i + 1}/{len(chapters)}] {ch['title'][:60]}")

    # ── 3. Update chapters with prev/next navigation ──
    for i, cp in enumerate(chapter_pages):
        ch = chapters[i]
        body_html = _md_to_html(ch["content"])

        prev_url = chapter_pages[i - 1]["link"] if i > 0 else None
        next_url = chapter_pages[i + 1]["link"] if i < len(chapter_pages) - 1 else None
        prev_title = chapter_pages[i - 1]["title"] if i > 0 else None
        next_title = chapter_pages[i + 1]["title"] if i < len(chapter_pages) - 1 else None

        final_content = _chapter_html(
            index=i + 1, total=len(chapters),
            title=ch["title"], body_html=body_html,
            sommaire_url=sommaire_url,
            prev_url=prev_url, next_url=next_url,
            prev_title=prev_title, next_title=next_title,
        )
        _update_page(cp["id"], final_content)

    # ── 4. Update sommaire with chapter links ──
    final_sommaire = _sommaire_html(dossier_title, len(chapters), chapter_pages)
    _update_page(sommaire_id, final_sommaire)
    print(f"\n  Sommaire updated")

    result = {
        "sommaire_url": sommaire_url,
        "sommaire_id": sommaire_id,
        "chapters": [{"title": cp["title"], "url": cp["link"]} for cp in chapter_pages],
    }
    print(f"\nDossier published: {sommaire_url}")
    return result


def delete_dossier(sommaire_id: int) -> int:
    """Delete a dossier and all its chapter pages. Returns count deleted."""
    deleted = 0
    page_num = 1
    while True:
        resp = requests.get(
            f"{WP_URL}/pages",
            auth=(WP_USER, WP_APP_PASSWORD),
            params={"parent": sommaire_id, "per_page": 100, "page": page_num, "status": "any"},
            timeout=30,
        )
        pages = resp.json()
        if not pages or not isinstance(pages, list):
            break
        for p in pages:
            requests.delete(
                f"{WP_URL}/pages/{p['id']}",
                auth=(WP_USER, WP_APP_PASSWORD),
                params={"force": True},
                timeout=15,
            )
            deleted += 1
        if len(pages) < 100:
            break
        page_num += 1

    requests.delete(
        f"{WP_URL}/pages/{sommaire_id}",
        auth=(WP_USER, WP_APP_PASSWORD),
        params={"force": True},
        timeout=15,
    )
    deleted += 1
    return deleted


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python export_wordpress.py <run_dir> [draft|publish]")
        print("  python export_wordpress.py --delete <sommaire_id>")
        sys.exit(1)

    if sys.argv[1] == "--delete":
        sid = int(sys.argv[2])
        n = delete_dossier(sid)
        print(f"Deleted {n} pages")
    else:
        run_dir = sys.argv[1]
        pub_status = sys.argv[2] if len(sys.argv) > 2 else "draft"
        result = publish_dossier(run_dir, status=pub_status)
        print(json.dumps(result, indent=2, ensure_ascii=False))
