#!/usr/bin/env python3
"""Markdown → audiobook via Kokoro-FastAPI (or OpenAI TTS).

Output: a directory containing
  - chapters/01_<slug>.mp3 ... per-chapter mp3
  - <basename>.m4b           combined audiobook with chapter markers
  - <basename>.zip           per-chapter mp3 + m4b bundled

Resume: per-chunk wav files are persisted in <state-dir>; re-running with the
same state dir skips chunks already on disk. Progress is written to
<state-dir>/../progress.json after every chunk.
"""
import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Markdown cleanup
# ─────────────────────────────────────────────────────────────────────────────

# Symbols that XTTS reads literally or oddly. Replace with French words.
# Order matters (longer multi-char patterns first).
_AUDIO_SYMBOL_REPLACEMENTS = [
    ("⟶", " vers "), ("→", " vers "), ("⇒", " donne "),
    ("⟵", " depuis "), ("←", " depuis "),
    ("↔", " entre "), ("⇔", " équivaut à "),
    ("≈", " environ "), ("≃", " environ "), ("≅", " environ "),
    ("≠", " différent de "),
    ("≤", " inférieur ou égal à "), ("≥", " supérieur ou égal à "),
    ("±", " plus ou moins "),
    ("∞", " infini "),
    ("×", " fois "), ("÷", " divisé par "),
    ("°C", " degrés Celsius "), ("°F", " degrés Fahrenheit "), ("°", " degrés "),
    ("™", ""), ("®", ""), ("©", ""),
    ("…", "."),
    ("«", " "), ("»", " "),
    ("“", " "), ("”", " "), ("„", " "),
    ("→", " vers "),
]


def clean_markdown(md: str) -> str:
    md = re.sub(r"```[\s\S]*?```", "code source.", md)  # code block → audible label
    md = re.sub(r"`[^`\n]+`", "", md)
    md = re.sub(r"!\[[^\]]*\]\([^)]+\)\s*", "", md)
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)
    md = re.sub(r"https?://\S+", "", md)
    md = re.sub(r"^\s*[-=*_]{3,}\s*$", "", md, flags=re.M)
    md = re.sub(r"\[CLM-[A-Za-z0-9_\-]+\]", "", md)
    md = re.sub(r"(\*\*|__)(.+?)\1", r"\2", md)
    md = re.sub(r"(\*|_)(.+?)\1", r"\2", md)
    md = re.sub(r"^\s*[-*+]\s+", "", md, flags=re.M)
    md = re.sub(r"^\s*\d+\.\s+", "", md, flags=re.M)
    # Quote/blockquote prefixes
    md = re.sub(r"^\s*>\s+", "", md, flags=re.M)
    # Audio-friendly symbol substitutions
    for src, dst in _AUDIO_SYMBOL_REPLACEMENTS:
        md = md.replace(src, dst)
    # < > as math operators surrounded by spaces (post hashtag-strip so we
    # don't break HTML — HTML is normally not present in adapted prose).
    md = re.sub(r"\s<\s", " inférieur à ", md)
    md = re.sub(r"\s>\s", " supérieur à ", md)
    # @ used as "at"
    md = re.sub(r"\s@\s*", " à ", md)
    # Collapse whitespace
    md = re.sub(r"[ \t]+", " ", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


_TOC_TITLE_RE = re.compile(
    r"^(table\s+des\s+mati[èe]res|sommaire|table\s+of\s+contents?|contents?|toc|index)\s*$",
    re.IGNORECASE,
)


def is_toc_title(title: str) -> bool:
    return bool(_TOC_TITLE_RE.match((title or "").strip()))


def split_into_chapters(md: str) -> list[tuple[str, str]]:
    """Split a cleaned-but-headed Markdown into [(title, body)] using ## as chapter boundary.

    A leading section before the first ## becomes "Introduction" (or is dropped if empty)."""
    lines = md.splitlines()
    chapters: list[tuple[str, list[str]]] = []
    current_title = "Introduction"
    current_body: list[str] = []
    for ln in lines:
        m = re.match(r"^(#{1,3})\s+(.+)$", ln)
        if m and len(m.group(1)) == 2:
            # commit previous
            if current_body:
                chapters.append((current_title, current_body))
            current_title = m.group(2).strip().rstrip(":") or "Chapitre"
            current_body = []
        else:
            current_body.append(ln)
    if current_body:
        chapters.append((current_title, current_body))

    out = []
    for title, body_lines in chapters:
        if is_toc_title(title):
            print(f"[skip TOC chapter] {title!r}", flush=True)
            continue
        body = "\n".join(body_lines)
        # Flatten remaining # / ### into sentences with terminal period
        body = re.sub(r"^(#{1,6})\s+(.+)$",
                      lambda m: m.group(2).strip().rstrip(":") + ".",
                      body, flags=re.M)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if body or out:  # keep empty-only-title chapters? skip
            out.append((title, body))
    return out


def chunk_text(text: str, target: int = 3000) -> list[str]:
    sentences = re.split(r"(?<=[.!?…])\s+(?=[A-ZА-ЯÀ-ÖØ-Þ«„])|\n\n+", text)
    chunks, cur, cur_size = [], [], 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        size = len(s) + 1
        if cur and cur_size + size > target:
            chunks.append(" ".join(cur))
            cur, cur_size = [s], size
        else:
            cur.append(s)
            cur_size += size
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# TTS engines
# ─────────────────────────────────────────────────────────────────────────────

XTTS_MAX_CHARS = 600  # ~150 tokens for typical prose; technical/numeric
                      # content tokenizes more aggressively, so we retry with
                      # a smaller cap on HTTP 500 (see _xtts_with_retry).


def _split_for_xtts(text: str, max_chars: int = XTTS_MAX_CHARS) -> list[str]:
    """Sub-split text so each piece fits XTTS's 400-token cap.
    Sentences first; if a sentence is still too long, fall back to
    clause boundaries (`, ; :`) then to word boundaries."""
    pieces = chunk_text(text, target=max_chars)
    out: list[str] = []
    for p in pieces:
        if len(p) <= max_chars:
            out.append(p)
            continue
        sub = re.split(r"(?<=[,;:])\s+", p)
        cur, cur_len = [], 0
        for s in sub:
            if not s:
                continue
            if cur and cur_len + len(s) + 1 > max_chars:
                out.append(" ".join(cur))
                cur, cur_len = [s], len(s)
            else:
                cur.append(s)
                cur_len += len(s) + 1
        if cur:
            joined = " ".join(cur)
            if len(joined) <= max_chars:
                out.append(joined)
            else:
                line = ""
                for w in joined.split():
                    if line and len(line) + len(w) + 1 > max_chars:
                        out.append(line)
                        line = w
                    else:
                        line = (line + " " + w).strip()
                if line:
                    out.append(line)
    # Last-resort hard slice — handles pathological no-whitespace input
    final: list[str] = []
    for p in out:
        if len(p) <= max_chars:
            final.append(p)
        else:
            for k in range(0, len(p), max_chars):
                final.append(p[k:k + max_chars])
    return final


def _xtts_request(text: str, voice: str, speed: float, host: str,
                  language: str) -> bytes:
    payload = {"voice": voice, "input": text, "language": language,
               "response_format": "mp3", "speed": speed}
    req = urllib.request.Request(
        f"{host.rstrip('/')}/v1/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return resp.read()


def _xtts_with_retry(text: str, voice: str, speed: float, host: str,
                     language: str, depth: int = 0) -> bytes:
    """If XTTS rejects (HTTP 500 — typically the 400-token assertion on
    technical/dense text), halve the cap and recurse. Caps recursion at
    depth 6 (text shrunk by ~64x) to avoid runaway."""
    try:
        return _xtts_request(text, voice, speed, host, language)
    except urllib.error.HTTPError as e:
        if e.code != 500 or depth >= 6 or len(text) <= 60:
            raise
        smaller = max(60, len(text) // 2)
        pieces = _split_for_xtts(text, max_chars=smaller)
        if len(pieces) <= 1:
            raise
        print(f"[xtts] {len(text)} chars rejected (depth {depth}); "
              f"re-splitting into {len(pieces)} sub-pieces (cap={smaller})",
              flush=True)
        out = bytearray()
        for p in pieces:
            if not p.strip():
                continue
            out.extend(_xtts_with_retry(p, voice, speed, host, language,
                                        depth=depth + 1))
        return bytes(out)


def tts_xtts(text: str, voice: str, speed: float, host: str,
             language: str = "fr") -> bytes:
    """Local XTTS-v2 server (Blackwell-compatible).

    XTTS has a hard 400-token-per-inference limit. Long inputs are
    transparently split at sentence (then clause, then word) boundaries.
    A piece that still trips the assertion (token-dense content like
    acronyms/numbers) triggers a recursive halving retry. Resulting
    libmp3lame fixed-bitrate MP3 fragments are byte-concatenated — valid
    since each MP3 frame is self-contained.
    """
    pieces = _split_for_xtts(text)
    if len(pieces) <= 1:
        return _xtts_with_retry(text, voice, speed, host, language)
    out = bytearray()
    for piece in pieces:
        if not piece.strip():
            continue
        out.extend(_xtts_with_retry(piece, voice, speed, host, language))
    return bytes(out)


def tts_openai(text: str, voice: str, speed: float, api_key: str,
               model: str = "gpt-4o-mini-tts") -> bytes:
    """OpenAI TTS — returns MP3 bytes."""
    payload = {"model": model, "voice": voice, "input": text,
               "response_format": "mp3", "speed": speed}
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return resp.read()


def call_tts(text: str, voice: str, speed: float, engine: str,
             host: str, openai_key: str, language: str = "fr") -> bytes:
    if engine == "openai":
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        return tts_openai(text, voice, speed, openai_key)
    if engine == "xtts":
        return tts_xtts(text, voice, speed, host, language)
    raise RuntimeError(f"unknown engine {engine!r}")


# ─────────────────────────────────────────────────────────────────────────────
# ffmpeg / ffprobe helpers
# ─────────────────────────────────────────────────────────────────────────────

def ffprobe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(r.stdout.strip() or 0)


def concat_mp3(parts: list[Path], out: Path):
    if not parts:
        raise RuntimeError("nothing to concat")
    list_file = out.parent / f"_cat_{out.stem}.txt"
    list_file.write_text("\n".join(f"file '{p}'" for p in parts), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c", "copy", str(out)],
        check=True, capture_output=True, text=True,
    )
    list_file.unlink(missing_ok=True)


def make_m4b(chapter_files: list[tuple[str, Path, float]], out_m4b: Path, title: str):
    """Combine per-chapter mp3s into a single M4B (AAC) with chapter markers.

    chapter_files = [(title, mp3_path, duration_seconds)]"""
    work = out_m4b.parent / "_m4b_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    # 1) Concat all mp3s into a single audio stream (re-encoded to AAC for M4B)
    list_file = work / "list.txt"
    list_file.write_text("\n".join(f"file '{p}'" for _, p, _ in chapter_files), encoding="utf-8")
    combined = work / "combined.m4a"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c:a", "aac", "-b:a", "96k", str(combined)],
        check=True, capture_output=True, text=True,
    )
    # 2) Build ffmetadata file with chapter markers
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
    # 3) Mux audio + chapter metadata into M4B (mp4 container, .m4b extension)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(combined), "-i", str(meta_path),
         "-map", "0:a", "-map_metadata", "1", "-c", "copy", str(out_m4b)],
        check=True, capture_output=True, text=True,
    )
    shutil.rmtree(work, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Optional: LLM summarization before TTS (Ollama Cloud)
# ─────────────────────────────────────────────────────────────────────────────

SUMMARY_PROMPT = (
    "You are adapting a written technical document for AUDIO NARRATION (audiobook).\n"
    "The goal is NOT to shorten — keep ALL the technical detail, every concept, "
    "every named entity. Just rewrite so it sounds natural when listened to.\n\n"
    "STRICT FORMATTING RULES (these override anything else):\n"
    "1. Reproduce every `# ` / `## ` / `### ` / `#### ` header VERBATIM, including "
    "any leading numbering like `## 2.`, `### 2.1`, `### 4.1.3` — keep the numbers.\n"
    "2. Use the same language as the source. If the source is French, output is French.\n"
    "3. Output the SAME LENGTH as the input — this is a transformation, not a summary.\n\n"
    "Content rules:\n"
    "- Preserve names, numbers, units, technical terms, acronyms verbatim. Numbers "
    "may be spelled out for clarity (e.g. `-14 LUFS` → \"moins quatorze LUFS\") but "
    "the underlying value must stay correct.\n"
    "- Tables → narrate them in prose: \"The first column lists X; for Y the value "
    "is Z, for W it is V…\".\n"
    "- Code blocks → describe in 1-2 sentences what the code does and the key "
    "parameters; do not dictate the syntax character by character.\n"
    "- Figures, images, diagrams → describe what they illustrate.\n"
    "- Bulleted lists → spoken enumerations: \"first…, second…, third…\".\n"
    "- Citation tags like [CLM-xxx], [claim_xxx], [r128-xxx] → drop silently.\n"
    "- URLs → drop or replace with \"the official documentation\".\n"
    "- Cross-references like \"see Figure 3\" → integrate naturally.\n\n"
    "Output format:\n"
    "- Output ONLY the adapted Markdown content.\n"
    "- No preamble (do NOT start with \"Voici…\", \"Here is…\", \"Sure…\").\n"
    "- No closing remarks.\n"
    "- No explanation of what you did.\n"
)


# ── Adaptation chunking ──────────────────────────────────────────────────────
# One chunk = one section (or subsection if available). Never grouped.
# Calls run sequentially by design.


_LINK_LINE = re.compile(r'^\s*[-*]\s*\[[^\]]+\]\([^)]+\).*$')

def _has_prose(text: str, min_chars: int = 200) -> bool:
    """Does this chunk contain real prose worth sending to the LLM?
    Headers / anchors / horizontal rules / link-lists do NOT count."""
    prose_chars = 0
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue
        if s.startswith("<"):
            continue
        if s.startswith(("```", "---", "===", "***")):
            continue
        if _LINK_LINE.match(line):
            continue
        prose_chars += len(s)
    return prose_chars >= min_chars


def _is_toc_or_links(block: str, min_links: int = 5) -> bool:
    body_lines = [l for l in block.splitlines()
                  if l.strip() and not l.strip().startswith("#")]
    if not body_lines:
        return True
    link_lines = sum(1 for l in body_lines if _LINK_LINE.match(l))
    return link_lines >= min_links and link_lines / len(body_lines) >= 0.6


def _greedy_pack(items: list[str], max_size: int) -> list[str]:
    out, buf = [], ""
    for it in items:
        if not buf:
            buf = it
        elif len(buf) + len(it) <= max_size:
            buf += it
        else:
            out.append(buf); buf = it
    if buf:
        out.append(buf)
    return out


def _split_atom_recursively(atom: str, max_size: int) -> list[str]:
    """Recursively split an oversized atom on inner headings, then paragraphs."""
    if len(atom) <= max_size:
        return [atom]
    head_match = re.match(r"^(#{1,6})\s", atom.lstrip())
    own_level = len(head_match.group(1)) if head_match else 0
    for level in range(own_level + 1, 7):
        marker = f"\n{'#' * level} "
        if marker not in atom:
            continue
        parts, cur = [], []
        for line in atom.splitlines(keepends=True):
            if line.startswith(f"{'#' * level} ") and cur:
                parts.append("".join(cur)); cur = [line]
            else:
                cur.append(line)
        if cur:
            parts.append("".join(cur))
        out = _greedy_pack(parts, max_size)
        result = []
        for p in out:
            result.extend(_split_atom_recursively(p, max_size) if len(p) > max_size else [p])
        if all(len(p) <= max_size for p in result):
            return result
    paragraphs = [p + "\n\n" for p in atom.split("\n\n") if p.strip()]
    return _greedy_pack(paragraphs, max_size)


def _split_md_into_smallest_sections(
    md: str, min_size: int = 1500, max_size: int = 5000,
) -> list[tuple[str, bool]]:
    """Returns a list of (chunk, is_passthrough) pairs.

    is_passthrough=True means the chunk has no real prose (TOC, anchors only,
    horizontal rules…) and must NOT be sent to the LLM — it is copied verbatim
    into the output.

    Algorithm:
      1. Atomize: walk the markdown, each heading + body until next heading is one atom.
      2. Sub-split atoms larger than max_size on their inner headings, falling
         back to paragraph boundaries.
      3. Tag each atom passthrough (TOC-like or no prose) or prose.
      4. Pack: consecutive passthrough atoms merge freely; consecutive prose
         atoms merge up to max_size.
      5. Merge undersized prose chunks (< min_size) with the previous prose
         chunk when the combined size fits within max_size.
    """
    atoms, cur = [], []
    for line in md.splitlines(keepends=True):
        if re.match(r"^#{1,6}\s+", line) and cur:
            atoms.append("".join(cur)); cur = [line]
        else:
            cur.append(line)
    if cur:
        atoms.append("".join(cur))

    refined: list[str] = []
    for a in atoms:
        refined.extend(_split_atom_recursively(a, max_size))

    def _tag(a: str) -> str:
        if _is_toc_or_links(a) or not _has_prose(a):
            return "passthrough"
        return "prose"
    tagged = [(a, _tag(a)) for a in refined]

    out: list[tuple[str, str]] = []
    buf, buf_tag = "", None
    for a, tag in tagged:
        if buf_tag is None:
            buf, buf_tag = a, tag
        elif tag == buf_tag == "passthrough":
            buf += a
        elif tag == buf_tag == "prose" and len(buf) + len(a) <= max_size:
            buf += a
        else:
            out.append((buf, buf_tag))
            buf, buf_tag = a, tag
    if buf:
        out.append((buf, buf_tag))

    i = 0
    while i < len(out):
        c, tag = out[i]
        if tag == "prose" and len(c) < min_size and i > 0:
            prev_c, prev_tag = out[i - 1]
            if prev_tag == "prose" and len(prev_c) + len(c) <= max_size:
                out[i - 1] = (prev_c + c, "prose")
                out.pop(i)
                continue
        i += 1
    return [(c, t == "passthrough") for c, t in out]


def _record_cost(provider: str, model: str, kind: str, units: float, **extra) -> None:
    """Append a cost entry to COST_LOG_PATH if set. Best-effort, never raises."""
    import os as _os
    path = _os.getenv("COST_LOG_PATH")
    if not path:
        return
    try:
        # Tariffs (kept in sync with core/cost.py)
        tariffs = {
            ("deepseek", "deepseek-v4-flash", "input_token"):  0.27e-6,
            ("deepseek", "deepseek-v4-flash", "output_token"): 1.10e-6,
            ("deepseek", "deepseek-v4-pro",   "input_token"):  0.55e-6,
            ("deepseek", "deepseek-v4-pro",   "output_token"): 2.19e-6,
        }
        rate = tariffs.get((provider, model, kind), 0.0)
        cost_usd = round(rate * units, 6)
        entry = {
            "ts": int(time.time()),
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provider": provider, "kind": kind, "model": model,
            "units": units, "cost_usd": cost_usd,
            "user_id": _os.getenv("COST_USER_ID") or None,
            "user_name": _os.getenv("COST_USER_NAME") or None,
            "job_id": _os.getenv("COST_JOB_ID") or None,
        }
        if extra: entry["details"] = extra
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # cost tracking must not break the job


def _llm_call(payload_obj: dict, base_url: str, api_key: str,
              max_attempts: int = 3, schema: str = "ollama") -> str:
    """schema: 'ollama' (Ollama-native /api/chat) or 'openai' (/chat/completions)."""
    if schema == "openai":
        # Translate Ollama-shaped payload into OpenAI shape
        oai_payload = {
            "model": payload_obj["model"],
            "messages": payload_obj["messages"],
            "stream": False,
        }
        # think:false → no equivalent in OpenAI; just drop
        # options.num_predict → max_tokens
        opts = payload_obj.get("options") or {}
        if "num_predict" in opts:
            oai_payload["max_tokens"] = opts["num_predict"]
        body = json.dumps(oai_payload).encode("utf-8")
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
    else:
        body = json.dumps(payload_obj).encode("utf-8")
        endpoint = f"{base_url.rstrip('/')}/api/chat"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last_err: Exception | None = None
    model = payload_obj.get("model", "")
    for attempt in range(1, max_attempts + 1):
        try:
            req = urllib.request.Request(endpoint, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as resp:
                d = json.loads(resp.read())
            if schema == "openai":
                choices = d.get("choices") or []
                content = (choices[0].get("message") or {}).get("content", "").strip() if choices else ""
                # OpenAI-compatible APIs (DeepSeek) return usage with prompt/completion tokens
                usage = d.get("usage") or {}
                pin = usage.get("prompt_tokens") or 0
                pout = usage.get("completion_tokens") or 0
                # Only record cost for known DeepSeek-server models (cloud-tagged
                # via Ollama is bundled in the subscription, $0).
                if model.startswith(("deepseek-v3", "deepseek-v4", "deepseek-chat",
                                     "deepseek-reasoner")) and (pin or pout):
                    if pin:
                        _record_cost("deepseek", model, "input_token", pin)
                    if pout:
                        _record_cost("deepseek", model, "output_token", pout)
            else:
                content = (d.get("message") or {}).get("content", "").strip()
            if content:
                return content
            raise RuntimeError("empty response")
        except urllib.error.HTTPError as e:
            last_err = e
            try:
                detail = e.read()[:300].decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            if e.code < 500 and e.code != 429:
                raise RuntimeError(f"HTTP {e.code}: {detail}") from e
            wait = min(300, 10 * (2 ** (attempt - 1)))
            print(f"[llm] attempt {attempt}/{max_attempts} → HTTP {e.code}; "
                  f"retry in {wait}s ({detail[:120]})", flush=True)
            time.sleep(wait)
        except Exception as e:
            last_err = e
            wait = min(300, 10 * (2 ** (attempt - 1)))
            print(f"[llm] attempt {attempt}/{max_attempts} → {e}; retry in {wait}s",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(f"call failed after {max_attempts} attempts: {last_err}")


def summarize_via_ollama(md: str, model: str, base_url: str, api_key: str,
                         schema: str = "ollama") -> str:
    """Adapt a markdown for audio narration. One LLM call per section/subsection.

    Splitting: prefers `### ` then `## `. Sections are NEVER grouped together —
    each unit gets its own focused LLM call. Calls run sequentially
    (one at a time) for predictability and to avoid cloud overload.

    Per-section results are cached on disk in <state-dir>/sections/<idx>.md
    so a re-run skips work already done.
    """
    tagged = _split_md_into_smallest_sections(md)
    if len(tagged) == 1 and not tagged[0][1]:
        return _llm_call({
            "model": model, "stream": False, "think": False,
            "options": {"num_predict": 64000},
            "messages": [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": md},
            ],
        }, base_url, api_key, schema=schema)

    pt_count = sum(1 for _, p in tagged if p)
    llm_count = len(tagged) - pt_count
    print(f"[adapt] input {len(md)} chars — {len(tagged)} chunks "
          f"({llm_count} LLM, {pt_count} passthrough), sequential", flush=True)

    sys_template = SUMMARY_PROMPT + (
        "\nThis is one section of a larger document. Adapt this section on its "
        "own. Keep its `### ` or `## ` header verbatim. The section will be "
        "concatenated with the others afterwards.\n"
    )

    # Section-level cache lives next to the script's state dir if available
    # via the env var ADAPT_CACHE_DIR; falls back to a tmp dir.
    import os as _os, tempfile as _tmp
    cache_dir = _os.getenv("ADAPT_CACHE_DIR") or _tmp.mkdtemp(prefix="adapt_")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    # progress.json sits next to <state-dir>; we get the parent of cache_dir's parent
    # via the ADAPT_PROGRESS_PATH env if set, otherwise we write next to cache_dir.
    progress_path = _os.getenv("ADAPT_PROGRESS_PATH") or str(Path(cache_dir).parent / "progress.json")
    started = time.time()

    def _write_adapt_progress(done: int):
        Path(progress_path).write_text(json.dumps({
            "done": done, "total": len(tagged),
            "pct": round(done / max(1, len(tagged)) * 100, 1),
            "stage": "adapt", "started_at": started,
            "updated_at": time.time(),
        }))

    out_parts: list[str] = []
    for idx, (content, is_passthrough) in enumerate(tagged):
        if not content.strip():
            _write_adapt_progress(idx + 1)
            continue
        if is_passthrough:
            # TOC, anchor lists, horizontal rules — useless for audio narration.
            # Skip entirely (don't cache, don't emit) so the audiobook contains
            # only adapted prose.
            print(f"[adapt] {idx+1}/{len(tagged)}: skip passthrough "
                  f"({len(content)} chars)", flush=True)
            _write_adapt_progress(idx + 1)
            continue
        cache_file = Path(cache_dir) / f"sec_{idx:05d}.md"
        if cache_file.exists() and cache_file.stat().st_size > 0:
            out_parts.append(cache_file.read_text(encoding="utf-8").strip())
            print(f"[adapt] {idx+1}/{len(tagged)}: cached", flush=True)
            _write_adapt_progress(idx + 1)
            continue
        out = _llm_call({
            "model": model, "stream": False, "think": False,
            "options": {"num_predict": 64000},
            "messages": [
                {"role": "system", "content": sys_template},
                {"role": "user", "content": content},
            ],
        }, base_url, api_key, schema=schema)
        cache_file.write_text(out, encoding="utf-8")
        out_parts.append(out.strip())
        print(f"[adapt] {idx+1}/{len(tagged)}: {len(content)}→{len(out)} chars "
              f"({len(out)/max(1,len(content))*100:.0f}%)", flush=True)
        _write_adapt_progress(idx + 1)
    return "\n\n".join(p for p in out_parts if p)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def update_progress(state_dir: Path, total: int, started_at: float, voice: str, stage: str = ""):
    done = sum(1 for p in state_dir.rglob("chunk_*.mp3") if p.stat().st_size > 0)
    (state_dir.parent / "progress.json").write_text(json.dumps({
        "done": done, "total": total,
        "pct": round(done / total * 100, 1) if total else 0,
        "started_at": started_at, "updated_at": time.time(),
        "voice": voice, "stage": stage,
    }))


def slugify(s: str, max_len: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9_\- ]", "", s).strip()[:max_len].replace(" ", "_")
    return s or "chapitre"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_md")
    ap.add_argument("output_dir", help="dir where the .m4b, .zip and chapters/ live")
    ap.add_argument("--basename", default="audiobook")
    ap.add_argument("--voice", default="ff_siwis")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=3000)
    ap.add_argument("--engine", choices=["xtts", "openai"], default="xtts")
    ap.add_argument("--kokoro-host", default="http://localhost:8021",
                    help="TTS server URL (legacy flag name; points to XTTS by default)")
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--summarize", action="store_true",
                    help="Pre-summarize the markdown via Ollama Cloud before TTS.")
    ap.add_argument("--summary-model", default="deepseek-v4-pro")
    ap.add_argument("--summary-origin", choices=["local", "cloud", "deepseek", "auto"],
                    default="auto",
                    help="Where the summary model lives. 'auto' uses heuristics on the name.")
    args = ap.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg/ffprobe required")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = Path(args.state_dir) if args.state_dir else out_dir / "_state"
    state.mkdir(parents=True, exist_ok=True)

    raw = Path(args.input_md).read_text(encoding="utf-8")

    # Optional summarization pass (one big LLM call). Cached on disk so resume skips it.
    if args.summarize:
        cache = out_dir / "summary.md"
        if cache.exists() and cache.stat().st_size > 0:
            print(f"[summary] using cached {cache}", flush=True)
            raw = cache.read_text(encoding="utf-8")
        else:
            import os
            model_name = args.summary_model
            origin = args.summary_origin
            if origin == "auto":
                if model_name.endswith(":cloud") or "-cloud" in model_name:
                    origin = "cloud"
                elif model_name.startswith(("deepseek-chat", "deepseek-reasoner",
                                            "deepseek-v3", "deepseek-v4")) \
                     and os.getenv("DEEPSEEK_API_KEY"):
                    origin = "deepseek"
                else:
                    origin = "local"

            if origin == "deepseek":
                base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
                key = os.getenv("DEEPSEEK_API_KEY", "")
                if not key:
                    sys.exit("DEEPSEEK_API_KEY missing for DeepSeek model")
                schema = "openai"
            elif origin == "cloud":
                base = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
                key = os.getenv("OLLAMA_API_KEY", "")
                if not key:
                    sys.exit("OLLAMA_API_KEY missing for cloud model")
                schema = "ollama"
            else:  # local
                base = os.getenv("OLLAMA_LOCAL_URL", "http://localhost:11434")
                key = ""
                schema = "ollama"
            print(f"[summary] {model_name} via {base} (origin={origin}, "
                  f"schema={schema}, {len(raw)} chars in)…", flush=True)
            raw = summarize_via_ollama(raw, model_name, base, key, schema=schema)
            cache.write_text(raw, encoding="utf-8")
            print(f"[summary] {len(raw)} chars out → {cache}", flush=True)

    text = clean_markdown(raw)
    chapters = split_into_chapters(text)
    if not chapters:
        sys.exit("nothing to read aloud")

    # Plan total chunks for progress
    plan: list[tuple[int, str, list[str]]] = []  # (chapter_idx, title, chunks)
    total_chunks = 0
    for i, (title, body) in enumerate(chapters):
        cs = chunk_text(body, args.chunk) if body else [title]
        plan.append((i, title, cs))
        total_chunks += len(cs)
    print(f"input: {args.input_md}  chapters: {len(chapters)}  chunks: {total_chunks}  "
          f"voice: {args.voice}  engine: {args.engine}", flush=True)

    started = time.time()
    update_progress(state, total_chunks, started, args.voice, "synth")

    import os
    openai_key = os.getenv("OPENAI_API_KEY", "")
    chapters_dir = out_dir / "chapters"
    chapters_dir.mkdir(exist_ok=True)
    chapter_files: list[tuple[str, Path, float]] = []

    for i, title, cs in plan:
        ch_state = state / f"ch_{i:03d}"
        ch_state.mkdir(exist_ok=True)
        # TTS each chunk (cached)
        for j, c in enumerate(cs):
            tgt = ch_state / f"chunk_{j:05d}.mp3"
            if tgt.exists() and tgt.stat().st_size > 0:
                print(f"  ch {i+1}/{len(plan)} chunk {j+1}/{len(cs)}: cached", flush=True)
                continue
            audio = call_tts(c, args.voice, args.speed, args.engine,
                             args.kokoro_host, openai_key)
            tgt.write_bytes(audio)
            print(f"  ch {i+1}/{len(plan)} chunk {j+1}/{len(cs)}: {len(audio)} bytes", flush=True)
            update_progress(state, total_chunks, started, args.voice, "synth")
        # Concat to chapter MP3
        ch_mp3 = chapters_dir / f"{i+1:02d}_{slugify(title)}.mp3"
        parts = sorted(p for p in ch_state.glob("chunk_*.mp3") if p.stat().st_size > 0)
        if not parts:
            sys.exit(f"chapter {i+1} produced no audio chunks — aborting")
        concat_mp3(parts, ch_mp3)
        dur = ffprobe_duration(ch_mp3)
        chapter_files.append((title, ch_mp3, dur))

    # Build M4B
    update_progress(state, total_chunks, started, args.voice, "muxing")
    m4b = out_dir / f"{args.basename}.m4b"
    print(f"[m4b] muxing {len(chapter_files)} chapters into {m4b}", flush=True)
    make_m4b(chapter_files, m4b, args.basename)

    # Build ZIP (per-chapter mp3 + m4b)
    zip_path = out_dir / f"{args.basename}.zip"
    print(f"[zip] {zip_path}", flush=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        z.write(m4b, arcname=m4b.name)
        for _, p, _ in chapter_files:
            z.write(p, arcname=f"chapters/{p.name}")

    total_dur = sum(d for _, _, d in chapter_files)
    print(f"DONE → {m4b.name} + {zip_path.name} "
          f"({len(chapter_files)} chapters, {int(total_dur)}s total)", flush=True)


if __name__ == "__main__":
    main()
