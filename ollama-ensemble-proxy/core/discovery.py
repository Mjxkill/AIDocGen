"""Subject-first research pipeline for the "dossier_recherche" prompt type.

The classic pipeline plans the table of contents BEFORE looking at the matter,
which is fine on broad topics but produces 500 pages of hallucinations on niche
topics where nothing exists in the wild for half the imagined sub-sections.

This module flips it:
  A. Understand the question (LLM, no web yet) -> structured brief.
  B. Search every "brick" identified in A — hardware refs, components, methods.
  C. Search every brick-PAIR -- how they connect, integrations, dataflow.
  D. Search "this reminds me of …" -- analogies / adjacent domains.
  E. Synthesize the corpus -> what is established / uncertain / missing.
  F. Build the plan FROM THE EVIDENCE, not from imagination.

Only when the corpus shows real findings does the writer start. If the corpus
is anemic, we emit a short honest brief instead of long-form filler.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import DossierConfig
from .llm import LLMClient
from .logging_config import get_logger
from .utils import emit_progress, save_json

log = get_logger("aidocgen.discovery")


# ---------------------------------------------------------------------------
# A. Subject brief — pure LLM, no web yet.
# ---------------------------------------------------------------------------

_BRIEF_SYSTEM = (
    "Tu es un Architecte de Recherche. Avant d'écrire un dossier technique, "
    "tu analyses la demande pour la décomposer en briques concrètes que l'on "
    "pourra rechercher individuellement sur le web. Tu ne rédiges rien : tu "
    "produis un BRIEF en JSON strict, ancré dans la demande.\n\n"
    "Réfléchis avec rigueur : qu'est-ce qui est réellement demandé ? quels "
    "objets matériels et logiciels apparaissent (références, modèles, normes) ? "
    "quelles contraintes physiques (latence, temps réel, bande passante) ? "
    "quels points de friction connus ? quels sujets adjacents — connus de toi — "
    "éclairent la question même s'ils ne sont pas explicites ?"
)

_BRIEF_USER_TMPL = (
    "QUESTION DE L'UTILISATEUR :\n"
    "{question}\n\n"
    "TAGS FOURNIS PAR L'UTILISATEUR : {tags}\n\n"
    "Produis UN SEUL bloc JSON (rien d'autre, pas de ```json), avec ces clés :\n"
    "{{\n"
    '  "reformulated": "la question reformulée de façon précise",\n'
    '  "bricks": [\n'
    "    /* chaque BRIQUE = un composant nommé que l'on cherchera tel quel.\n"
    "       Préférer les références exactes (IMX8MP, TAC5212, U-Net, etc.)\n"
    "       à des concepts vagues. 5 à 20 briques. */\n"
    '    {{ "name": "...", "kind": "hardware|software|method|standard|metric", '
    '"why": "pourquoi c\'est central dans la question" }}\n'
    "  ],\n"
    '  "constraints": [\n'
    "    /* contraintes transverses : \"latence < 10 ms\", \"bus I2S/TDM\",\n"
    "       \"audio temps réel sous Linux\", \"alimentation embarquée\".\n"
    "       1 à 8 contraintes. */\n"
    '    "..."\n'
    "  ],\n"
    '  "friction_points": [\n'
    "    /* questions techniques épineuses que la rédaction devra trancher.\n"
    "       Exemples : \"comment partager la charge DSP/NPU ?\",\n"
    "       \"quel jitter buffer accepter ?\". 3 à 10 points. */\n"
    '    "..."\n'
    "  ],\n"
    '  "combos": [\n'
    "    /* paires de briques dont l'INTERACTION sera centrale et qui méritent\n"
    "       une recherche dédiée. 3 à 12 paires. */\n"
    '    {{ "a": "BriqueA", "b": "BriqueB", "why": "ce qu\'on cherche" }}\n'
    "  ],\n"
    '  "analogies": [\n'
    "    /* \"ce sujet fait penser à\" : 5 à 15 sujets adjacents, projets,\n"
    "       papiers, technologies connexes — même s'ils ne sont pas dans la\n"
    "       question. Sois généreux, on cherchera tout. */\n"
    '    "..."\n'
    "  ],\n"
    '  "unknowns": [\n'
    "    /* ce que TU NE SAIS PAS et qu'il faudra absolument trouver sur le\n"
    "       web. Sois honnête — c'est mieux que d'inventer. */\n"
    '    "..."\n'
    "  ]\n"
    "}}"
)


async def analyze_subject(
    question: str,
    tags: list[str] | None,
    llm: LLMClient,
    model: str,
    llm_logs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Phase A: produce a JSON brief describing the topic and what to search."""
    tag_str = ", ".join(tags) if tags else "(aucun)"
    user = _BRIEF_USER_TMPL.format(question=question, tags=tag_str)
    raw = await llm.ask(
        model, _BRIEF_SYSTEM, user, llm_logs,
        stage="discovery_brief", temperature=0.4,
    )
    brief = await llm.parse_json(raw, model, "discovery_brief", llm_logs)
    # Defensive normalisation
    out = {
        "reformulated": str(brief.get("reformulated") or question).strip(),
        "bricks": _norm_bricks(brief.get("bricks")),
        "constraints": _norm_strings(brief.get("constraints"), limit=12),
        "friction_points": _norm_strings(brief.get("friction_points"), limit=15),
        "combos": _norm_combos(brief.get("combos")),
        "analogies": _norm_strings(brief.get("analogies"), limit=20),
        "unknowns": _norm_strings(brief.get("unknowns"), limit=15),
    }
    return out


def _norm_strings(value, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    seen = set()
    for v in value:
        if isinstance(v, dict):
            v = v.get("text") or v.get("name") or json.dumps(v, ensure_ascii=False)
        s = str(v or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s[:200])
        if len(out) >= limit:
            break
    return out


def _norm_bricks(value) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out, seen = [], set()
    for b in value[:25]:
        if isinstance(b, str):
            b = {"name": b}
        if not isinstance(b, dict):
            continue
        name = str(b.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "name": name[:120],
            "kind": (b.get("kind") or "").strip()[:30],
            "why": str(b.get("why") or "").strip()[:300],
        })
    return out


def _norm_combos(value) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out, seen = [], set()
    for c in value[:15]:
        if not isinstance(c, dict):
            continue
        a = str(c.get("a") or "").strip()
        b = str(c.get("b") or "").strip()
        if not a or not b or a.lower() == b.lower():
            continue
        key = tuple(sorted([a.lower(), b.lower()]))
        if key in seen:
            continue
        seen.add(key)
        out.append({"a": a[:80], "b": b[:80],
                    "why": str(c.get("why") or "").strip()[:200]})
    return out


# ---------------------------------------------------------------------------
# Phase A.5: turn the brief into search seeds (sub_questions for the existing
# search_subquestions(...) pipeline).
# ---------------------------------------------------------------------------

def brief_to_sub_questions(brief: dict[str, Any], tags: list[str] | None = None) -> list[dict[str, str]]:
    """Build a search plan from the brief.

    Each brick gets several angles: datasheet, specs, benchmarks, troubleshooting.
    Each combo gets its own dedicated query. Each analogy gets a wider query.
    Each unknown gets a direct query. Each friction_point too.
    """
    seeds: list[tuple[str, str]] = []  # (group, query)

    # --- Bricks: 4 angles each
    for b in brief.get("bricks", []):
        name = b["name"]
        seeds.append(("brick", f'{name} datasheet specifications'))
        seeds.append(("brick", f'{name} architecture reference manual'))
        seeds.append(("brick", f'{name} benchmark performance latency'))
        seeds.append(("brick", f'{name} integration example tutorial'))

    # --- Combos: how-to and dataflow
    for c in brief.get("combos", []):
        a, b = c["a"], c["b"]
        seeds.append(("combo", f'{a} {b} integration interface'))
        seeds.append(("combo", f'{a} {b} dataflow latency'))
        if c.get("why"):
            seeds.append(("combo", f'{a} {b} {c["why"][:120]}'))

    # --- Friction points: open the doors
    for fp in brief.get("friction_points", []):
        seeds.append(("friction", fp))

    # --- Analogies: adjacent domains
    for an in brief.get("analogies", []):
        seeds.append(("analogy", an))

    # --- Unknowns: ask the web directly
    for u in brief.get("unknowns", []):
        seeds.append(("unknown", u))

    # --- Tag-only queries to keep coverage if user typed precise refs
    for t in tags or []:
        seeds.append(("tag", f'{t} reference design 2026'))

    # Dedupe + format as sub_questions for the existing search engine
    sub: list[dict[str, str]] = []
    seen = set()
    for group, q in seeds:
        q = re.sub(r"\s+", " ", q).strip()
        if not q or q.lower() in seen:
            continue
        seen.add(q.lower())
        sub.append({"id": f"SQ{len(sub):03d}", "question": q, "group": group})
    return sub


# ---------------------------------------------------------------------------
# Phase E. Synthesis of the corpus.
# ---------------------------------------------------------------------------

_SYNTH_SYSTEM = (
    "Tu es un Analyste de Recherche. On te donne le brief initial et un corpus "
    "de sources extraites du web. Ton rôle : synthétiser ce qui a été TROUVÉ, "
    "ce qui reste INCERTAIN, et ce qui est ABSENT du corpus. "
    "JAMAIS inventer : si une brique n'a pas de sources, dis-le. Produis un JSON."
)

_SYNTH_USER_TMPL = (
    "BRIEF :\n{brief}\n\n"
    "CORPUS (titre, domaine, longueur en kilo-caractères, snippet) :\n"
    "{corpus_summary}\n\n"
    "Réponds UN SEUL bloc JSON avec :\n"
    "{{\n"
    '  "established": [\n'
    "    /* faits clairement documentés par AU MOINS UNE source du corpus.\n"
    "       Cite l'source_id (SRC-xxx) pour chaque. */\n"
    '    {{ "fact": "...", "sources": ["SRC-..."] }}\n'
    "  ],\n"
    '  "uncertain": [\n'
    "    /* sujets sur lesquels les sources se contredisent ou sont vagues. */\n"
    '    {{ "topic": "...", "sources": ["SRC-..."], "tension": "..." }}\n'
    "  ],\n"
    '  "gaps": [\n'
    "    /* briques du brief pour lesquelles le corpus n'a RIEN d'utile. */\n"
    '    {{ "brick": "...", "what_is_missing": "..." }}\n'
    "  ],\n"
    '  "brick_links": [\n'
    "    /* observations sur la façon dont des briques se combinent\n"
    "       d\\'après les sources. */\n"
    '    {{ "a": "...", "b": "...", "observation": "...", "sources": ["SRC-..."] }}\n'
    "  ]\n"
    "}}"
)


def _corpus_summary_for_synth(corpus: dict[str, Any], max_sources: int = 80) -> str:
    """Concise list-form view of the corpus to fit a synthesis prompt."""
    lines = []
    base_path = corpus.get("__base_path__")  # injected by caller; optional
    for s in (corpus.get("sources") or [])[:max_sources]:
        sid = s.get("source_id", "?")
        dom = s.get("domain", "?")
        title = (s.get("title") or "")[:140]
        # snippet from clean/<sid>.txt if available
        snippet = ""
        cp = s.get("content_path")
        if base_path and cp:
            f = Path(base_path) / cp
            if f.exists():
                try:
                    txt = f.read_text(encoding="utf-8", errors="ignore")[:600]
                    snippet = re.sub(r"\s+", " ", txt).strip()[:300]
                except Exception:
                    pass
        lines.append(f"- {sid} [{dom}] {title} :: {snippet}")
    return "\n".join(lines)


async def synthesize_corpus(
    brief: dict[str, Any],
    corpus: dict[str, Any],
    base_path: Path,
    llm: LLMClient,
    model: str,
    llm_logs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    corpus_with_base = dict(corpus)
    corpus_with_base["__base_path__"] = str(base_path)
    corpus_summary = _corpus_summary_for_synth(corpus_with_base)
    brief_compact = json.dumps({
        "reformulated": brief.get("reformulated"),
        "bricks": [b["name"] for b in brief.get("bricks", [])],
        "constraints": brief.get("constraints"),
        "friction_points": brief.get("friction_points"),
    }, ensure_ascii=False, indent=2)
    user = _SYNTH_USER_TMPL.format(brief=brief_compact, corpus_summary=corpus_summary)
    raw = await llm.ask(
        model, _SYNTH_SYSTEM, user, llm_logs,
        stage="discovery_synth", temperature=0.3,
    )
    return await llm.parse_json(raw, model, "discovery_synth", llm_logs)


# ---------------------------------------------------------------------------
# Phase F. Plan generated from the synthesis (not from imagination).
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = (
    "Tu es un Architecte de Plan. Tu reçois (1) la question originale, (2) le brief, "
    "(3) la synthèse du corpus. Tu construis un plan de dossier ANCRÉ dans la "
    "synthèse : chaque chapitre couvre des faits réellement trouvés et chaque "
    "sous-section a une dette explicite envers des sources existantes. "
    "Si un sujet du brief est dans les 'gaps' (rien trouvé), tu peux le mentionner "
    "dans un chapitre 'Zones grises / à expérimenter' — mais tu ne fais pas un "
    "chapitre de 30 pages sur du vide."
)

_PLAN_USER_TMPL = (
    "QUESTION : {question}\n\n"
    "BRIEF :\n{brief}\n\n"
    "SYNTHÈSE :\n{synth}\n\n"
    "Produis UN SEUL bloc JSON :\n"
    "{{\n"
    '  "question_reformulated": "...",\n'
    '  "master_outline": [\n'
    '    {{\n'
    '      "chapter_title": "...",\n'
    '      "rationale": "pourquoi ce chapitre, ancré dans la synthèse",\n'
    '      "anchor_sources": ["SRC-..."],\n'
    '      "sub_sections": [\n'
    '        {{ "title": "...", "covers": "ce que la section répondra",\n'
    '           "anchor_sources": ["SRC-..."] }}\n'
    "      ]\n"
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "Règles :\n"
    "- Aligne le NOMBRE de chapitres sur la richesse réelle du corpus. Pas de "
    "  remplissage. 5 chapitres avec 100 sources solides > 15 chapitres creux.\n"
    "- Chaque sous-section DOIT lister au moins 1 anchor_source (SRC-...). "
    "  Si tu ne peux pas, ne crée pas la sous-section."
)


async def build_plan_from_corpus(
    question: str,
    brief: dict[str, Any],
    synthesis: dict[str, Any],
    llm: LLMClient,
    model: str,
    coder_model: str,
    llm_logs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Phase F: emit a master_outline grounded in the synthesis."""
    brief_compact = json.dumps({
        "reformulated": brief.get("reformulated"),
        "bricks": [b["name"] for b in brief.get("bricks", [])],
        "constraints": brief.get("constraints"),
    }, ensure_ascii=False, indent=2)
    synth_compact = json.dumps(synthesis, ensure_ascii=False, indent=2)[:14000]
    user = _PLAN_USER_TMPL.format(
        question=question, brief=brief_compact, synth=synth_compact,
    )
    raw = await llm.ask(
        model, _PLAN_SYSTEM, user, llm_logs,
        stage="discovery_plan", temperature=0.5,
    )
    try:
        parsed = await llm.parse_json(raw, model, "discovery_plan", llm_logs)
    except Exception:
        # one repair via the JSON-coder model
        parsed = await llm.parse_json(raw, coder_model, "discovery_plan_repair", llm_logs)

    return _normalize_plan_output(parsed, fallback_question=question)


def _normalize_plan_output(parsed: Any, fallback_question: str) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {"question_reformulated": fallback_question, "master_outline": [], "sub_questions": []}
    question_reformulated = str(parsed.get("question_reformulated") or fallback_question).strip()
    outline_raw = parsed.get("master_outline") or []
    if not isinstance(outline_raw, list):
        outline_raw = []
    outline: list[dict[str, Any]] = []
    sub_questions: list[dict[str, str]] = []
    sq_id = 0
    for ch in outline_raw:
        if not isinstance(ch, dict):
            continue
        title = str(ch.get("chapter_title") or ch.get("title") or "").strip()
        if not title:
            continue
        subs_raw = ch.get("sub_sections") or ch.get("sections") or []
        subs: list[dict[str, Any]] = []
        for s in subs_raw if isinstance(subs_raw, list) else []:
            if isinstance(s, str):
                s = {"title": s}
            if not isinstance(s, dict):
                continue
            stitle = str(s.get("title") or s.get("name") or "").strip()
            if not stitle:
                continue
            anchors = [str(x) for x in (s.get("anchor_sources") or []) if x]
            covers = str(s.get("covers") or "").strip()
            subs.append({"title": stitle, "covers": covers, "anchor_sources": anchors})
            sub_questions.append({
                "id": f"SQ{sq_id:03d}",
                "question": f"{stitle} ({title})",
                "anchor_sources": anchors,
            })
            sq_id += 1
        outline.append({
            "chapter_title": title,
            "rationale": str(ch.get("rationale") or "").strip(),
            "anchor_sources": [str(x) for x in (ch.get("anchor_sources") or []) if x],
            "sub_sections": subs,
        })
    return {
        "question_reformulated": question_reformulated,
        "master_outline": outline,
        "sub_questions": sub_questions,
    }


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_brief(run_dir: Path, brief: dict[str, Any]) -> None:
    save_json(run_dir / "discovery_brief.json", brief)


def save_synthesis(run_dir: Path, synth: dict[str, Any]) -> None:
    save_json(run_dir / "discovery_synthesis.json", synth)
