import json
import os
import time
import re
from datetime import date
from pathlib import Path
from typing import Any, Callable
from .config import DossierConfig
from .llm import LLMClient
from .utils import emit_progress, markdown_anchor
from .logging_config import get_logger

log = get_logger("aidocgen.writer")

def _date_instruction() -> str:
    today = date.today().strftime("%d/%m/%Y")
    return f"Nous sommes le {today}. Priorise les informations les plus récentes. Il est important que les infos soient les plus récentes possible."

class Writer:
    def __init__(self, config: DossierConfig, llm: LLMClient):
        self.config = config
        self.llm = llm

    def parse_markdown_outline(self, text: str) -> list[dict[str, Any]]:
        """Parse Markdown outline with 3 levels: Parties -> Chapters -> Sections."""
        result = []
        lines = text.split('\n')

        current_party: dict[str, Any] | None = None
        current_chapter: dict[str, Any] | None = None

        party_regex = r'^#{1,2}\s*(?:Partie|Part)\s*([IVX\d]+)[\s\:\-\u2013\u2014]*\s*(.*?)(?:\s*#|$)'
        chap_regex = r'^#{1,4}\s*(?:Chapitre|Chapter)\s*[\d\.]*\s*[\:\-\.\s\u2014]*\s*(.*?)(?:\s*#|$)'

        for line in lines:
            line = line.strip()
            if not line:
                continue

            party_match = re.search(party_regex, line, re.IGNORECASE)
            if party_match:
                if current_chapter and current_chapter.get('sub_sections'):
                    if current_party:
                        current_party.setdefault('chapters', []).append(current_chapter)
                    else:
                        result.append(current_chapter)
                current_chapter = None

                party_title = f"Partie {party_match.group(1)}"
                if party_match.group(2):
                    party_title += f" : {party_match.group(2).strip()}"
                current_party = {"party_title": party_title, "chapters": []}
                result.append(current_party)
                continue

            chap_match = re.search(chap_regex, line, re.IGNORECASE)
            if chap_match:
                if current_chapter and current_chapter.get('sub_sections'):
                    if current_party:
                        current_party.setdefault('chapters', []).append(current_chapter)
                    else:
                        result.append(current_chapter)

                title = chap_match.group(1).strip()
                if len(title) > 3:
                    current_chapter = {"chapter_title": title, "sub_sections": []}
                continue

            sec_match = re.search(r'^[*\-]\s*\*{0,2}(?:[\d\.]+)?\s*(.*?)\*{0,2}\s*$', line)
            if sec_match and current_chapter is not None:
                sec_title = sec_match.group(1).strip()
                if len(sec_title) > 3:
                    current_chapter.setdefault('sub_sections', []).append({"title": sec_title, "brief": "Analyse detaillee."})

        if current_chapter and current_chapter.get('sub_sections'):
            if current_party:
                current_party.setdefault('chapters', []).append(current_chapter)
            else:
                result.append(current_chapter)

        filtered = []
        for item in result:
            if 'party_title' in item:
                item['chapters'] = [c for c in item.get('chapters', []) if c.get('sub_sections')]
                if item['chapters']:
                    filtered.append(item)
            elif 'chapter_title' in item and item.get('sub_sections'):
                filtered.append(item)

        return filtered

    def normalize_outline(self, data: Any) -> list[dict[str, Any]]:
        """Recursive extraction of chapters from any JSON structure."""
        normalized = []
        def walk(obj, parent_title=None):
            if isinstance(obj, dict):
                title = obj.get("chapter_title") or obj.get("title") or obj.get("titre") or obj.get("part_title") or obj.get("nom") or obj.get("intitule")
                content = obj.get("sub_sections") or obj.get("sections") or obj.get("chapitres") or obj.get("chapters") or obj.get("contenu") or obj.get("points")
                if title and isinstance(content, list) and len(content) > 0:
                    if all(isinstance(x, str) for x in content[:2]):
                        full_title = f"{parent_title} - {title}" if parent_title else str(title)
                        normalized.append({"chapter_title": full_title, "sub_sections": [{"title": s, "brief": "Analyse."} for s in content]})
                    else:
                        for item in content: walk(item, parent_title=title)
                else:
                    for v in obj.values(): walk(v, parent_title=parent_title)
            elif isinstance(obj, list):
                for v in obj: walk(v, parent_title=parent_title)
        walk(data); return normalized

    def generate_sub_questions_from_outline(self, outline: list[dict[str, Any]], main_question: str, tags: list[str] = None, max_questions: int | None = None) -> list[dict[str, str]]:
        # Pull the cap from config so it's tunable per-run (UI / env).
        # 0 means "no cap" → use one query per (chapter, sub_section) pair.
        if max_questions is None:
            cfg_cap = getattr(self.config, "web_max_sub_questions", 0) or 0
            max_questions = cfg_cap if cfg_cap > 0 else 10**6
        """Generate search sub-questions from the outline sections."""
        sub_questions = []
        sq_id = 1

        main_topic = self._extract_main_topic(main_question)
        tag_str = " ".join(tags) if tags else ""

        for item in outline:
            if "chapters" in item:
                p_title = item.get("party_title", "")
                p_keywords = self._extract_keywords(p_title) if p_title else []
                for chapter in item.get("chapters", []):
                    c_title = chapter.get("chapter_title", "")
                    c_keywords = self._extract_keywords(c_title)
                    for section in chapter.get("sub_sections", []):
                        s_title = section.get("title", "")
                        query = self._build_search_query(s_title, c_keywords + p_keywords, main_topic, tags)
                        sub_questions.append({"id": f"SQ{sq_id}", "question": query})
                        sq_id += 1
                        if sq_id > max_questions:
                            return sub_questions
            else:
                c_title = item.get("chapter_title", "")
                c_keywords = self._extract_keywords(c_title)
                for section in item.get("sub_sections", []):
                    s_title = section.get("title", "")
                    query = self._build_search_query(s_title, c_keywords, main_topic, tags)
                    sub_questions.append({"id": f"SQ{sq_id}", "question": query})
                    sq_id += 1
                    if sq_id > max_questions:
                        return sub_questions

        return sub_questions

    def _extract_main_topic(self, question: str) -> str:
        question = re.sub(r'(?i)^(fais|fait|faire|ecris|ecrire|redige|rediger|cree|creer|donne|donner)\s+(moi\s+)?(un\s+|une\s+)?(dossier|rapport|document|etude|article)\s+(complet|detaille|completa?)?\s*(sur|de|about)?\s*', '', question)
        stop_words = {"le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "mais", "sur", "dans", "pour", "avec",
                      "the", "a", "an", "and", "or", "but", "for", "with", "about", "of", "in", "to", "on"}
        words = re.findall(r'\b[A-Z\u00c0-\u00ff][a-zA-Z\u00c0-\u00ff]*\b|\b[a-zA-Z\u00c0-\u00ff]{4,}\b', question)
        keywords = [w for w in words if w.lower() not in stop_words]
        proper_nouns = [w for w in words if w[0].isupper() and w.lower() not in stop_words]
        other_terms = [w for w in keywords if w not in proper_nouns]
        result = proper_nouns[:3] + other_terms[:2]
        return " ".join(result)

    def _extract_keywords(self, text: str) -> list[str]:
        stop_words = {"le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "mais", "sur", "dans", "pour", "avec", "etc",
                      "the", "a", "an", "and", "or", "but", "for", "with", "about", "of", "in", "to", "on", "as"}
        words = re.findall(r'\b[a-zA-Z\u00c0-\u00ff]{3,}\b', text.lower())
        return [w for w in words if w not in stop_words][:6]

    def _section_match_tokens(self, text: str) -> set[str]:
        """Tokenize for image-matching: includes acronyms (DSP, FFT, IIR\u2026)
        which are common in technical section titles and would otherwise
        be stripped by the 3+ char filter or hidden by .lower()."""
        tokens: set[str] = set()
        # Long words, case-insensitive (>= 4 chars)
        for w in re.findall(r"[A-Za-z\u00c0-\u00ff]{4,}", text):
            tokens.add(w.lower())
        # Acronyms: 2-6 char ALL-CAPS sequences (DSP, FFT, LUFS, EBU, DAW\u2026)
        for w in re.findall(r"\b[A-Z]{2,6}\b", text):
            tokens.add(w.lower())
        # Stop words (long enough to slip past first pass)
        tokens -= {"avec", "dans", "pour", "with", "from", "this", "that",
                   "their", "your", "our"}
        return tokens

    def _build_search_query(self, section_title: str, context_keywords: list[str], main_topic: str, tags: list[str] = None) -> str:
        section_keywords = self._extract_keywords(section_title)
        query_parts = section_keywords[:4]
        for kw in context_keywords[:2]:
            if kw not in query_parts:
                query_parts.append(kw)
        main_terms = main_topic.split()
        for term in main_terms[:2]:
            if term.lower() not in [p.lower() for p in query_parts]:
                query_parts.append(term)
        if tags:
            for tag in tags[:2]:
                tag_clean = re.sub(r'[^a-zA-Z0-9]', '', tag)
                if tag_clean.lower() not in [p.lower() for p in query_parts]:
                    query_parts.append(tag_clean)
        return " ".join(query_parts)

    # ─────────────────────────────────────────────
    # PLAN DOSSIER
    # ─────────────────────────────────────────────

    async def plan_dossier(self, question: str, detail_level: str, llm_logs: list[dict[str, Any]], prompt_type: str = "generic", language: str = "fr", presearch_results: dict[str, Any] | None = None, run_dir: Path | None = None, coder_model_override: str | None = None, tags: list[str] = None) -> dict[str, Any]:
        lang_name = {"fr": "Francais", "en": "English", "es": "Espanol", "de": "Deutsch", "ru": "Russkiy"}.get(language, "Francais")
        debug_info = {"attempts": [], "planner_prompt": {"system": "", "user": ""}}

        tag_instruction = ""
        if tags:
            tag_str = ", ".join([f"#{t}" for t in tags])
            tag_instruction = f"\n\nIMPORTANT: Ce dossier DOIT se concentrer exclusivement sur les sujets lies aux tags suivants: {tag_str}. Toute section ou information hors de ce scope doit etre evitee."

        if presearch_results and len(presearch_results) > 0:
            def sanitize(t): return re.sub(r'http[s]?://\S+', '', str(t or ""))
            web_context = "\n".join([f"- {sanitize(l.get('title'))}: {sanitize(l.get('snippet'))}" for l in presearch_results[:15]])
        else: web_context = "Pas de donnees web."

        # STEP 1: RICH DRAFT
        chapter_counts = {"synthetic": "5-8", "medium": "10-14", "dissertation": "15-20"}
        chapter_count = chapter_counts.get(detail_level, "10-14")
        draft_sys = (
            f"Tu es un Expert Technique Senior. {_date_instruction()} "
            f"Mission: plan de DISSERTATION TECHNIQUE ({chapter_count} chapitres) en {lang_name}.{tag_instruction}\n\n"
            f"REGLES STRICTES:\n"
            f"- ZERO historique, ZERO contexte general, ZERO philosophie\n"
            f"- 100% PRATIQUE: architecture, implementation, specs, configurations, benchmarks\n"
            f"- Chaque section doit repondre a un besoin concret d'ingenieur\n"
            f"- Privilegier: schemas d'architecture, comparatifs techniques, guides d'implementation\n"
            f"- Inclure: troubleshooting, limitations connues, best practices"
        )
        draft_user = f"Sujet: {question}\nWEB:\n{web_context}\n\nReflechis (Thinking) puis donne le sommaire detaille en Markdown apres '---'."
        debug_info["planner_prompt"] = {"system": draft_sys, "user": draft_user}

        full_res = await self.llm.ask(self.config.planner_model, draft_sys, draft_user, llm_logs, "planner_rich_draft", temperature=0.8)
        debug_info["planner_response_raw"] = full_res
        if run_dir: (run_dir / "planner_draft.txt").write_text(full_res, encoding="utf-8")
        draft_md = full_res.split("---", 1)[-1] if "---" in full_res else full_res

        # STEP 2: DETERMINISTIC EXTRACTION
        if run_dir: await emit_progress(None, run_dir, "planner", "Extraction du plan...")
        master_outline = self.parse_markdown_outline(draft_md)

        if len(master_outline) >= 5:
            sub_questions = self.generate_sub_questions_from_outline(master_outline, question, tags=tags)
            debug_info["sub_questions_generated"] = len(sub_questions)
            debug_info["tags_used"] = tags
            if run_dir: (run_dir / "planner_debug.json").write_text(json.dumps(debug_info, ensure_ascii=False, indent=2))
            return {"question_reformulated": question, "master_outline": master_outline, "sub_questions": sub_questions}

        # STEP 3: IA FALLBACK
        coder_model = coder_model_override or self.config.planner_book_model_4_json
        for attempt in range(2):
            if run_dir: await emit_progress(None, run_dir, "planner", f"Recours IA ({attempt+1}/2)")
            raw_json = await self.llm.ask(coder_model, "Ingenieur JSON.", f"Convertis en JSON master_outline :\n{draft_md}", llm_logs, "planner_crystallize", temperature=0.1)
            attempt_info = {"attempt": attempt + 1, "raw_response": raw_json[:2000], "success": False}
            try:
                parsed = await self.llm.parse_json(raw_json, coder_model, "planner", llm_logs)
                master_outline = self.normalize_outline(parsed)
                attempt_info["success"] = True
                attempt_info["chapters_found"] = len(master_outline)
                debug_info["attempts"].append(attempt_info)
                if len(master_outline) >= 3:
                    sub_questions = self.generate_sub_questions_from_outline(master_outline, question, tags=tags)
                    debug_info["sub_questions_generated"] = len(sub_questions)
                    debug_info["tags_used"] = tags
                    if run_dir: (run_dir / "planner_debug.json").write_text(json.dumps(debug_info, ensure_ascii=False, indent=2))
                    return {"question_reformulated": question, "master_outline": master_outline, "sub_questions": sub_questions}
            except Exception as e:
                attempt_info["error"] = str(e)
            debug_info["attempts"].append(attempt_info)

        if run_dir: (run_dir / "planner_debug.json").write_text(json.dumps(debug_info, ensure_ascii=False, indent=2))
        raise RuntimeError("Echec total de l'extraction du plan.")

    # ─────────────────────────────────────────────
    # IMPROVED CLAIM MATCHING
    # ─────────────────────────────────────────────

    def _score_claim_relevance(self, claim: dict[str, Any], section_title: str, chapter_title: str, party_title: str) -> float:
        """Score how relevant a claim is to a section using weighted keyword matching."""
        claim_text = str(claim.get("claim_text") or "").lower()
        if not claim_text:
            return 0.0

        # Build keyword sets with weights
        section_keywords = set(self._extract_keywords(section_title))
        chapter_keywords = set(self._extract_keywords(chapter_title))
        party_keywords = set(self._extract_keywords(party_title))

        claim_words = set(re.findall(r'\b[a-z\u00e0-\u00ff]{3,}\b', claim_text))

        # Weighted scoring: section keywords matter most
        score = 0.0
        section_matches = len(section_keywords & claim_words)
        chapter_matches = len(chapter_keywords & claim_words)
        party_matches = len(party_keywords & claim_words)

        if section_keywords:
            score += (section_matches / len(section_keywords)) * 0.6
        if chapter_keywords:
            score += (chapter_matches / len(chapter_keywords)) * 0.3
        if party_keywords:
            score += (party_matches / len(party_keywords)) * 0.1

        return score

    # ─────────────────────────────────────────────
    # 2-PASS WRITING: DRAFT + CRITIQUE + REWRITE
    # ─────────────────────────────────────────────

    async def _critique_section(self, content: str, s_title: str, c_title: str, llm_logs: list[dict[str, Any]], language: str) -> str:
        """Ask LLM to critique a draft section and suggest improvements."""
        lang_name = {"fr": "Francais", "en": "English", "ru": "Russkiy"}.get(language, "Francais")
        system = (
            f"Tu es un relecteur scientifique exigeant. Langue: {lang_name}. {_date_instruction()}\n"
            f"Analyse cette section de dossier et identifie:\n"
            f"1. Les lacunes factuelles ou manques d'information\n"
            f"2. Les affirmations vagues qui devraient etre etayees par des donnees\n"
            f"3. Les transitions faibles entre paragraphes\n"
            f"4. Les opportunites d'approfondir avec des exemples concrets\n"
            f"5. La qualite de la structure et de l'argumentation\n\n"
            f"Sois constructif et precis. Donne 3-5 points d'amelioration prioritaires."
        )
        prompt = f"Chapitre: {c_title}\nSection: {s_title}\n\nContenu a critiquer:\n{content}"
        return await self.llm.ask(self.config.writer_model, system, prompt, llm_logs, "critique", temperature=0.4)

    async def _rewrite_section(self, original: str, critique: str, claims_context: str, s_title: str, c_title: str, p_title: str, llm_logs: list[dict[str, Any]], language: str, tags: list[str] = None) -> str:
        """Rewrite a section incorporating critique feedback and additional claims."""
        lang_name = {"fr": "Francais", "en": "English", "ru": "Russkiy"}.get(language, "Francais")
        tag_instruction = ""
        if tags:
            tag_str = ", ".join([f"#{t}" for t in tags])
            tag_instruction = f"\nReste STRICTEMENT focalise sur les tags: {tag_str}."

        system = (
            f"Tu es un redacteur scientifique senior. {_date_instruction()} Langue: {lang_name}.\n"
            f"Tu recois un brouillon, les critiques d'un relecteur, et des donnees factuelles supplementaires.\n"
            f"Ta mission: reecrire la section en integrant TOUTES les ameliorations suggerees.\n"
            f"Regles:\n"
            f"- Enrichis avec des chiffres, dates, exemples concrets\n"
            f"- Ameliore les transitions entre paragraphes\n"
            f"- Cite les sources avec [CLM-id] quand tu utilises des faits du contexte\n"
            f"- Minimum {self.config.writer_min_words_per_section} mots, objectif {self.config.writer_target_words_per_section} mots\n"
            f"- Style academique mais accessible{tag_instruction}"
        )
        prompt = (
            f"Partie: {p_title}\nChapitre: {c_title}\nSection: {s_title}\n\n"
            f"=== BROUILLON INITIAL ===\n{original}\n\n"
            f"=== CRITIQUE DU RELECTEUR ===\n{critique}\n\n"
            f"=== DONNEES FACTUELLES SUPPLEMENTAIRES ===\n{claims_context}\n\n"
            f"Reecris INTEGRALEMENT la section en tenant compte de tous ces elements."
        )
        return await self.llm.ask(self.config.writer_model, system, prompt, llm_logs, "rewrite", temperature=0.6)

    async def write_sections(self, planner: dict[str, Any], claims: list[dict[str, Any]], llm_logs: list[dict[str, Any]], progress_cb: Callable | None, run_dir: Path | None, language: str = "fr", tags: list[str] = None) -> dict[str, Any]:
        outline = planner.get("master_outline", [])
        lang_name = {"fr": "Francais", "en": "English", "es": "Espanol", "de": "Deutsch", "ru": "Russkiy"}.get(language, "Francais")
        tasks = []

        tag_instruction = ""
        if tags:
            tag_str = ", ".join([f"#{t}" for t in tags])
            tag_instruction = f"\nIMPORTANT: Reste STRICTEMENT focalise sur les tags: {tag_str}. Ignore tout contenu hors sujet."

        for item in outline:
            if "chapters" in item:
                p_title = item.get("party_title", "")
                for chapter in item.get("chapters", []):
                    c_title = chapter.get("chapter_title", "")
                    for s in chapter.get("sub_sections", []):
                        tasks.append((p_title, c_title, s))
            else:
                for s in item.get("sub_sections", []):
                    tasks.append(("", item.get("chapter_title", ""), s))

        # ── RESUME: load already-written sections ──
        sections_content = []
        start_idx = 0
        if run_dir:
            partial_path = run_dir / "sections.json"
            if partial_path.exists():
                try:
                    partial = json.loads(partial_path.read_text(encoding="utf-8"))
                    existing = partial.get("sections", [])
                    # If the cache already has all sections, return as-is.
                    if existing and len(existing) >= len(tasks):
                        log.info("RESUME: all sections present, skipping writer",
                                 extra={"count": len(existing)})
                        return {"sections": existing}
                    if existing and len(existing) < len(tasks):
                        sections_content = existing
                        start_idx = len(existing)
                        log.info("RESUME: continuing from section",
                                 extra={"start_idx": start_idx, "total": len(tasks)})
                except Exception:
                    pass

        enable_critique = self.config.writer_enable_critique

        for idx, (p_title, c_title, sec) in enumerate(tasks, 1):
            if idx <= start_idx:
                continue
            s_title = sec.get("title")
            await emit_progress(progress_cb, run_dir, "sections", f"Section {idx}/{len(tasks)}: {s_title}")

            # ── IMPROVED CLAIM MATCHING: scored relevance ──
            scored_claims = []
            for c in claims:
                score = self._score_claim_relevance(c, s_title, c_title, p_title)
                if score > 0.05:
                    scored_claims.append((score, c))
            scored_claims.sort(key=lambda x: x[0], reverse=True)
            relevant = [c for _, c in scored_claims[:40]]

            context = json.dumps(relevant[:30], ensure_ascii=False)

            # ── PASS 1: INITIAL DRAFT ──
            prompt = (
                f"Dissertation Section: {s_title}\nPartie: {p_title}\nChapitre: {c_title}\n"
                f"Brief: {sec.get('brief')}\n"
                f"Context (verified claims):\n{context}\n\n"
                f"Rules: Technical depth, min {self.config.writer_min_words_per_section} words, "
                f"target {self.config.writer_target_words_per_section} words, cite [CLM-id]. "
                f"LANGUAGE: {lang_name}.{tag_instruction}"
            )
            writer_system = (
                f"Tu es un redacteur technique senior. {_date_instruction()} Langue: {lang_name}.\n\n"
                f"STYLE OBLIGATOIRE: Dissertation PUREMENT TECHNIQUE avec ILLUSTRATIONS.\n"
                f"- ZERO historique, ZERO introduction generale, ZERO philosophie\n"
                f"- Va DROIT aux specifications, implementations, configurations\n"
                f"- Donne des exemples de code, commandes, configurations quand pertinent\n"
                f"- Inclus des chiffres, benchmarks, comparatifs\n"
                f"- Mentionne les limitations, edge cases, troubleshooting\n"
                f"- Cite tes sources avec [CLM-id]\n"
                f"- Si tu ne sais pas, dis-le clairement plutot que d'inventer\n\n"
                f"ILLUSTRATIONS REQUISES (inclus-les quand c'est pertinent):\n"
                f"1. DIAGRAMMES MERMAID (rendus en images dans le PDF final):\n"
                f"   IMPORTANT: TOUJOURS utiliser l'orientation verticale 'TD' (top-down), JAMAIS 'LR' (left-right).\n"
                f"   ```mermaid\n"
                f"   graph TD\n"
                f"     SAI7[SAI7 TX] --> AUDIOMIX[AUDIOMIX Router]\n"
                f"     AUDIOMIX --> DSP[DSP HiFi4]\n"
                f"     DSP --> SOF[SOF Firmware]\n"
                f"   ```\n"
                f"2. FLOWCHARTS pour les sequences (boot, reset, power-on):\n"
                f"   ```mermaid\n"
                f"   flowchart TD\n"
                f"     A[Power On] --> B[Enable PD_AUDIO]\n"
                f"     B --> C[Start PLL_AUDIO]\n"
                f"     C --> D[Configure SAI7]\n"
                f"     D --> E[Load SOF firmware]\n"
                f"   ```\n"
                f"3. SEQUENCE DIAGRAMS pour les interactions:\n"
                f"   ```mermaid\n"
                f"   sequenceDiagram\n"
                f"     AP->>DSP: run-stall bit = 0\n"
                f"     DSP->>SOF: boot firmware\n"
                f"     SOF->>AP: ready IPC\n"
                f"   ```\n"
                f"4. STATE DIAGRAMS pour les etats:\n"
                f"   ```mermaid\n"
                f"   stateDiagram-v2\n"
                f"     [*] --> Reset\n"
                f"     Reset --> Idle: release stall\n"
                f"     Idle --> Active: start pipeline\n"
                f"   ```\n"
                f"5. TABLEAUX pour les registres, pins, bit fields (Markdown standard)\n\n"
                f"REGLES:\n"
                f"- Utilise Mermaid quand il y a plus de 2 blocs/etapes a illustrer\n"
                f"- Au moins 1 diagramme Mermaid toutes les 2-3 sections\n"
                f"- Reste simple: 4-8 nodes max par diagramme\n"
                f"- Les tableaux restent en Markdown standard"
            )
            draft = await self.llm.ask(
                self.config.writer_model,
                writer_system,
                prompt, llm_logs, "writing"
            )

            final_content = draft

            # ── PASS 2: CRITIQUE + REWRITE ──
            if enable_critique:
                await emit_progress(progress_cb, run_dir, "sections",
                    f"Critique {idx}/{len(tasks)}: {s_title}")

                critique = await self._critique_section(draft, s_title, c_title, llm_logs, language)

                # Get additional claims not used in first pass
                extra_claims = [c for _, c in scored_claims[30:60]]
                extra_context = json.dumps(extra_claims[:20], ensure_ascii=False) if extra_claims else "Pas de donnees supplementaires."

                final_content = await self._rewrite_section(
                    draft, critique, extra_context, s_title, c_title, p_title,
                    llm_logs, language, tags
                )

            sections_content.append({
                "type": "section",
                "p_title": p_title,
                "c_title": c_title,
                "s_title": s_title,
                "content": final_content
            })

            # ── INCREMENTAL SAVE after each section ──
            if run_dir:
                (run_dir / "sections.json").write_text(
                    json.dumps({"sections": sections_content}, ensure_ascii=False),
                    encoding="utf-8"
                )

        return {"sections": sections_content}

    # ─────────────────────────────────────────────
    # ILLUSTRATIONS — pick images from cited sources, optional AI fallback
    # ─────────────────────────────────────────────

    @staticmethod
    def _score_image(image: dict, section_title: str, chapter_title: str) -> float:
        """Relevance score on [0..1] : combines vision caption (priorité) and alt text.

        Returns 0 if the image was tagged DECORATIVE by vision, or has no usable text.
        Higher score = more relevant to the section title / chapter title.
        """
        # vision_caption=="" => DECORATIVE -> never use
        vc = image.get("vision_caption")
        if vc == "":
            return 0.0
        text = (vc or "") + " " + (image.get("alt") or "")
        text = text.lower().strip()
        if not text:
            return 0.0
        title_words = set(re.findall(r'\b[a-zà-ÿ]{4,}\b',
                                     f"{section_title} {chapter_title}".lower()))
        text_words = set(re.findall(r'\b[a-zà-ÿ]{4,}\b', text))
        if not title_words or not text_words:
            return 0.0
        overlap = len(title_words & text_words)
        # 0..1 score: fraction of section keywords covered by the caption
        coverage = overlap / max(1, len(title_words))
        # vision captions deserve a small boost vs alt-only text
        boost = 0.15 if vc else 0.0
        return min(1.0, coverage + boost)

    async def _generate_ai_image(self, prompt: str, run_dir: Path, idx: int) -> str | None:
        """Generate one illustration via an external image API. Returns local path or None.

        Requires IMAGE_GEN_API_KEY in env. Uses Replicate's Flux Schnell by default.
        IMAGE_GEN_PROVIDER=replicate (default) or 'openai'.
        """
        api_key = os.getenv("IMAGE_GEN_API_KEY", "").strip()
        if not api_key:
            log.warning("IMAGE_GEN_API_KEY missing — skipping AI illustration "
                        "(section will have no image)")
            return None
        provider = os.getenv("IMAGE_GEN_PROVIDER", "replicate").lower()
        images_dir = run_dir / "ai_images"
        images_dir.mkdir(exist_ok=True)
        out = images_dir / f"img_{idx:03d}.png"

        import httpx, asyncio as _asyncio
        async with httpx.AsyncClient(timeout=120.0) as client:
            if provider == "openai":
                resp = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": "gpt-image-1", "prompt": prompt[:1000], "size": "1024x1024", "n": 1},
                )
                resp.raise_for_status()
                data = resp.json()
                img_url = data["data"][0].get("url")
                if not img_url:
                    b64 = data["data"][0].get("b64_json")
                    if not b64:
                        raise RuntimeError(f"openai image gen: no url and no b64 in response")
                    import base64
                    out.write_bytes(base64.b64decode(b64))
                    return str(out)
                bin_resp = await client.get(img_url)
                out.write_bytes(bin_resp.content)
                return str(out)
            # Replicate Flux Schnell
            create = await client.post(
                "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json",
                         "Prefer": "wait=60"},
                json={"input": {"prompt": prompt[:1000], "aspect_ratio": "16:9",
                                "output_format": "png", "num_outputs": 1}},
            )
            create.raise_for_status()
            data = create.json()
            output = data.get("output")
            if not output:
                pid = data.get("id")
                if not pid:
                    raise RuntimeError(f"replicate: no output and no prediction id ({data})")
                for _ in range(30):
                    await _asyncio.sleep(2)
                    poll = await client.get(
                        f"https://api.replicate.com/v1/predictions/{pid}",
                        headers={"Authorization": f"Bearer {api_key}"})
                    pdata = poll.json()
                    if pdata.get("status") in ("succeeded", "failed", "canceled"):
                        output = pdata.get("output")
                        break
                if not output:
                    raise RuntimeError(f"replicate prediction did not finish in 60s")
            img_url = output[0] if isinstance(output, list) else output
            bin_resp = await client.get(img_url)
            out.write_bytes(bin_resp.content)
            return str(out)

    async def inject_illustrations(
        self, sections_data: dict[str, Any], claims: list[dict[str, Any]],
        corpus: dict[str, Any], run_dir: Path | None,
        generate_ai: bool = False, language: str = "fr",
    ) -> dict[str, Any]:
        """Mutate sections to add one image per section, drawn from cited sources.
        Falls back to AI gen if generate_ai=True and no corpus image is available."""
        if not sections_data or "sections" not in sections_data:
            return sections_data

        # claim_id -> source_id
        claim_to_src = {c.get("claim_id"): c.get("source_id") for c in (claims or [])}
        # source_id -> source dict
        src_by_id = {s.get("source_id"): s for s in (corpus or {}).get("sources", [])}

        # Top-ranked sources from the shortlist, used as last-resort image
        # pool when neither citations nor title-matching yield an image.
        shortlist_top: list[tuple[str, float]] = []
        if run_dir is not None:
            sl_path = run_dir / "shortlist.json"
            if sl_path.exists():
                try:
                    sl = json.loads(sl_path.read_text(encoding="utf-8"))
                    items = sl.get("shortlist", []) if isinstance(sl, dict) else []
                    shortlist_top = [(it.get("source_id", ""), it.get("score", 0.0))
                                     for it in items[:150] if it.get("source_id")]
                except Exception as e:
                    log.warning("could not load shortlist for image fallback",
                                extra={"error": str(e)})

        used_urls: set[str] = set()
        # If we resume after a crash, some sections already contain an image.
        # Seed used_urls with them AND skip those sections so we don't inject
        # a second illustration on top of the existing one.
        already_illustrated: set[int] = set()
        for idx, sec in enumerate(sections_data.get("sections", []) or []):
            existing_urls = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', sec.get("content") or "")
            if existing_urls:
                used_urls.update(existing_urls)
                already_illustrated.add(idx)

        threshold = float(getattr(self.config, "vision_score_threshold", 0.5))

        ai_idx = 0
        for idx, sec in enumerate(sections_data.get("sections", [])):
            if idx in already_illustrated:
                continue
            content = sec.get("content") or ""
            if not content:
                continue
            s_title = sec.get("title", "")
            c_title = sec.get("chapter_title", "")
            # The writer cites with [CLM-xxx] where xxx may be:
            #   - a real claim_id (claim_007, …)         → lookup claim_to_src
            #   - a source_id (SRC-abc123…)              → use directly
            #   - the source_id stripped of its prefix   → re-prefix and try
            # Also accept bare [SRC-xxx] citations.
            cited_tokens = list(dict.fromkeys(
                re.findall(r'\[(?:CLM|SRC)-([A-Za-z0-9_\-]+)\]', content)
            ))

            # Collect ALL candidates from cited sources + shortlist top-150,
            # score them all via _score_image (uses vision_caption when
            # available), then keep the best one — IF it clears the
            # relevance threshold. No image is better than a hors-sujet one.
            candidates: list[tuple[float, dict, str]] = []

            def _add_candidates(src: dict | None):
                if not src:
                    return
                for img in (src.get("images") or [])[:5]:
                    if img.get("url") in used_urls:
                        continue
                    score = self._score_image(img, s_title, c_title)
                    if score > 0:
                        candidates.append((score, img, src.get("url", "")))

            for cid in cited_tokens[:12]:
                src_id = (
                    claim_to_src.get(f"CLM-{cid}")
                    or claim_to_src.get(cid)
                    or (cid if cid in src_by_id else None)
                    or (f"SRC-{cid}" if f"SRC-{cid}" in src_by_id else None)
                )
                _add_candidates(src_by_id.get(src_id) if src_id else None)

            # Title-keyword sources (fallback A folded into the candidate pool)
            key_set = self._section_match_tokens(f"{s_title} {c_title}")
            if key_set:
                for src in src_by_id.values():
                    haystack = (src.get("title", "") + " " + src.get("url", "")).lower()
                    if not any(k in haystack for k in key_set):
                        continue
                    _add_candidates(src)

            # Shortlist top sources (fallback B folded in)
            for sid, _rank in shortlist_top:
                _add_candidates(src_by_id.get(sid))

            picked: dict | None = None
            picked_source: str | None = None
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                best_score, best_img, best_src = candidates[0]
                if best_score >= threshold:
                    picked, picked_source = best_img, best_src
                    used_urls.add(picked["url"])

            if not picked and generate_ai and run_dir is not None:
                ai_idx += 1
                prompt = (
                    f"Editorial illustration for a technical document. "
                    f"Subject: {s_title}. Context: {c_title}. "
                    f"Style: clean, professional, schematic, neutral background, "
                    f"no text overlay. Photographic or vector flat illustration."
                )
                try:
                    local = await self._generate_ai_image(prompt, run_dir, ai_idx)
                except Exception as e:
                    # Image generation must NEVER kill the pipeline. We've
                    # already produced hours of valuable text content; an
                    # illustration glitch is cosmetic. Log and continue.
                    log.warning("AI image generation failed; section gets no image",
                                extra={"error": str(e), "section": s_title[:80]})
                    local = None
                if local:
                    rel = Path(local).relative_to(run_dir)
                    picked = {"url": str(rel), "alt": s_title}
                    picked_source = "ai-generated"

            if not picked:
                continue

            caption = picked.get("alt") or s_title
            attrib = f" — Source: {picked_source}" if picked_source else ""
            md_img = f"\n\n![{caption}]({picked['url']})\n*{caption}{attrib}*\n\n"
            # Insert after the first paragraph (after first blank line) or at start
            parts = content.split("\n\n", 1)
            if len(parts) == 2:
                sec["content"] = parts[0] + md_img + parts[1]
            else:
                sec["content"] = content + md_img

        if run_dir is not None:
            (run_dir / "sections.json").write_text(
                json.dumps(sections_data, ensure_ascii=False), encoding="utf-8")
        return sections_data

    # ─────────────────────────────────────────────
    # EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────

    async def generate_executive_summary(self, report_md: str, question: str, llm_logs: list[dict[str, Any]], language: str = "fr") -> str:
        """Generate a compelling executive summary from the full report."""
        lang_name = {"fr": "Francais", "en": "English", "ru": "Russkiy"}.get(language, "Francais")

        # Take first 15K chars of report for context (covers most key content)
        report_excerpt = report_md[:15000]

        system = (
            f"Tu es un redacteur executif senior. {_date_instruction()} Langue: {lang_name}.\n"
            f"A partir du rapport de recherche fourni, redige un RESUME EXECUTIF percutant.\n\n"
            f"Structure obligatoire:\n"
            f"1. **Contexte et enjeux** (2-3 phrases): Pourquoi ce sujet est critique\n"
            f"2. **Resultats cles** (3-5 bullet points): Les decouvertes les plus importantes\n"
            f"3. **Donnees marquantes** (2-3 chiffres/faits saillants avec sources)\n"
            f"4. **Recommandations** (2-3 actions concretes)\n"
            f"5. **Perspectives** (1-2 phrases): Tendances futures\n\n"
            f"Style: Concis, factuel, orienté decision. Maximum 500 mots."
        )
        prompt = f"Question de recherche: {question}\n\nRapport (extrait):\n{report_excerpt}"

        return await self.llm.ask(self.config.writer_model, system, prompt, llm_logs, "executive_summary", temperature=0.5)

    # ─────────────────────────────────────────────
    # REPORT ASSEMBLY
    # ─────────────────────────────────────────────

    async def assemble_report(self, planner: dict[str, Any], sections_payload: dict[str, Any], claims: list[dict[str, Any]], verdicts: dict[str, Any], corpus: dict[str, Any], executive_summary: str = "") -> tuple[str, str]:
        title = planner.get("question_reformulated", "Dossier")
        sections = sections_payload.get("sections", [])
        toc = ["## Table des Matieres\n"]
        current_c = "__START__"
        for s in sections:
            if s["c_title"] != current_c: current_c = s["c_title"]; toc.append(f"- [{current_c}](#{markdown_anchor(current_c)})")
            toc.append(f"  - [{s['s_title']}](#{markdown_anchor(s['s_title'])})")
        body = [f"# {title}\n", "\n".join(toc), "\n---\n"]

        # Insert executive summary if provided
        if executive_summary:
            body.append(f"\n## Resume Executif\n\n{executive_summary}\n\n---\n")

        current_c = "__START__"
        for s in sections:
            if s["c_title"] != current_c:
                current_c = s["c_title"]
                body.append(f"\n## {current_c}\n<a name='{markdown_anchor(current_c)}'></a>")
            content = re.sub(r"\[CLM-([a-f0-9-]+)\]", r"[[*]](annexes.md#CLM-\1)", s["content"])
            body.append(f"\n### {s['s_title']}\n<a name='{markdown_anchor(s['s_title'])}'></a>\n\n{content}\n")

        # ── ANNEXES with bibliography ──
        annex = ["# Annexes Techniques\n"]
        v_dict = {v["claim_id"]: v for v in verdicts.get("verdicts", [])}
        src_dict = {s["source_id"]: s for s in corpus.get("sources", [])}

        # Group claims by source for cleaner bibliography
        claims_by_source = {}
        for c in claims:
            sid = c.get("source_id", "unknown")
            claims_by_source.setdefault(sid, []).append(c)

        for c in claims:
            cid = c["claim_id"]
            v = v_dict.get(cid, {})
            src = src_dict.get(c.get("source_id"), {})
            claim_text = c.get("claim_text") or c.get("claim_product_image") or "N/A"
            status_badge = {"ACCEPTED": "VERIFIE", "REJECTED": "REJETE", "UNCERTAIN": "INCERTAIN"}.get(v.get("status", ""), "?")
            annex.append(
                f"<a name='CLM-{cid}'></a>\n### Preuve {cid}\n"
                f"- **Fait :** {claim_text}\n"
                f"- **Statut :** {status_badge}\n"
                f"- **Source :** [{src.get('title') or 'Lien'}]({src.get('canonical_url', '#')})\n\n---\n"
            )

        # Add bibliography section
        annex.append("\n## Bibliographie\n\n")
        seen_sources = set()
        for idx, src in enumerate(corpus.get("sources", []), 1):
            sid = src.get("source_id", "")
            if sid in seen_sources:
                continue
            seen_sources.add(sid)
            annex.append(f"{idx}. {src.get('title', 'Sans titre')}. *{src.get('domain', '')}*. URL: {src.get('canonical_url', src.get('url', '#'))}\n")

        return "\n".join(body), "\n".join(annex)