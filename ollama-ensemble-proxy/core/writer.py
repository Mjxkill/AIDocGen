import json
import time
import re
from pathlib import Path
from typing import Any, Callable
from .config import DossierConfig
from .llm import LLMClient
from .utils import emit_progress, markdown_anchor

class Writer:
    def __init__(self, config: DossierConfig, llm: LLMClient):
        self.config = config
        self.llm = llm

    def parse_markdown_outline(self, text: str) -> list[dict[str, Any]]:
        """Parse Markdown outline with 3 levels: Parties → Chapters → Sections."""
        result = []
        lines = text.split('\n')
        
        current_party: dict[str, Any] | None = None
        current_chapter: dict[str, Any] | None = None
        
        party_regex = r'^#{1,2}\s*(?:Partie|Part)\s*([IVX\d]+)[\s\:\-–—]*\s*(.*?)(?:\s*#|$)'
        chap_regex = r'^#{1,4}\s*(?:Chapitre|Chapter)\s*[\d\.]*\s*[\:\-\.\s—]*\s*(.*?)(?:\s*#|$)'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect Partie Header
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
            
            # Detect Chapter Header
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
            
            # Detect Section
            sec_match = re.search(r'^[*\-]\s*\*{0,2}(?:[\d\.]+)?\s*(.*?)\*{0,2}\s*$', line)
            if sec_match and current_chapter is not None:
                sec_title = sec_match.group(1).strip()
                if len(sec_title) > 3:
                    current_chapter.setdefault('sub_sections', []).append({"title": sec_title, "brief": "Analyse détaillée."})
        
        # Don't forget the last chapter
        if current_chapter and current_chapter.get('sub_sections'):
            if current_party:
                current_party.setdefault('chapters', []).append(current_chapter)
            else:
                result.append(current_chapter)
        
        # Filter
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

    def generate_sub_questions_from_outline(self, outline: list[dict[str, Any]], main_question: str, tags: list[str] = None, max_questions: int = 35) -> list[dict[str, str]]:
        """Generate search sub-questions from the outline sections."""
        sub_questions = []
        sq_id = 1
        
        # Extract main topic from question
        main_topic = self._extract_main_topic(main_question)
        
        # Build tag string for queries
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
        """Extract the main topic/product/entity from a question."""
        question = re.sub(r'(?i)^(fais|fait|faire|écris|écrire|rédige|rédiger|crée|créer|donne|donner)\s+(moi\s+)?(un\s+|une\s+)?(dossier|rapport|document|étude|article)\s+(complet|détaillé|completa?)?\s*(sur|de|about)?\s*', '', question)
        
        stop_words = {"le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "mais", "sur", "dans", "pour", "avec", 
                      "the", "a", "an", "and", "or", "but", "for", "with", "about", "of", "in", "to", "on"}
        
        words = re.findall(r'\b[A-ZÀ-ÿ][a-zA-ZÀ-ÿ]*\b|\b[a-zA-ZÀ-ÿ]{4,}\b', question)
        keywords = [w for w in words if w.lower() not in stop_words]
        
        proper_nouns = [w for w in words if w[0].isupper() and w.lower() not in stop_words]
        other_terms = [w for w in keywords if w not in proper_nouns]
        
        result = proper_nouns[:3] + other_terms[:2]
        return " ".join(result)

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract key terms from a title."""
        stop_words = {"le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "mais", "sur", "dans", "pour", "avec", "etc",
                      "the", "a", "an", "and", "or", "but", "for", "with", "about", "of", "in", "to", "on", "as"}
        
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', text.lower())
        return [w for w in words if w not in stop_words][:6]

    def _build_search_query(self, section_title: str, context_keywords: list[str], main_topic: str, tags: list[str] = None) -> str:
        """Build an optimized search query from section context."""
        section_keywords = self._extract_keywords(section_title)
        
        query_parts = section_keywords[:4]
        
        for kw in context_keywords[:2]:
            if kw not in query_parts:
                query_parts.append(kw)
        
        main_terms = main_topic.split()
        for term in main_terms[:2]:
            if term.lower() not in [p.lower() for p in query_parts]:
                query_parts.append(term)
        
        # Add tags to query (most important for filtering)
        if tags:
            for tag in tags[:2]:
                tag_clean = re.sub(r'[^a-zA-Z0-9]', '', tag)
                if tag_clean.lower() not in [p.lower() for p in query_parts]:
                    query_parts.append(tag_clean)
        
        return " ".join(query_parts)

    async def plan_dossier(self, question: str, detail_level: str, llm_logs: list[dict[str, Any]], prompt_type: str = "generic", language: str = "fr", presearch_results: dict[str, Any] | None = None, run_dir: Path | None = None, coder_model_override: str | None = None, tags: list[str] = None) -> dict[str, Any]:
        lang_name = {"fr": "Français", "en": "English", "es": "Español", "de": "Deutsch"}.get(language, "Français")
        debug_info = {"attempts": [], "planner_prompt": {"system": "", "user": ""}}
        
        # Build tag instruction
        tag_instruction = ""
        if tags:
            tag_str = ", ".join([f"#{t}" for t in tags])
            tag_instruction = f"\n\nIMPORTANT: Ce dossier DOIT se concentrer exclusivement sur les sujets liés aux tags suivants: {tag_str}. Toute section ou information hors de ce scope doit être évitée."
        
        if presearch_results and len(presearch_results) > 0:
            def sanitize(t): return re.sub(r'http[s]?://\S+', '', str(t or ""))
            web_context = "\n".join([f"- {sanitize(l.get('title'))}: {sanitize(l.get('snippet'))}" for l in presearch_results[:15]])
        else: web_context = "Pas de données web."

        # STEP 1: RICH DRAFT
        draft_sys = f"Tu es un Expert Scientifique. Mission: plan d'enquête MASSAL (15-20 chapitres) en {lang_name}.{tag_instruction}"
        draft_user = f"Sujet: {question}\nWEB:\n{web_context}\n\nRéfléchis (Thinking) puis donne le sommaire détaillé en Markdown après '---'."
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
            raw_json = await self.llm.ask(coder_model, "Ingénieur JSON.", f"Convertis en JSON master_outline :\n{draft_md}", llm_logs, "planner_crystallize", temperature=0.1)
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
        raise RuntimeError("Échec total de l'extraction du plan.")

    async def write_sections(self, planner: dict[str, Any], claims: list[dict[str, Any]], llm_logs: list[dict[str, Any]], progress_cb: Callable | None, run_dir: Path | None, language: str = "fr", tags: list[str] = None) -> dict[str, Any]:
        outline = planner.get("master_outline", [])
        lang_name = {"fr": "Français", "en": "English", "es": "Español", "de": "Deutsch"}.get(language, "Français")
        tasks = []
        
        # Build tag instruction for writer
        tag_instruction = ""
        if tags:
            tag_str = ", ".join([f"#{t}" for t in tags])
            tag_instruction = f"\nIMPORTANT: Reste STRICTEMENT focalisé sur les tags: {tag_str}. Ignore tout contenu hors sujet."
        
        # Handle both 2-level and 3-level structures
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
        
        sections_content = []
        for idx, (p_title, c_title, sec) in enumerate(tasks, 1):
            s_title = sec.get("title")
            if progress_cb: await emit_progress(progress_cb, run_dir, "writing", f"Writing {idx}/{len(tasks)}: {s_title}")
            keywords = s_title.lower().split()
            relevant = [c for c in claims if any(k in c.get("claim_text", "").lower() for k in keywords[:3])]
            context = json.dumps(relevant[:30], ensure_ascii=False)
            prompt = f"Dissertation Section: {s_title}\nPartie: {p_title}\nChapitre: {c_title}\nBrief: {sec.get('brief')}\nContext: {context}\n\nRules: Technical depth, min 1000 words, cite [CLM-id]. LANGUAGE: {lang_name}.{tag_instruction}"
            content = await self.llm.ask(self.config.writer_model, f"Academic {lang_name} Writer", prompt, llm_logs, "writing")
            sections_content.append({"type": "section", "p_title": p_title, "c_title": c_title, "s_title": s_title, "content": content})
        return {"sections": sections_content}

    async def assemble_report(self, planner: dict[str, Any], sections_payload: dict[str, Any], claims: list[dict[str, Any]], verdicts: dict[str, Any], corpus: dict[str, Any]) -> tuple[str, str]:
        title = planner.get("question_reformulated", "Dossier")
        sections = sections_payload.get("sections", [])
        toc = ["## Table des Matières\n"]
        current_c = "__START__"
        for s in sections:
            if s["c_title"] != current_c: current_c = s["c_title"]; toc.append(f"- [{current_c}](#{markdown_anchor(current_c)})")
            toc.append(f"  - [{s['s_title']}](#{markdown_anchor(s['s_title'])})")
        body = [f"# {title}\n", "\n".join(toc), "\n---\n"]
        current_c = "__START__"
        for s in sections:
            if s["c_title"] != current_c:
                current_c = s["c_title"]
                body.append(f"\n## {current_c}\n<a name='{markdown_anchor(current_c)}'></a>")
            content = re.sub(r"\[CLM-([a-f0-9-]+)\]", r"[[*]](annexes.md#CLM-\1)", s["content"])
            body.append(f"\n### {s['s_title']}\n<a name='{markdown_anchor(s['s_title'])}'></a>\n\n{content}\n")
        annex = ["# Annexes Techniques\n"]
        v_dict = {v["claim_id"]: v for v in verdicts.get("verdicts", [])}
        src_dict = {s["source_id"]: s for s in corpus.get("sources", [])}
        for c in claims:
            cid = c["claim_id"]
            v = v_dict.get(cid, {})
            src = src_dict.get(c.get("source_id"), {})
            annex.append(f"<a name='CLM-{cid}'></a>\n### Preuve {cid}\n- **Fait :** {c['claim_text']}\n- **Source :** [{src.get('title') or 'Lien'}]({src.get('canonical_url', '#')})\n\n---\n")
        return "\n".join(body), "\n".join(annex)
