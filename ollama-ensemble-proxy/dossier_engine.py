import asyncio
import hashlib
import io
import json
import os
import re
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from pypdf import PdfReader

from dossier_config import DossierConfig, env_bool

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]



class DossierEngine:
    def __init__(self, base_url: str, config: DossierConfig):
        self.base_url = base_url.rstrip("/")
        self.config = config
        self._run_locks: dict[str, asyncio.Lock] = {}

    async def run(
        self,
        question: str,
        run_id: str | None = None,
        resume: bool = True,
        progress_cb: ProgressCallback | None = None,
        prompt_type: str | None = None,
        detail_level: str = "medium",
    ) -> dict[str, Any]:
        cleaned_question = " ".join(question.split()).strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty")

        # Apply Detail Level Config
        if detail_level == "synthetic":
            self.config.writer_target_words_per_section = 800
            self.config.planner_book_page_chars = 2500
        elif detail_level == "dissertation":
            self.config.writer_target_words_per_section = 4000
            self.config.planner_book_page_chars = 12000
        else: # medium
            self.config.writer_target_words_per_section = 2200
            self.config.planner_book_page_chars = 7000

        # Load Prompt Template
        planner_prompt_content = None
        if prompt_type:
            prompt_path = Path(__file__).parent / "prompts" / f"planner_{prompt_type}.txt"
            if prompt_path.exists():
                planner_prompt_content = prompt_path.read_text(encoding="utf-8")
            else:
                print(f"Warning: Prompt template '{prompt_type}' not found at {prompt_path}")
        
        # Append dissertation instruction if needed
        if detail_level == "dissertation" and planner_prompt_content:
             planner_prompt_content += "\n\nCRITIQUE: PRODUIS UNE 'DISSERTATION' DE TRES GRANDE ENVERGURE (12-18 CHAPITRES). SOIS EXHAUSTIF."
        if detail_level == "synthetic" and planner_prompt_content:
             planner_prompt_content += "\n\nCRITIQUE: PRODUIS UN RAPPORT SYNTHETIQUE ET CONCIS (3-5 CHAPITRES). VA A L'ESSENTIEL."

        run_id = run_id or self._new_run_id(cleaned_question)
        run_dir = Path(self.config.data_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "raw").mkdir(parents=True, exist_ok=True)
        (run_dir / "clean").mkdir(parents=True, exist_ok=True)

        lock = self._run_locks.setdefault(run_id, asyncio.Lock())
        if lock.locked():
            raise RuntimeError(f"Run {run_id} is already executing")

        async with lock:
            started_at = int(time.time())
            status = self._load_status(run_dir)
            if status and status.get("state") == "completed" and resume:
                report_path = run_dir / "report.md"
                audit_path = run_dir / "audit.json"
                report_markdown = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
                await self._emit(
                    progress_cb,
                    run_dir,
                    "resume",
                    "Run completed found on disk, returning cached dossier.",
                )
                return {
                    "run_id": run_id,
                    "report_markdown": report_markdown,
                    "report_path": str(report_path),
                    "audit_path": str(audit_path),
                    "cached": True,
                }

            self._write_status(
                run_dir,
                {
                    "run_id": run_id,
                    "state": "running",
                    "started_at": status.get("started_at", started_at) if status else started_at,
                    "updated_at": started_at,
                    "question": cleaned_question,
                    "stage": "init",
                    "events": status.get("events", []) if status else [],
                    "error": None,
                },
            )
            await self._emit(progress_cb, run_dir, "init", "Run initialized", question=cleaned_question)

            llm_logs: list[dict[str, Any]] = []

            try:
                presearch = await self._stage_json(
                    run_dir=run_dir,
                    filename="presearch.json",
                    stage="presearch",
                    resume=resume,
                    progress_cb=progress_cb,
                    task_factory=lambda: self._presearch_question(cleaned_question, llm_logs, progress_cb, run_dir),
                )

                planner = await self._stage_json(
                    run_dir=run_dir,
                    filename="planner.json",
                    stage="planner",
                    resume=resume,
                    progress_cb=progress_cb,
                    task_factory=lambda: self._plan_question(cleaned_question, presearch, llm_logs, prompt_override=planner_prompt_content),
                )

                # --- HUMAN VALIDATION PAUSE ---
                validation_flag = run_dir / "validated.txt"
                if not validation_flag.exists():
                    await self._emit(progress_cb, run_dir, "awaiting_validation", "Plan generated. Awaiting human approval.")
                    return
                # ------------------------------

                search_results = await self._stage_json(
                    run_dir=run_dir,
                    filename="search_results.json",
                    stage="search",
                    resume=resume,
                    progress_cb=progress_cb,
                    task_factory=lambda: self._search_subquestions(planner, llm_logs, progress_cb, run_dir),
                )

                outline_seed = await self._stage_json(
                    run_dir=run_dir,
                    filename="outline_seed.json",
                    stage="outline",
                    resume=resume,
                    progress_cb=progress_cb,
                    task_factory=lambda: self._build_outline_from_search(planner, search_results, llm_logs),
                )

                corpus = await self._stage_json(
                    run_dir=run_dir,
                    filename="corpus.json",
                    stage="corpus",
                    resume=resume,
                    progress_cb=progress_cb,
                    task_factory=lambda: self._build_corpus(search_results, progress_cb, run_dir),
                )

                shortlist = await self._stage_json(
                    run_dir=run_dir,
                    filename="shortlist.json",
                    stage="ranking",
                    resume=resume,
                    progress_cb=progress_cb,
                    task_factory=lambda: self._rank_and_shortlist(planner, search_results, corpus, llm_logs, progress_cb, run_dir),
                )

                claims = await self._stage_json(
                    run_dir=run_dir,
                    filename="claims.json",
                    stage="claims",
                    resume=resume,
                    progress_cb=progress_cb,
                    task_factory=lambda: self._extract_claims(planner, shortlist, corpus, llm_logs, progress_cb, run_dir),
                )

                verdicts = await self._stage_json(
                    run_dir=run_dir,
                    filename="verdicts.json",
                    stage="verification",
                    resume=resume,
                    progress_cb=progress_cb,
                    task_factory=lambda: self._verify_claims(planner, claims, llm_logs, progress_cb, run_dir),
                )

                sections = await self._stage_json(
                    run_dir=run_dir,
                    filename="sections.json",
                    stage="writing",
                    resume=resume,
                    progress_cb=progress_cb,
                    task_factory=lambda: self._write_sections(
                        planner,
                        outline_seed,
                        corpus,
                        claims,
                        verdicts,
                        llm_logs,
                        progress_cb,
                        run_dir,
                    ),
                )

                report_markdown = await self._assemble_report(
                    planner,
                    outline_seed,
                    corpus,
                    claims,
                    verdicts,
                    sections,
                    llm_logs,
                )
                report_path = run_dir / "report.md"
                report_path.write_text(report_markdown, encoding="utf-8")

                audit_payload = {
                    "run_id": run_id,
                    "question": cleaned_question,
                    "started_at": self._load_status(run_dir).get("started_at", started_at),
                    "finished_at": int(time.time()),
                    "config": asdict(self.config),
                    "presearch": presearch,
                    "planner": planner,
                    "search_results": search_results,
                    "outline_seed": outline_seed,
                    "corpus": corpus,
                    "shortlist": shortlist,
                    "claims": claims,
                    "verdicts": verdicts,
                    "sections": sections,
                    "llm_logs": llm_logs,
                }
                audit_path = run_dir / "audit.json"
                audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")

                status = self._load_status(run_dir)
                status["state"] = "completed"
                status["stage"] = "completed"
                status["updated_at"] = int(time.time())
                status["report_path"] = str(report_path)
                status["audit_path"] = str(audit_path)
                self._write_status(run_dir, status)
                await self._emit(progress_cb, run_dir, "completed", "Dossier completed", run_id=run_id)

                return {
                    "run_id": run_id,
                    "report_markdown": report_markdown,
                    "report_path": str(report_path),
                    "audit_path": str(audit_path),
                    "cached": False,
                }
            except Exception as exc:  # noqa: BLE001
                detail = str(exc).strip() or repr(exc)
                status = self._load_status(run_dir)
                status["state"] = "failed"
                status["stage"] = "failed"
                status["updated_at"] = int(time.time())
                status["error"] = detail
                self._write_status(run_dir, status)
                await self._emit(progress_cb, run_dir, "failed", f"Run failed: {detail}")
                raise

    def get_status(self, run_id: str) -> dict[str, Any] | None:
        run_dir = Path(self.config.data_dir) / run_id
        status = self._load_status(run_dir)
        if not status:
            return None
        return status

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        root = Path(self.config.data_dir)
        if not root.exists():
            return []

        statuses: list[dict[str, Any]] = []
        for run_dir in root.iterdir():
            if not run_dir.is_dir():
                continue
            status = self._load_status(run_dir)
            if status:
                statuses.append(status)

        statuses.sort(key=lambda item: int(item.get("updated_at", 0)), reverse=True)
        return statuses[:limit]

    def get_report(self, run_id: str) -> str | None:
        report_path = Path(self.config.data_dir) / run_id / "report.md"
        if not report_path.exists():
            return None
        return report_path.read_text(encoding="utf-8")

    def get_audit(self, run_id: str) -> dict[str, Any] | None:
        audit_path = Path(self.config.data_dir) / run_id / "audit.json"
        if not audit_path.exists():
            return None
        return json.loads(audit_path.read_text(encoding="utf-8"))

    async def _stage_json(
        self,
        run_dir: Path,
        filename: str,
        stage: str,
        resume: bool,
        progress_cb: ProgressCallback | None,
        task_factory: Callable[[], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        stage_path = run_dir / filename
        status = self._load_status(run_dir)
        status["stage"] = stage
        status["updated_at"] = int(time.time())
        self._write_status(run_dir, status)

        if resume and stage_path.exists():
            await self._emit(progress_cb, run_dir, stage, f"Reusing checkpoint {filename}")
            return json.loads(stage_path.read_text(encoding="utf-8"))

        await self._emit(progress_cb, run_dir, stage, f"Starting stage {stage}")
        payload = await task_factory()
        stage_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        await self._emit(progress_cb, run_dir, stage, f"Stage {stage} completed")
        return payload

    async def _presearch_question(
        self,
        question: str,
        llm_logs: list[dict[str, Any]],
        progress_cb: ProgressCallback | None,
        run_dir: Path,
    ) -> dict[str, Any]:
        queries = self._generate_query_variants(question, self.config.presearch_query_variants)

        # Determine Search Engine
        searx_available = False
        engine_choice = self.config.web_search_engine
        
        if engine_choice == "searxng":
            searx_available = True # Force try
        elif engine_choice == "duckduckgo":
            searx_available = False
        else: # auto
            searx_available = await self._probe_searxng()

        engine = "searxng" if searx_available else "duckduckgo"
        
        await self._emit(
            progress_cb,
            run_dir,
            "presearch",
            f"Topic presearch using {engine} with {len(queries)} queries",
        )

        links: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        query_errors: list[dict[str, str]] = []
        for query in queries:
            if len(links) >= self.config.presearch_max_links:
                break
            
            # Anti-ban Delay
            if self.config.web_request_delay > 0:
                await asyncio.sleep(self.config.web_request_delay)

            try:
                if searx_available:
                    hits = await self._searx_search(
                        query=query,
                        max_results=self.config.presearch_per_query_results,
                    )
                else:
                    hits = await self._ddg_search(
                        query=query,
                        max_results=self.config.presearch_per_query_results,
                    )
            except Exception as exc:  # noqa: BLE001
                query_errors.append({"query": query, "error": str(exc) or repr(exc)})
                continue

            for hit in hits:
                canonical_url = self._canonicalize_url(str(hit.get("url") or ""))
                if not canonical_url or canonical_url in seen_urls:
                    continue
                seen_urls.add(canonical_url)
                links.append(
                    {
                        "id": f"PRE-{uuid.uuid4().hex[:10]}",
                        "query": query,
                        "title": str(hit.get("title") or "").strip(),
                        "url": canonical_url,
                        "domain": self._extract_domain(canonical_url),
                        "snippet": str(hit.get("snippet") or "").strip(),
                        "published_at": str(hit.get("published_at") or "").strip(),
                    }
                )
                if len(links) >= self.config.presearch_max_links:
                    break

        if not links:
            error_summary = "; ".join(f"{item['query']}: {item['error']}" for item in query_errors[:6]) or "none"
            raise RuntimeError(
                f"Presearch returned no links for question '{question}'. Query errors: {error_summary}"
            )

        warnings: list[str] = []
        filtered_links: list[dict[str, Any]] = []
        rejected_links: list[dict[str, Any]] = []
        strong_anchors = self._build_question_anchor_terms(question)
        for link in links:
            decision = self._evaluate_presearch_link(
                question=question,
                strong_anchors=strong_anchors,
                queries=queries,
                link=link,
            )
            if decision.get("keep"):
                enriched = dict(link)
                enriched["presearch_score"] = round(float(decision.get("score", 0.0)), 4)
                enriched["presearch_signals"] = decision.get("signals", {})
                filtered_links.append(enriched)
            else:
                rejected_links.append(
                    {
                        "url": link.get("url", ""),
                        "title": link.get("title", ""),
                        "reason": str(decision.get("reason") or "filtered"),
                    }
                )

        filtered_links.sort(
            key=lambda item: float(item.get("presearch_score", 0.0)),
            reverse=True,
        )

        minimum_links = max(20, min(50, self.config.presearch_max_links // 2))
        if len(filtered_links) < minimum_links:
            warnings.append(
                (
                    "presearch relevance filter retained few links "
                    f"({len(filtered_links)}/{len(links)}); using best available set."
                )
            )
        if not filtered_links:
            raise RuntimeError(
                (
                    f"Presearch relevance filter removed every link for question '{question}'. "
                    f"Anchors={sorted(list(strong_anchors))[:12]}"
                )
            )

        links = filtered_links[: self.config.presearch_max_links]

        synthesis_payload = {
            "question": question,
            "search_links": [
                {
                    "title": link.get("title", ""),
                    "url": link.get("url", ""),
                    "domain": link.get("domain", ""),
                    "snippet": link.get("snippet", ""),
                }
                for link in links[:120]
            ],
            "output_schema": {
                "topic_candidates": [
                    {
                        "title": "string",
                        "why_relevant": "string",
                        "priority": "high|medium|low",
                        "viewpoint": "benefits|limitations|both",
                    }
                ],
                "possible_out_of_scope": [{"title": "string", "reason": "string"}],
                "question_rewrite_hint": "string",
            },
        }
        synthesis_raw = await self._ask_text_model(
            model=self.config.judge_model,
            system_prompt=(
                "You are a topic cartographer. Build a first-pass subject map from search snippets. "
                "Return strict JSON only."
            ),
            user_prompt=json.dumps(synthesis_payload, ensure_ascii=False),
            llm_logs=llm_logs,
            stage="presearch",
        )
        parsed = await self._parse_or_repair_json(
            raw_text=synthesis_raw,
            model=self.config.judge_model,
            stage="presearch",
            llm_logs=llm_logs,
            expected_type="dict",
            schema_hint=synthesis_payload["output_schema"],
        )

        topic_candidates: list[dict[str, str]] = []
        seen_titles: set[str] = set()
        for item in parsed.get("topic_candidates", []) if isinstance(parsed.get("topic_candidates"), list) else []:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            key = title.casefold()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            priority = str(item.get("priority") or "medium").strip().lower()
            if priority not in {"high", "medium", "low"}:
                priority = "medium"
            viewpoint = str(item.get("viewpoint") or "both").strip().lower()
            if viewpoint not in {"benefits", "limitations", "both"}:
                viewpoint = "both"
            topic_candidates.append(
                {
                    "title": title,
                    "why_relevant": str(item.get("why_relevant") or "").strip(),
                    "priority": priority,
                    "viewpoint": viewpoint,
                }
            )

        possible_out_of_scope: list[dict[str, str]] = []
        seen_rejected: set[str] = set()
        for item in parsed.get("possible_out_of_scope", []) if isinstance(parsed.get("possible_out_of_scope"), list) else []:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            key = title.casefold()
            if key in seen_rejected:
                continue
            seen_rejected.add(key)
            possible_out_of_scope.append(
                {
                    "title": title,
                    "reason": str(item.get("reason") or "").strip(),
                }
            )

        if not topic_candidates:
            warnings.append("presearch produced no usable topic candidates")
            for link in links[:10]:
                title = str(link.get("title") or "").strip()
                if not title:
                    continue
                topic_candidates.append(
                    {
                        "title": title[:140],
                        "why_relevant": "Topic derive heuristiquement depuis les resultats de presearch.",
                        "priority": "medium",
                        "viewpoint": "both",
                    }
                )

        return {
            "question": question,
            "engine": engine,
            "queries": queries,
            "raw_link_count": len(filtered_links) + len(rejected_links),
            "links": links,
            "filtered_link_count": len(links),
            "rejected_link_count": len(rejected_links),
            "rejected_links": rejected_links[:300],
            "query_errors": query_errors,
            "topic_candidates": topic_candidates[:32],
            "possible_out_of_scope": possible_out_of_scope[:24],
            "question_rewrite_hint": str(parsed.get("question_rewrite_hint") or "").strip(),
            "warnings": warnings,
            "generated_at": int(time.time()),
        }

    async def _plan_question(
        self,
        question: str,
        presearch_payload: dict[str, Any],
        llm_logs: list[dict[str, Any]],
        prompt_override: str | None = None,
    ) -> dict[str, Any]:
        def _to_clean_list(raw: Any, max_items: int = 24) -> list[str]:
            items = raw if isinstance(raw, list) else [raw]
            out: list[str] = []
            seen: set[str] = set()
            for item in items:
                value = str(item or "").strip()
                if not value:
                    continue
                key = self._normalize_text_for_dedupe(value)
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(value)
                if max_items > 0 and len(out) >= max_items:
                    break
            return out

        planner_schema_hint = {
            "question_reformulated": "string",
            "ambiguities": ["string"],
            "scope_notes": "string",
            "inclusion_criteria": ["string"],
            "exclusion_criteria": ["string"],
            "master_outline": [
                {
                    "id": "CH1",
                    "title": "string",
                    "goal": "string",
                    "linked_sub_questions": ["SQ1"],
                    "status": "planned|rejected",
                    "reason": "string",
                    "sub_sections": ["string"],
                }
            ],
            "sub_questions": [
                {
                    "id": "SQ1",
                    "question": "string",
                    "proof_criteria": ["string"],
                    "search_queries": ["string"],
                }
            ],
        }
        planner_json_example = {
            "question_reformulated": "Topographie detaillee du sujet",
            "ambiguities": ["Portee temporelle precisee", "Terminologie a clarifier"],
            "scope_notes": "Le dossier couvre uniquement le sujet principal et ses sous-domaines directement relies.",
            "inclusion_criteria": [
                "Informations techniques verificables",
                "Configurations et exemples d'application utiles",
            ],
            "exclusion_criteria": [
                "Contenu hors sujet",
                "Affirmations sans preuve exploitable",
            ],
            "sub_questions": [
                {
                    "id": "SQ1",
                    "question": "Quels sont les composants et mecanismes majeurs ?",
                    "proof_criteria": [
                        "Sources techniques independantes convergentes",
                        "Definitions et mesures explicites",
                    ],
                    "search_queries": [
                        "sujet composants architecture",
                        "sujet mecanismes details techniques",
                    ],
                }
            ],
            "master_outline": [
                {
                    "id": "CH1",
                    "title": "Architecture et composants",
                    "goal": "Expliquer le fonctionnement et les blocs essentiels",
                    "linked_sub_questions": ["SQ1"],
                    "status": "planned",
                    "reason": "Chapitre fondamental pour comprendre le reste du dossier",
                    "sub_sections": [
                        "Vue d'ensemble et perimetre",
                        "Sous-systemes et interactions",
                        "Contraintes techniques et limites",
                    ],
                }
            ],
        }

        planner_warnings: list[str] = []
        book_models = self._planner_book_models()
        json_coder_model = self._planner_book_json_model()
        if len(book_models) < 3:
            raise RuntimeError(
                "Planner configuration invalid: 3 book models required "
                "(model_1, model_2, model_3)."
            )
        if not json_coder_model:
            raise RuntimeError(
                "Planner configuration invalid: JSON coder model required "
                "(ENSEMBLE_DOSSIER_BOOK_MODEL_4_JSON)."
            )

        raw_links = presearch_payload.get("links", []) if isinstance(presearch_payload.get("links"), list) else []
        candidates: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for link in raw_links:
            canonical_url = self._canonicalize_url(str(link.get("url") or ""))
            if not canonical_url or canonical_url in seen_urls:
                continue
            seen_urls.add(canonical_url)
            candidates.append(
                {
                    "url": canonical_url,
                    "title": str(link.get("title") or "").strip(),
                    "snippet": str(link.get("snippet") or "").strip(),
                    "domain": self._extract_domain(canonical_url),
                }
            )

        selected_links: list[dict[str, Any]] = []
        selected_urls: set[str] = set()
        domain_counts: dict[str, int] = {}
        domain_cap = max(1, min(4, self.config.planner_book_web_links // 4 or 1))

        for link in candidates:
            if len(selected_links) >= self.config.planner_book_web_links:
                break
            domain = str(link.get("domain") or "__unknown__").strip() or "__unknown__"
            if domain_counts.get(domain, 0) >= domain_cap:
                continue
            url = str(link.get("url") or "")
            if not url or url in selected_urls:
                continue
            selected_urls.add(url)
            selected_links.append(link)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        if len(selected_links) < self.config.planner_book_web_links:
            for link in candidates:
                if len(selected_links) >= self.config.planner_book_web_links:
                    break
                url = str(link.get("url") or "")
                if not url or url in selected_urls:
                    continue
                selected_urls.add(url)
                selected_links.append(link)

        if not selected_links:
            raise RuntimeError("Planner cannot start: presearch returned no usable links")

        web_notes, web_note_warnings = await self._collect_planner_web_notes(
            question=question,
            selected_links=selected_links,
            models=book_models,
            llm_logs=llm_logs,
        )
        planner_warnings.extend(web_note_warnings)
        if not web_notes:
            raise RuntimeError(
                "Planner web extraction failed: no usable notes generated from the selected links"
            )

        web_notes.sort(key=lambda item: float(item.get("relevance_score", 0.0)), reverse=True)
        aggregated_topics = self._aggregate_planner_topics(web_notes, max_items=24)
        aggregated_facts = self._aggregate_planner_facts(web_notes, max_items=120)
        if not aggregated_topics:
            aggregated_topics = self._collect_presearch_topic_titles(presearch_payload, max_items=16)

        web_digest = [
            {
                "rank": idx,
                "url": note.get("url", ""),
                "title": note.get("title", ""),
                "domain": note.get("domain", ""),
                "model": note.get("model", ""),
                "relevance_score": note.get("relevance_score", 0.0),
                "key_topics": note.get("key_topics", [])[:8],
                "key_facts": note.get("key_facts", [])[:10],
                "components": note.get("components", [])[:8],
                "software_stack": note.get("software_stack", [])[:8],
                "configuration_examples": note.get("configuration_examples", [])[:8],
                "limitations": note.get("limitations", [])[:8],
            }
            for idx, note in enumerate(web_notes[: self.config.planner_book_web_links], start=1)
        ]

        stage1_payload = {
            "question": question,
            "instruction": (
                "Si tu devais ecrire un livre sur ce sujet, propose un sommaire detaille et approfondi "
                "avec chapitres et sous-chapitres en francais. "
                "Le sommaire doit etre exploitable pour une redaction longue, technique et structuree."
            ),
            "web_info": web_digest,
            "aggregated_topics": aggregated_topics,
            "aggregated_facts": aggregated_facts,
            "constraints": {
                "language": "fr",
                "cover_positive_and_negative_views": True,
                "cover_implementation_and_configuration": True,
                "cover_examples_and_limits": True,
                "avoid_generic_titles": True,
                "do_not_force_fixed_chapter_count": True,
            },
        }
        
        # Determine Planner Prompt (Stage 1)
        if prompt_override:
            s1_prompt = prompt_override
        else:
            p1_path = Path(__file__).parent / "prompts" / "planner_book_1.txt"
            s1_prompt = p1_path.read_text(encoding="utf-8").strip() if p1_path.exists() else "Create a comprehensive book outline."

        draft_1 = await self._ask_text_model(
            model=book_models[0],
            system_prompt=s1_prompt,
            user_prompt=json.dumps(stage1_payload, ensure_ascii=False),
            llm_logs=llm_logs,
            stage="planner-book-1",
        )

        stage2_payload = {
            "question": question,
            "instruction": (
                "Complete et ameliore le sommaire precedent, corrige les trous, "
                "ajoute les chapitres manquants, garde seulement les sujets pertinents. "
                "Ne supprime pas des sous-chapitres utiles deja presents: conserve-les et enrichis."
            ),
            "previous_outline": draft_1,
            "web_info": web_digest,
            "aggregated_topics": aggregated_topics,
            "aggregated_facts": aggregated_facts,
            "constraints": {
                "language": "fr",
                "must_include_tradeoffs": True,
                "must_include_operational_details": True,
                "must_include_contradictions_or_uncertainties": True,
                "avoid_raw_link_titles_as_chapters": True,
                "do_not_compress_or_merge_chapters_without_strong_reason": True,
            },
        }
        p2_path = Path(__file__).parent / "prompts" / "planner_book_2.txt"
        s2_prompt = p2_path.read_text(encoding="utf-8").strip() if p2_path.exists() else "Refine and complete the book outline."
        
        draft_2 = await self._ask_text_model(
            model=book_models[1],
            system_prompt=s2_prompt,
            user_prompt=json.dumps(stage2_payload, ensure_ascii=False),
            llm_logs=llm_logs,
            stage="planner-book-2",
        )

        stage3_payload = {
            "question": question,
            "web_info": web_digest,
            "aggregated_topics": aggregated_topics,
            "aggregated_facts": aggregated_facts,
            "draft_outline_1": draft_1,
            "draft_outline_2": draft_2,
            "constraints": {
                "language": "fr",
                "no_raw_urls_as_titles": True,
                "no_verbatim_user_prompt_as_chapter_title": True,
                "chapter_titles_must_be_specific": True,
                "keep_relevant_structure_from_drafts": True,
                "do_not_shorten_outline_arbitrarily": True,
                "preserve_maximum_subsection_granularity_from_drafts": True,
                "preserve_all_distinct_subsections_from_drafts": True,
            },
        }
        draft_3 = await self._ask_text_model(
            model=book_models[2],
            system_prompt=(
                "Tu es le redacteur en chef. "
                "Tu prends les brouillons des auteurs et tu livres le sommaire editorial final en texte. "
                "Ne retourne pas de JSON. "
                "Retourne uniquement un texte structure en francais, detaille et coherent. "
                "Produis un sommaire editorial de type livre: chapitres et sous-chapitres riches, precis et exploitables. "
                "Conserve les sous-chapitres distincts provenant des brouillons, et ajoute uniquement ceux manquants. "
                "N'uniformise pas artificiellement les chapitres avec un petit nombre de sous-sections. "
                "Le texte doit aussi expliciter les elements necessaires au schema final "
                "(question reformulee, ambiguities, scope_notes, inclusion/exclusion criteria, sous-questions et plan maitre)."
            ),
            user_prompt=json.dumps(stage3_payload, ensure_ascii=False),
            llm_logs=llm_logs,
            stage="planner-book-3",
        )

        stage4_payload = {
            "question": question,
            "web_info": web_digest,
            "aggregated_topics": aggregated_topics,
            "aggregated_facts": aggregated_facts,
            "draft_outline_final": draft_3,
            "required_schema": planner_schema_hint,
            "json_example": planner_json_example,
            "constraints": {
                "language": "fr",
                "strict_json_only": True,
                "do_not_invent_content": True,
                "convert_existing_outline_only": True,
                "preserve_semantic_granularity": True,
                "preserve_subsections_from_final_draft": True,
                "all_required_fields_must_be_present_and_non_empty": True,
                "status_values_allowed": ["planned", "rejected"],
            },
        }
        final_json_raw = await self._ask_text_model(
            model=json_coder_model,
            system_prompt=(
                "Tu es un AI coder specialise en structuration JSON. "
                "Tu convertis le sommaire editorial en JSON strict conforme au schema. "
                "N'invente aucun chapitre, aucune sous-question, aucune sous-section. "
                "Aucun texte hors JSON. "
                "Toutes les cles requises doivent etre presentes. "
                "Tous les champs textuels doivent etre non vides. "
                "Les statuts de chapitre doivent etre uniquement 'planned' ou 'rejected'. "
                "Construis le JSON uniquement a partir de draft_outline_final. "
                "Preserve les sous-sections telles qu'elles apparaissent dans draft_outline_final. "
                "Utilise json_example comme contrat explicite de forme et de structure (cles, types, listes non vides)."
            ),
            user_prompt=json.dumps(stage4_payload, ensure_ascii=False),
            llm_logs=llm_logs,
            stage="planner-book-4-json",
        )
        if self.config.strict_no_fallback:
            parsed = self._parse_json_payload(final_json_raw)
            if not isinstance(parsed, dict):
                raise RuntimeError(
                    "Planner strict/no-fallback: planner-book-4-json returned invalid JSON object. "
                    f"Excerpt: {final_json_raw[:320]}"
                )
        else:
            parsed = await self._parse_or_repair_json(
                raw_text=final_json_raw,
                model=json_coder_model,
                stage="planner-book-4-json",
                llm_logs=llm_logs,
                expected_type="dict",
                schema_hint=planner_schema_hint,
            )

        if self.config.strict_no_fallback:
            return self._validate_planner_output_strict(
                parsed=parsed,
                question=question,
                book_models=book_models,
                json_coder_model=json_coder_model,
                links_requested=self.config.planner_book_web_links,
                links_used=len(selected_links),
                web_notes_used=len(web_notes),
            )

        for repair_idx in range(1, 3):
            raw_sq = parsed.get("sub_questions", []) if isinstance(parsed.get("sub_questions"), list) else []
            raw_ch = parsed.get("master_outline", []) if isinstance(parsed.get("master_outline"), list) else []
            if len(raw_sq) >= 1 and len(raw_ch) >= 1:
                break
            planner_warnings.append(
                "planner-book-3 semantic retry: "
                f"attempt={repair_idx}, sub_questions={len(raw_sq)}, master_outline={len(raw_ch)}"
            )
            repair_payload = {
                "question": question,
                "invalid_json_payload": parsed,
                "required_schema": planner_schema_hint,
                "constraints": {
                    "language": "fr",
                    "strict_json_only": True,
                },
            }
            repaired_raw = await self._ask_text_model(
                model=json_coder_model,
                system_prompt=(
                    "Repare le JSON pour respecter strictement le schema. "
                    "Tu dois produire un JSON complet avec des listes non vides pour sub_questions et master_outline. "
                    "Aucun texte hors JSON."
                ),
                user_prompt=json.dumps(repair_payload, ensure_ascii=False),
                llm_logs=llm_logs,
                stage=f"planner-book-4-json-semantic-repair-{repair_idx}",
            )
            parsed = await self._parse_or_repair_json(
                raw_text=repaired_raw,
                model=json_coder_model,
                stage=f"planner-book-4-json-semantic-repair-{repair_idx}",
                llm_logs=llm_logs,
                expected_type="dict",
                schema_hint=planner_schema_hint,
            )

        raw_sub_questions = parsed.get("sub_questions", []) if isinstance(parsed.get("sub_questions"), list) else []
        normalized_sub_questions: list[dict[str, Any]] = []
        seen_sq_questions: set[str] = set()
        for idx, item in enumerate(raw_sub_questions, start=1):
            sq_question = str(item.get("question") or "").strip()
            if not sq_question:
                continue
            key = self._normalize_text_for_dedupe(sq_question)
            if not key or key in seen_sq_questions:
                continue
            seen_sq_questions.add(key)
            normalized_sub_questions.append(
                {
                    "id": str(item.get("id") or f"SQ{idx}").strip() or f"SQ{idx}",
                    "question": sq_question,
                    "proof_criteria": _to_clean_list(item.get("proof_criteria", []), max_items=0),
                    "search_queries": _to_clean_list(item.get("search_queries", []), max_items=self.config.web_query_variants),
                }
            )

        if len(normalized_sub_questions) < 1:
            raise RuntimeError(
                "Planner strict validation failed: stage-3 JSON returned zero sub-questions."
            )

        if self.config.web_max_sub_questions > 0:
            normalized_sub_questions = normalized_sub_questions[: self.config.web_max_sub_questions]
        for idx, item in enumerate(normalized_sub_questions, start=1):
            item["id"] = f"SQ{idx}"
            if not item.get("proof_criteria"):
                item["proof_criteria"] = [
                    "Sources independantes convergentes",
                    "Elements techniques ou operationnels verificables",
                    "Limites et incertitudes explicites",
                ]
            if len(item.get("search_queries", [])) < min(4, self.config.web_query_variants):
                for query in self._generate_query_variants(item["question"], self.config.web_query_variants):
                    if query not in item["search_queries"]:
                        item["search_queries"].append(query)
                    if len(item["search_queries"]) >= self.config.web_query_variants:
                        break

        planner_ids = [str(item["id"]) for item in normalized_sub_questions]
        raw_master_outline = parsed.get("master_outline", []) if isinstance(parsed.get("master_outline"), list) else []
        master_outline: list[dict[str, Any]] = []
        seen_outline_titles: set[str] = set()
        for idx, item in enumerate(raw_master_outline, start=1):
            title = self._localize_common_outline_label(str(item.get("title") or "").strip())
            if not title:
                continue
            key = self._normalize_text_for_dedupe(title)
            if not key or key in seen_outline_titles:
                continue
            seen_outline_titles.add(key)
            status = str(item.get("status") or "planned").strip().lower()
            if status not in {"planned", "rejected"}:
                status = "planned"
            master_outline.append(
                {
                    "id": str(item.get("id") or f"CH{idx}").strip() or f"CH{idx}",
                    "title": title,
                    "goal": str(item.get("goal") or "").strip(),
                    "linked_sub_questions_raw": _to_clean_list(item.get("linked_sub_questions", []), max_items=0),
                    "status": status,
                    "reason": str(item.get("reason") or "").strip(),
                    "sub_sections": self._normalize_subsections(
                        item.get("sub_sections", []),
                        chapter_title=title,
                        fill_missing=False,
                    ),
                }
            )

        if len(master_outline) < 1:
            raise RuntimeError(
                "Planner strict validation failed: stage-3 JSON returned zero chapters."
            )

        for idx, chapter in enumerate(master_outline, start=1):
            linked_ids: list[str] = []
            for raw_id in chapter.pop("linked_sub_questions_raw", []):
                mapped = self._map_sub_question_id(str(raw_id or ""), planner_ids)
                if mapped and mapped not in linked_ids:
                    linked_ids.append(mapped)
            if not linked_ids:
                linked_ids = self._match_sub_questions_for_topic(
                    chapter.get("title", ""),
                    normalized_sub_questions,
                    limit=0,
                )
            chapter["id"] = str(chapter.get("id") or f"CH{idx}").strip() or f"CH{idx}"
            chapter["linked_sub_questions"] = linked_ids
            sub_sections = self._normalize_subsections(
                chapter.get("sub_sections", []),
                chapter_title=str(chapter.get("title") or ""),
                fill_missing=False,
            )
            if not sub_sections:
                planner_warnings.append(
                    f"planner strict mapping: chapter {chapter['id']} has no sub-sections"
                )
            if self.config.outline_max_subsections > 0:
                chapter["sub_sections"] = sub_sections[: self.config.outline_max_subsections]
            else:
                chapter["sub_sections"] = sub_sections

        deduped_master: list[dict[str, Any]] = []
        seen_master_titles: set[str] = set()
        for idx, chapter in enumerate(master_outline, start=1):
            title = self._localize_common_outline_label(str(chapter.get("title") or "").strip())
            if not title:
                continue
            key = self._normalize_text_for_dedupe(title)
            if not key or key in seen_master_titles:
                continue
            seen_master_titles.add(key)
            linked = [
                sq_id
                for sq_id in chapter.get("linked_sub_questions", [])
                if isinstance(sq_id, str) and sq_id in planner_ids
            ]
            status = str(chapter.get("status") or "planned").strip().lower()
            if status not in {"planned", "rejected"}:
                status = "planned"
            deduped_master.append(
                {
                    "id": str(chapter.get("id") or f"CH{idx}").strip() or f"CH{idx}",
                    "title": title,
                    "goal": str(chapter.get("goal") or "").strip(),
                    "linked_sub_questions": linked,
                    "status": status,
                    "reason": str(chapter.get("reason") or "").strip(),
                    "sub_sections": (
                        chapter.get("sub_sections", [])[: self.config.outline_max_subsections]
                        if self.config.outline_max_subsections > 0
                        else chapter.get("sub_sections", [])
                    ),
                }
            )

        if len(deduped_master) < 1:
            raise RuntimeError(
                "Planner strict validation failed after normalization: zero chapters remain."
            )

        topic_seed = aggregated_topics or self._collect_presearch_topic_titles(presearch_payload, max_items=16)

        ambiguities = _to_clean_list(parsed.get("ambiguities", []), max_items=20)
        inclusion_criteria = _to_clean_list(parsed.get("inclusion_criteria", []), max_items=24)
        exclusion_criteria = _to_clean_list(parsed.get("exclusion_criteria", []), max_items=24)
        if not inclusion_criteria:
            inclusion_criteria = topic_seed[:12]
        if not exclusion_criteria:
            exclusion_criteria = [
                "Sujets hors perimetre non relies au sujet principal",
                "Contenu purement promotionnel sans valeur technique exploitable",
            ]

        scope_notes = str(parsed.get("scope_notes") or "").strip()
        if not scope_notes:
            scope_notes = (
                "Plan de dossier construit a partir d'une revue web multi-sources et d'une synthese "
                "iterative multi-modeles (Gemma -> Mistral -> Qwen)."
            )

        return {
            "question_reformulated": str(parsed.get("question_reformulated") or question).strip() or question,
            "ambiguities": ambiguities,
            "scope_notes": scope_notes,
            "inclusion_criteria": inclusion_criteria,
            "exclusion_criteria": exclusion_criteria,
            "master_outline": deduped_master,
            "planner_warnings": planner_warnings,
            "sub_questions": normalized_sub_questions,
            "planner_pipeline": {
                "models": {
                    "stage_1_outline": book_models[0],
                    "stage_2_enrichment": book_models[1],
                    "stage_3_editorial": book_models[2],
                    "stage_4_json_coder": json_coder_model,
                },
                "links_requested": self.config.planner_book_web_links,
                "links_used": len(selected_links),
                "web_notes_used": len(web_notes),
            },
        }

    def _validate_planner_output_strict(
        self,
        parsed: Any,
        question: str,
        book_models: list[str],
        json_coder_model: str,
        links_requested: int,
        links_used: int,
        web_notes_used: int,
    ) -> dict[str, Any]:
        if not isinstance(parsed, dict):
            raise RuntimeError("Planner strict/no-fallback: stage-3 output is not a JSON object.")

        question_reformulated = str(parsed.get("question_reformulated") or "").strip()
        if not question_reformulated:
            raise RuntimeError("Planner strict/no-fallback: missing 'question_reformulated'.")

        def _strict_list_of_strings(
            value: Any,
            label: str,
            allow_empty: bool,
            normalize_outline_label: bool = False,
        ) -> list[str]:
            if not isinstance(value, list):
                raise RuntimeError(f"Planner strict/no-fallback: '{label}' must be a list.")
            output: list[str] = []
            for item in value:
                text = str(item or "").strip()
                if normalize_outline_label:
                    text = re.sub(r"^\s*(?:[-*•]+|\d+(?:\.\d+)*[.)-]?)\s*", "", text).strip()
                    text = self._localize_common_outline_label(text)
                if text:
                    output.append(text)
            if not allow_empty and not output:
                raise RuntimeError(f"Planner strict/no-fallback: '{label}' cannot be empty.")
            return output

        ambiguities = _strict_list_of_strings(parsed.get("ambiguities", []), "ambiguities", allow_empty=True)
        inclusion_criteria = _strict_list_of_strings(
            parsed.get("inclusion_criteria", []),
            "inclusion_criteria",
            allow_empty=False,
        )
        exclusion_criteria = _strict_list_of_strings(
            parsed.get("exclusion_criteria", []),
            "exclusion_criteria",
            allow_empty=False,
        )
        scope_notes = str(parsed.get("scope_notes") or "").strip()
        if not scope_notes:
            raise RuntimeError("Planner strict/no-fallback: missing 'scope_notes'.")

        raw_sub_questions = parsed.get("sub_questions", [])
        if not isinstance(raw_sub_questions, list) or not raw_sub_questions:
            raise RuntimeError("Planner strict/no-fallback: 'sub_questions' must be a non-empty list.")
        normalized_sub_questions: list[dict[str, Any]] = []
        seen_sq_ids: set[str] = set()
        seen_sq_questions: set[str] = set()
        for idx, item in enumerate(raw_sub_questions, start=1):
            if not isinstance(item, dict):
                raise RuntimeError(f"Planner strict/no-fallback: sub_questions[{idx}] is not an object.")
            sq_id = str(item.get("id") or "").strip()
            if not sq_id:
                raise RuntimeError(f"Planner strict/no-fallback: sub_questions[{idx}] missing 'id'.")
            if sq_id in seen_sq_ids:
                raise RuntimeError(f"Planner strict/no-fallback: duplicate sub-question id '{sq_id}'.")
            sq_question = str(item.get("question") or "").strip()
            if not sq_question:
                raise RuntimeError(
                    f"Planner strict/no-fallback: sub_questions[{idx}] missing 'question'."
                )
            sq_key = self._normalize_text_for_dedupe(sq_question)
            if sq_key and sq_key in seen_sq_questions:
                raise RuntimeError(
                    "Planner strict/no-fallback: duplicate sub-question text detected."
                )
            proof = _strict_list_of_strings(
                item.get("proof_criteria", []),
                f"sub_questions[{idx}].proof_criteria",
                allow_empty=False,
            )
            queries = _strict_list_of_strings(
                item.get("search_queries", []),
                f"sub_questions[{idx}].search_queries",
                allow_empty=False,
            )
            seen_sq_ids.add(sq_id)
            if sq_key:
                seen_sq_questions.add(sq_key)
            normalized_sub_questions.append(
                {
                    "id": sq_id,
                    "question": sq_question,
                    "proof_criteria": proof,
                    "search_queries": queries,
                }
            )

        raw_master_outline = parsed.get("master_outline", [])
        if not isinstance(raw_master_outline, list) or not raw_master_outline:
            raise RuntimeError("Planner strict/no-fallback: 'master_outline' must be a non-empty list.")
        normalized_outline: list[dict[str, Any]] = []
        seen_ch_ids: set[str] = set()
        seen_ch_titles: set[str] = set()
        valid_sq_ids = {item["id"] for item in normalized_sub_questions}
        for idx, item in enumerate(raw_master_outline, start=1):
            if not isinstance(item, dict):
                raise RuntimeError(f"Planner strict/no-fallback: master_outline[{idx}] is not an object.")
            ch_id = str(item.get("id") or "").strip()
            if not ch_id:
                raise RuntimeError(f"Planner strict/no-fallback: master_outline[{idx}] missing 'id'.")
            if ch_id in seen_ch_ids:
                raise RuntimeError(f"Planner strict/no-fallback: duplicate chapter id '{ch_id}'.")
            title = str(item.get("title") or "").strip()
            if not title:
                raise RuntimeError(f"Planner strict/no-fallback: master_outline[{idx}] missing 'title'.")
            title_key = self._normalize_text_for_dedupe(title)
            if title_key and title_key in seen_ch_titles:
                raise RuntimeError("Planner strict/no-fallback: duplicate chapter title detected.")
            goal = str(item.get("goal") or "").strip()
            if not goal:
                raise RuntimeError(f"Planner strict/no-fallback: master_outline[{idx}] missing 'goal'.")
            status = str(item.get("status") or "").strip().lower()
            if status not in {"planned", "rejected"}:
                raise RuntimeError(
                    f"Planner strict/no-fallback: master_outline[{idx}].status must be 'planned' or 'rejected'."
                )
            reason = str(item.get("reason") or "").strip()
            if not reason:
                raise RuntimeError(f"Planner strict/no-fallback: master_outline[{idx}] missing 'reason'.")
            linked_ids_raw = item.get("linked_sub_questions", [])
            if not isinstance(linked_ids_raw, list) or not linked_ids_raw:
                raise RuntimeError(
                    f"Planner strict/no-fallback: master_outline[{idx}].linked_sub_questions cannot be empty."
                )
            linked_ids: list[str] = []
            for raw_id in linked_ids_raw:
                sq_id = str(raw_id or "").strip()
                if not sq_id:
                    continue
                if sq_id not in valid_sq_ids:
                    raise RuntimeError(
                        f"Planner strict/no-fallback: chapter '{ch_id}' links unknown sub-question '{sq_id}'."
                    )
                if sq_id not in linked_ids:
                    linked_ids.append(sq_id)
            if not linked_ids:
                raise RuntimeError(
                    f"Planner strict/no-fallback: chapter '{ch_id}' has no valid linked_sub_questions."
                )
            sub_sections = _strict_list_of_strings(
                item.get("sub_sections", []),
                f"master_outline[{idx}].sub_sections",
                allow_empty=False,
                normalize_outline_label=True,
            )
            seen_ch_ids.add(ch_id)
            if title_key:
                seen_ch_titles.add(title_key)
            normalized_outline.append(
                {
                    "id": ch_id,
                    "title": title,
                    "goal": goal,
                    "linked_sub_questions": linked_ids,
                    "status": status,
                    "reason": reason,
                    "sub_sections": sub_sections,
                }
            )

        return {
            "question_reformulated": question_reformulated or question,
            "ambiguities": ambiguities,
            "scope_notes": scope_notes,
            "inclusion_criteria": inclusion_criteria,
            "exclusion_criteria": exclusion_criteria,
            "master_outline": normalized_outline,
            "planner_warnings": [],
            "sub_questions": normalized_sub_questions,
            "planner_pipeline": {
                "models": {
                    "stage_1_outline": book_models[0],
                    "stage_2_enrichment": book_models[1],
                    "stage_3_editorial": book_models[2],
                    "stage_4_json_coder": json_coder_model,
                },
                "links_requested": links_requested,
                "links_used": links_used,
                "web_notes_used": web_notes_used,
            },
        }

    async def _search_subquestions(
        self,
        planner: dict[str, Any],
        llm_logs: list[dict[str, Any]],
        progress_cb: ProgressCallback | None,
        run_dir: Path,
    ) -> dict[str, Any]:
        output: dict[str, Any] = {"sub_questions": [], "engine": None, "generated_at": int(time.time())}

        # Determine Search Engine
        searx_available = False
        engine_choice = self.config.web_search_engine
        
        if engine_choice == "searxng":
            searx_available = True # Force try
        elif engine_choice == "duckduckgo":
            searx_available = False
        else: # auto
            searx_available = await self._probe_searxng()

        output["engine"] = "searxng" if searx_available else "duckduckgo"

        sub_questions = planner.get("sub_questions", [])
        for sq_index, sub_question in enumerate(sub_questions, start=1):
            sq_id = str(sub_question.get("id") or f"SQ{sq_index}")
            question = str(sub_question.get("question") or "")
            queries = [q for q in sub_question.get("search_queries", []) if q]

            if self.config.strict_no_fallback and not queries:
                raise RuntimeError(
                    f"Search strict/no-fallback: sub-question {sq_id} has no search_queries."
                )

            if (not self.config.strict_no_fallback) and len(queries) < self.config.web_query_variants:
                generated = self._generate_query_variants(question, self.config.web_query_variants)
                for query in generated:
                    if query not in queries:
                        queries.append(query)
                    if len(queries) >= self.config.web_query_variants:
                        break

            await self._emit(
                progress_cb,
                run_dir,
                "search",
                f"Searching sub-question {sq_id} with {len(queries)} query variants",
            )

            collected_links: list[dict[str, Any]] = []
            seen_urls: set[str] = set()
            query_errors: list[dict[str, str]] = []
            for query in queries:
                remaining = self.config.web_max_links_per_subquestion - len(collected_links)
                if remaining <= 0:
                    break
                
                # Anti-ban Delay
                if self.config.web_request_delay > 0:
                    await asyncio.sleep(self.config.web_request_delay)

                try:
                    if searx_available:
                        query_hits = await self._searx_search(
                            query=query,
                            max_results=min(remaining, self.config.web_per_query_results),
                        )
                    else:
                        query_hits = await self._ddg_search(
                            query=query,
                            max_results=min(remaining, self.config.web_per_query_results),
                        )
                except Exception as exc:  # noqa: BLE001
                    query_errors.append({"query": query, "error": str(exc) or repr(exc)})
                    continue

                for hit in query_hits:
                    url = self._canonicalize_url(str(hit.get("url") or ""))
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    collected_links.append(
                        {
                            "id": f"LNK-{uuid.uuid4().hex[:10]}",
                            "query": query,
                            "title": str(hit.get("title") or "").strip(),
                            "url": url,
                            "snippet": str(hit.get("snippet") or "").strip(),
                            "published_at": str(hit.get("published_at") or "").strip(),
                            "engine": output["engine"],
                        }
                    )
                    if len(collected_links) >= self.config.web_max_links_per_subquestion:
                        break

            if not collected_links:
                if query_errors:
                    error_summary = "; ".join(f"{item['query']}: {item['error']}" for item in query_errors[:6])
                    raise RuntimeError(
                        f"Search failed for sub-question {sq_id} ({question}). Query errors: {error_summary}"
                    )
                raise RuntimeError(f"Search returned zero links for sub-question {sq_id} ({question})")

            output["sub_questions"].append(
                {
                    "id": sq_id,
                    "question": question,
                    "queries": queries,
                    "links": collected_links,
                    "query_errors": query_errors,
                }
            )
            await self._emit(
                progress_cb,
                run_dir,
                "search",
                f"Sub-question {sq_id}: {len(collected_links)} links collected",
            )

        return output

    async def _build_outline_from_search(
        self,
        planner: dict[str, Any],
        search_results: dict[str, Any],
        llm_logs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        outline_warnings: list[str] = []
        planner_sq_ids = [str(item.get("id") or "").strip() for item in planner.get("sub_questions", []) if str(item.get("id") or "").strip()]
        input_rows: list[dict[str, Any]] = []
        for sq in search_results.get("sub_questions", []):
            input_rows.append(
                {
                    "id": sq.get("id"),
                    "question": sq.get("question"),
                    "links": [
                        {
                            "title": link.get("title", ""),
                            "url": link.get("url", ""),
                            "snippet": link.get("snippet", ""),
                        }
                        for link in sq.get("links", [])[:30]
                    ],
                }
            )

        payload = {
            "question": planner.get("question_reformulated") or "",
            "planner_master_outline": planner.get("master_outline", []),
            "sub_questions": planner.get("sub_questions", []),
            "search_evidence": input_rows,
            "requirements": {
                "build_global_outline": True,
                "allow_new_topics_if_related": True,
                "reject_out_of_scope_topics": True,
                "include_multiple_viewpoints": [
                    "benefits",
                    "limitations",
                    "risks",
                    "implementation",
                    "performance tradeoffs",
                ],
            },
            "output_schema": {
                "global_topics": [
                    {
                        "id": "TOP1",
                        "title": "string",
                        "scope": "string",
                        "related_sub_question_ids": ["SQ1"],
                        "status": "planned|rejected",
                        "reason": "string",
                    }
                ],
                "sub_question_topics": [
                    {
                        "id": "SQ1",
                        "topics": [
                            {
                                "title": "string",
                                "why_relevant": "string",
                                "viewpoint": "benefits|limitations|both",
                            }
                        ],
                        "rejected_topics": [
                            {"title": "string", "reason": "string"}
                        ],
                    }
                ],
            },
        }

        response_text = await self._ask_text_model(
            model=self.config.judge_model,
            system_prompt=(
                "You are an outline architect for deep research dossiers. "
                "Build a practical and exhaustive table of contents from search evidence. "
                "Return strict JSON only."
            ),
            user_prompt=json.dumps(payload, ensure_ascii=False),
            llm_logs=llm_logs,
            stage="outline",
        )
        parsed = await self._parse_or_repair_json(
            raw_text=response_text,
            model=self.config.judge_model,
            stage="outline",
            llm_logs=llm_logs,
            expected_type="dict",
            schema_hint=payload["output_schema"],
        )

        global_topics = parsed.get("global_topics")
        sub_topics = parsed.get("sub_question_topics")
        if not isinstance(global_topics, list) or not isinstance(sub_topics, list):
            raise RuntimeError("Outline JSON missing 'global_topics' or 'sub_question_topics'")

        by_sq: dict[str, dict[str, Any]] = {}
        for item in sub_topics:
            sq_id_raw = str(item.get("id") or "").strip()
            sq_id = self._map_sub_question_id(sq_id_raw, planner_sq_ids)
            if not sq_id:
                continue
            topics = item.get("topics", [])
            rejected_topics = item.get("rejected_topics", [])
            if not isinstance(topics, list):
                topics = []
            if not isinstance(rejected_topics, list):
                rejected_topics = []
            current = by_sq.get(
                sq_id,
                {
                    "id": sq_id,
                    "topics": [],
                    "rejected_topics": [],
                },
            )
            current["topics"].extend(
                [
                    {
                        "title": str(topic.get("title") or "").strip(),
                        "why_relevant": str(topic.get("why_relevant") or "").strip(),
                        "viewpoint": str(topic.get("viewpoint") or "both").strip().lower(),
                    }
                    for topic in topics
                    if str(topic.get("title") or "").strip()
                ]
            )
            current["rejected_topics"].extend(
                [
                    {
                        "title": str(topic.get("title") or "").strip(),
                        "reason": str(topic.get("reason") or "").strip(),
                    }
                    for topic in rejected_topics
                    if str(topic.get("title") or "").strip()
                ]
            )
            by_sq[sq_id] = {
                "id": sq_id,
                "topics": current["topics"],
                "rejected_topics": current["rejected_topics"],
            }

        normalized_global_topics = []
        for topic in global_topics:
            title = str(topic.get("title") or "").strip()
            if not title:
                continue
            normalized_global_topics.append(
                {
                    "id": str(topic.get("id") or f"TOP-{len(normalized_global_topics)+1}"),
                    "title": title,
                    "scope": str(topic.get("scope") or "").strip(),
                    "related_sub_question_ids": [
                        str(v).strip()
                        for v in topic.get("related_sub_question_ids", [])
                        if str(v).strip()
                    ],
                    "status": str(topic.get("status") or "planned").strip().lower(),
                    "reason": str(topic.get("reason") or "").strip(),
                }
            )

        if self.config.strict_no_fallback:
            if not normalized_global_topics:
                raise RuntimeError(
                    "Outline strict/no-fallback: model returned no global_topics."
                )
            missing_sq_ids = [sq_id for sq_id in planner_sq_ids if sq_id and sq_id not in by_sq]
            if missing_sq_ids:
                outline_warnings.append(
                    "Outline strict/no-fallback (BYPASSED): model omitted sub-question topics for: "
                    + ", ".join(missing_sq_ids)
                )
            for sq_id, payload_sq in by_sq.items():
                payload_sq["topics"] = self._dedupe_topics(payload_sq.get("topics", []), max_items=20)
                payload_sq["rejected_topics"] = self._dedupe_rejected_topics(
                    payload_sq.get("rejected_topics", []),
                    max_items=30,
                )
                if not payload_sq["topics"]:
                    raise RuntimeError(
                        f"Outline strict/no-fallback: empty topics for sub-question {sq_id}."
                    )
            normalized_global_topics = self._dedupe_global_topics(normalized_global_topics)
            return {
                "global_topics": normalized_global_topics,
                "sub_question_topics": by_sq,
                "warnings": [],
                "generated_at": int(time.time()),
            }

        planner_sq_by_id = {
            str(item.get("id") or "").strip(): item
            for item in planner.get("sub_questions", [])
            if str(item.get("id") or "").strip()
        }
        if not normalized_global_topics:
            outline_warnings.append(
                "Outline stage returned empty global topic list; generated from planner master_outline."
            )
            master_outline = planner.get("master_outline", [])
            for idx, item in enumerate(master_outline if isinstance(master_outline, list) else [], start=1):
                title = str(item.get("title") or "").strip()
                if not title:
                    continue
                normalized_global_topics.append(
                    {
                        "id": str(item.get("id") or f"TOP-{idx}").strip() or f"TOP-{idx}",
                        "title": title,
                        "scope": str(item.get("goal") or "").strip(),
                        "related_sub_question_ids": [
                            str(v).strip()
                            for v in item.get("linked_sub_questions", [])
                            if str(v).strip()
                        ],
                        "status": str(item.get("status") or "planned").strip().lower(),
                        "reason": str(item.get("reason") or "derived_from_planner").strip(),
                    }
                )
            if not normalized_global_topics:
                raise RuntimeError("Outline stage produced empty global topic list and planner fallback is empty")

        missing_sq_ids = [sq_id for sq_id in planner_sq_ids if sq_id and sq_id not in by_sq]
        if missing_sq_ids:
            outline_warnings.append(
                "Outline model omitted sub-question topics for: " + ", ".join(missing_sq_ids)
            )
            for sq_id in missing_sq_ids:
                sq = planner_sq_by_id.get(sq_id, {})
                sq_question = str(sq.get("question") or sq_id).strip()
                proof_criteria = sq.get("proof_criteria", [])
                if not isinstance(proof_criteria, list):
                    proof_criteria = []
                fallback_topics = [
                    {
                        "title": sq_question,
                        "why_relevant": "Topic genere automatiquement car omis par l'outline LLM.",
                        "viewpoint": "both",
                    }
                ]
                if proof_criteria:
                    fallback_topics.append(
                        {
                            "title": "Verification et limites",
                            "why_relevant": str(proof_criteria[0])[:220],
                            "viewpoint": "limitations",
                        }
                    )
                by_sq[sq_id] = {
                    "id": sq_id,
                    "topics": fallback_topics,
                    "rejected_topics": [
                        {
                            "title": "Topic manquant dans la sortie outline",
                            "reason": "Generation explicite a partir du planner pour conserver la continuite du run.",
                        }
                    ],
                }
        empty_topic_sq_ids = [sq_id for sq_id, value in by_sq.items() if not value.get("topics")]
        if empty_topic_sq_ids:
            outline_warnings.append(
                "Outline produced empty topic list for: " + ", ".join(empty_topic_sq_ids)
            )
            for sq_id in empty_topic_sq_ids:
                sq = planner_sq_by_id.get(sq_id, {})
                sq_question = str(sq.get("question") or sq_id).strip()
                by_sq[sq_id]["topics"] = [
                    {
                        "title": sq_question,
                        "why_relevant": "Topic principal injecte car liste vide.",
                        "viewpoint": "both",
                    }
                ]

        for sq_id, payload_sq in by_sq.items():
            payload_sq["topics"] = self._dedupe_topics(payload_sq.get("topics", []), max_items=20)
            payload_sq["rejected_topics"] = self._dedupe_rejected_topics(
                payload_sq.get("rejected_topics", []),
                max_items=30,
            )
            if not payload_sq["topics"]:
                sq = planner_sq_by_id.get(sq_id, {})
                sq_question = str(sq.get("question") or sq_id).strip()
                payload_sq["topics"] = [
                    {
                        "title": sq_question,
                        "why_relevant": "Topic principal injecte apres deduplication.",
                        "viewpoint": "both",
                    }
                ]
                outline_warnings.append(
                    f"Deduplication removed all topics for {sq_id}; injected primary topic."
                )

        normalized_global_topics = self._dedupe_global_topics(normalized_global_topics)

        return {
            "global_topics": normalized_global_topics,
            "sub_question_topics": by_sq,
            "warnings": outline_warnings,
            "generated_at": int(time.time()),
        }

    async def _build_corpus(
        self,
        search_results: dict[str, Any],
        progress_cb: ProgressCallback | None,
        run_dir: Path,
    ) -> dict[str, Any]:
        link_records: list[tuple[str, dict[str, Any]]] = []
        for sq in search_results.get("sub_questions", []):
            sq_id = sq.get("id")
            for link in sq.get("links", [])[: self.config.web_fetch_limit_per_subquestion]:
                link_records.append((sq_id, link))

        dedup_by_url: dict[str, dict[str, Any]] = {}
        link_to_sq: dict[str, set[str]] = {}
        for sq_id, link in link_records:
            url = self._canonicalize_url(str(link.get("url") or ""))
            if not url:
                continue
            if url not in dedup_by_url:
                dedup_by_url[url] = link
            link_to_sq.setdefault(url, set()).add(str(sq_id))

        urls = list(dedup_by_url.keys())
        await self._emit(
            progress_cb,
            run_dir,
            "corpus",
            f"Fetching and normalizing {len(urls)} unique sources",
        )

        semaphore = asyncio.Semaphore(self.config.max_parallel_fetch)
        processed_count = 0

        async def fetch_one(url: str) -> dict[str, Any] | None:
            nonlocal processed_count
            async with semaphore:
                result = await self._fetch_source(url, run_dir)
                processed_count += 1
                if processed_count % 10 == 0:
                    await self._emit(
                        progress_cb,
                        run_dir,
                        "corpus",
                        f"Processed {processed_count}/{len(urls)} sources",
                    )
                return result

        docs = [doc for doc in await asyncio.gather(*(fetch_one(url) for url in urls)) if doc]
        doc_map = {doc["canonical_url"]: doc for doc in docs}

        for sq in search_results.get("sub_questions", []):
            doc_ids: list[str] = []
            for link in sq.get("links", []):
                canonical_url = self._canonicalize_url(str(link.get("url") or ""))
                doc = doc_map.get(canonical_url)
                if not doc:
                    continue
                doc_ids.append(doc["source_id"])
            sq["doc_ids"] = list(dict.fromkeys(doc_ids))

        return {
            "stats": {
                "fetched_unique_urls": len(urls),
                "valid_documents": len(docs),
            },
            "sources": docs,
            "source_ids_by_url": {doc["canonical_url"]: doc["source_id"] for doc in docs},
            "sub_question_mapping": {url: sorted(list(sqs)) for url, sqs in link_to_sq.items()},
        }

    async def _rank_and_shortlist(
        self,
        planner: dict[str, Any],
        search_results: dict[str, Any],
        corpus: dict[str, Any],
        llm_logs: list[dict[str, Any]],
        progress_cb: ProgressCallback | None = None,
        run_dir: Path | None = None,
    ) -> dict[str, Any]:
        source_by_id = {source["source_id"]: source for source in corpus.get("sources", [])}
        source_ids_by_url = corpus.get("source_ids_by_url", {})
        shortlist_by_sq: list[dict[str, Any]] = []

        for sub_question in planner.get("sub_questions", []):
            sq_id = sub_question.get("id")
            sq_text = str(sub_question.get("question") or "")
            sq_search = next((sq for sq in search_results.get("sub_questions", []) if sq.get("id") == sq_id), {})
            candidate_source_ids: list[str] = []
            for source_id in sq_search.get("doc_ids", []):
                if isinstance(source_id, str) and source_id:
                    candidate_source_ids.append(source_id)

            if not candidate_source_ids:
                for link in sq_search.get("links", []):
                    canonical_url = self._canonicalize_url(str(link.get("url") or ""))
                    source_id = source_ids_by_url.get(canonical_url)
                    if source_id:
                        candidate_source_ids.append(str(source_id))

            candidate_source_ids = list(dict.fromkeys(candidate_source_ids))
            candidates = []
            for source_id in candidate_source_ids:
                source = source_by_id.get(source_id)
                if not source:
                    continue
                score = self._heuristic_source_score(sq_text, source)
                candidates.append(
                    {
                        "source_id": source_id,
                        "score": score,
                        "title": source.get("title", ""),
                        "domain": source.get("domain", ""),
                        "published_at": source.get("published_at", ""),
                        "url": source.get("canonical_url", ""),
                    }
                )

            candidates.sort(key=lambda item: item["score"]["final"], reverse=True)
            top_candidates = candidates[: max(self.config.web_shortlist_per_subquestion * 2, 20)]

            reranked = await self._llm_rerank_sources(
                question=sq_text,
                candidates=top_candidates,
                llm_logs=llm_logs,
                progress_cb=progress_cb,
                run_dir=run_dir,
                context_label=sq_id,
            )
            rerank_index = {item["source_id"]: item for item in reranked}

            enriched: list[dict[str, Any]] = []
            for idx, candidate in enumerate(top_candidates, start=1):
                rerank = rerank_index.get(candidate["source_id"])
                llm_score = float(rerank.get("score", 0.0)) if rerank else 0.0
                final_score = candidate["score"]["final"] * 0.7 + llm_score * 0.3
                enriched.append(
                    {
                        **candidate,
                        "heuristics": candidate["score"],
                        "llm_rerank_score": llm_score,
                        "combined_score": final_score,
                        "pre_rank": idx,
                        "llm_reason": (rerank or {}).get("reason", ""),
                    }
                )

            enriched.sort(key=lambda item: item["combined_score"], reverse=True)
            shortlist = enriched[: self.config.web_shortlist_per_subquestion]

            shortlist_by_sq.append(
                {
                    "id": sq_id,
                    "question": sq_text,
                    "selected_sources": shortlist,
                }
            )

        return {
            "shortlist_by_subquestion": shortlist_by_sq,
            "generated_at": int(time.time()),
        }

    async def _extract_claims(
        self,
        planner: dict[str, Any],
        shortlist: dict[str, Any],
        corpus: dict[str, Any],
        llm_logs: list[dict[str, Any]],
        progress_cb: ProgressCallback | None,
        run_dir: Path,
    ) -> dict[str, Any]:
        source_by_id = {source["source_id"]: source for source in corpus.get("sources", [])}
        claims: list[dict[str, Any]] = []
        sources_without_claims: list[str] = []

        jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for sq in shortlist.get("shortlist_by_subquestion", []):
            for selected in sq.get("selected_sources", []):
                source = source_by_id.get(selected.get("source_id"))
                if not source:
                    continue
                jobs.append((sq, source))

        semaphore = asyncio.Semaphore(self.config.max_parallel_llm)
        processed = 0

        async def extract_job(sq: dict[str, Any], source: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
            nonlocal processed
            async with semaphore:
                extracted = await self._extract_claims_from_source(
                    sub_question=sq,
                    source=source,
                    llm_logs=llm_logs,
                )
                processed += 1
                if processed % 5 == 0:
                    await self._emit(
                        progress_cb,
                        run_dir,
                        "claims",
                        f"Claim extraction {processed}/{len(jobs)} sources",
                    )
                return str(source.get("source_id") or ""), extracted

        results = await asyncio.gather(*(extract_job(sq, source) for sq, source in jobs))
        for source_id, batch in results:
            if not batch and source_id:
                sources_without_claims.append(source_id)
            claims.extend(batch)

        claims = claims[: 4000]
        if not claims:
            raise RuntimeError(
                "Claims stage produced zero claims. "
                f"Sources processed: {len(jobs)}; empty extractions: {len(sources_without_claims)}"
            )
        return {
            "claims": claims,
            "sources_without_claims": sorted(list(dict.fromkeys(sources_without_claims))),
            "generated_at": int(time.time()),
        }

    async def _verify_claims(
        self,
        planner: dict[str, Any],
        claims_payload: dict[str, Any],
        llm_logs: list[dict[str, Any]],
        progress_cb: ProgressCallback | None,
        run_dir: Path,
    ) -> dict[str, Any]:
        claims = claims_payload.get("claims", [])
        claims_by_id = {claim["claim_id"]: claim for claim in claims}
        
        # --- RESUME LOGIC ---
        partial_path = run_dir / "verdicts.partial.json"
        verdicts: list[dict[str, Any]] = []
        if partial_path.exists():
            try:
                verdicts = json.loads(partial_path.read_text(encoding="utf-8"))
                await self._emit(progress_cb, run_dir, "verification", f"Resuming verification: {len(verdicts)}/{len(claims)} already done")
            except Exception:
                pass
        
        verified_ids = {v["claim_id"] for v in verdicts}
        # --------------------

        semaphore = asyncio.Semaphore(self.config.max_parallel_llm)
        processed = len(verdicts)

        async def verify_one(claim: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                related = self._find_related_claims(claim, claims)
                verdict = await self._verify_single_claim(claim, related, llm_logs)
                return verdict

        # Filter claims to verify
        to_verify = [c for c in claims if c["claim_id"] not in verified_ids]
        
        # Batch processing to allow incremental save
        batch_size = 5  # Small batch to save often
        for i in range(0, len(to_verify), batch_size):
            batch = to_verify[i : i + batch_size]
            batch_results = await asyncio.gather(*(verify_one(c) for c in batch))
            
            verdicts.extend(batch_results)
            processed += len(batch_results)
            
            # Save Partial
            try:
                partial_path.write_text(json.dumps(verdicts, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                print(f"Warning: Failed to save partial verdicts: {e}")
            
            await self._emit(
                progress_cb,
                run_dir,
                "verification",
                f"Verified {processed}/{len(claims)} claims",
            )

        # Cleanup Partial
        if partial_path.exists():
            try:
                partial_path.unlink()
            except Exception:
                pass

        graph_edges: list[dict[str, Any]] = []
        for verdict in verdicts:
            claim_id = verdict["claim_id"]
            for support_id in verdict.get("supporting_claim_ids", []):
                if support_id in claims_by_id:
                    graph_edges.append(
                        {
                            "from": support_id,
                            "to": claim_id,
                            "relation": "supports",
                        }
                    )
            for contradict_id in verdict.get("contradicting_claim_ids", []):
                if contradict_id in claims_by_id:
                    graph_edges.append(
                        {
                            "from": contradict_id,
                            "to": claim_id,
                            "relation": "contradicts",
                        }
                    )

        by_status = {
            "ACCEPTED": sum(1 for v in verdicts if v.get("status") == "ACCEPTED"),
            "UNCERTAIN": sum(1 for v in verdicts if v.get("status") == "UNCERTAIN"),
            "REJECTED": sum(1 for v in verdicts if v.get("status") == "REJECTED"),
        }

        return {
            "verdicts": verdicts,
            "graph": {
                "nodes": [{"claim_id": claim["claim_id"], "sub_question_id": claim["sub_question_id"]} for claim in claims],
                "edges": graph_edges,
            },
            "stats": by_status,
            "generated_at": int(time.time()),
        }

    async def _write_sections(
        self,
        planner: dict[str, Any],
        outline_seed: dict[str, Any],
        corpus: dict[str, Any],
        claims_payload: dict[str, Any],
        verdicts_payload: dict[str, Any],
        llm_logs: list[dict[str, Any]],
        progress_cb: ProgressCallback | None = None,
        run_dir: Path | None = None,
    ) -> dict[str, Any]:
        master_outline = planner.get("master_outline", [])
        if not isinstance(master_outline, list):
            master_outline = []

        # Indexing claims
        claims_by_sq: dict[str, list[dict[str, Any]]] = {}
        all_verdicts = {v.get("claim_id"): v for v in verdicts_payload.get("verdicts", [])}
        for claim in claims_payload.get("claims", []):
            sq_id = str(claim.get("sub_question_id") or "").strip()
            if not sq_id: continue
            if sq_id not in claims_by_sq: claims_by_sq[sq_id] = []
            v = all_verdicts.get(claim.get("claim_id"))
            if v:
                enriched = dict(claim)
                enriched["verdict"] = v
                claims_by_sq[sq_id].append(enriched)

        sections_content = []
        is_hierarchical = False
        if master_outline and isinstance(master_outline[0], dict) and "party_title" in master_outline[0]:
            is_hierarchical = True

        total_units = 0
        if is_hierarchical:
            for p in master_outline:
                for c in p.get("chapters", []):
                    total_units += len(c.get("sub_sections", []))
        else:
            total_units = len(master_outline)

        processed_units = 0

        if is_hierarchical:
            for p_idx, party in enumerate(master_outline, 1):
                p_title = party.get("party_title", f"Partie {p_idx}")
                sections_content.append({"type": "party", "title": p_title})
                for c_idx, chapter in enumerate(party.get("chapters", []), 1):
                    c_title = chapter.get("chapter_title", f"Chapitre {c_idx}")
                    sections_content.append({"type": "chapter", "title": c_title})
                    for s_idx, sub in enumerate(chapter.get("sub_sections", []), 1):
                        s_title = sub.get("title", f"Section {s_idx}")
                        s_brief = sub.get("brief", "")
                        processed_units += 1
                        await self._emit(progress_cb, run_dir, "writing", f"Writing {processed_units}/{total_units}: {p_title} > {c_title} > {s_title}")
                        prompt = f"Write a detailed academic sub-section for a dissertation.\nPartie: {p_title}\nChapitre: {c_title}\nSection: {s_title}\nDescription: {s_brief}\n\nRules:\n- Min 800 words.\n- Technical depth.\n- Cite [CLM-id] if possible.\n- Use Markdown."
                        content = await self._ask_text_model(model=self.config.writer_model, system_prompt="Expert Dissertation Writer.", user_prompt=prompt, llm_logs=llm_logs, stage="writing")
                        sections_content.append({"type": "section", "title": s_title, "content": content})
        else:
            for idx, chap in enumerate(master_outline, 1):
                title = chap.get("chapter_title") or chap.get("title") or f"Chapitre {idx}"
                processed_units += 1
                await self._emit(progress_cb, run_dir, "writing", f"Writing chapter {processed_units}/{total_units}")
                content = await self._ask_text_model(model=self.config.writer_model, system_prompt="Standard Writer.", user_prompt=f"Write chapter: {title}", llm_logs=llm_logs, stage="writing")
                sections_content.append({"type": "chapter", "title": title, "content": content})

        return {"sections": sections_content, "generated_at": int(time.time())}

    async def _assemble_report(
        master_outline = planner.get("master_outline", [])
        if not isinstance(master_outline, list) or not master_outline:
            master_outline = []

        # 1. Index claims by sub-question ID
        claims_by_sq: dict[str, list[dict[str, Any]]] = {}
        # Also index by source ID to enrich with source title/url later if needed
        source_by_id = {source["source_id"]: source for source in corpus.get("sources", [])}
        
        all_verdicts = {v.get("claim_id"): v for v in verdicts_payload.get("verdicts", [])}

        for claim in claims_payload.get("claims", []):
            sq_id = str(claim.get("sub_question_id") or claim.get("sq_id") or "").strip()
            if not sq_id:
                continue
            if sq_id not in claims_by_sq:
                claims_by_sq[sq_id] = []
            
            verdict = all_verdicts.get(claim.get("claim_id"))
            enriched_claim = dict(claim)
            if verdict:
                enriched_claim["status"] = verdict.get("status", "UNCERTAIN")
                enriched_claim["confidence"] = verdict.get("confidence_score", 0.5)
            else:
                enriched_claim["status"] = "UNCERTAIN"
                enriched_claim["confidence"] = 0.5
            
            # Add source context
            source = source_by_id.get(claim.get("source_id"))
            if source:
                enriched_claim["source_title"] = source.get("title")
                enriched_claim["source_url"] = source.get("canonical_url")

            claims_by_sq[sq_id].append(enriched_claim)

        # 2. Iterate through chapters
        semaphore = asyncio.Semaphore(self.config.max_parallel_llm)
        total_chapters = len(master_outline)
        completed_chapters = 0

        async def write_one_chapter(chapter: dict[str, Any]) -> dict[str, Any] | None:
            nonlocal completed_chapters
            async with semaphore:
                chapter_id = str(chapter.get("id") or "").strip()
                linked_sqs = chapter.get("linked_sub_questions", [])
                
                # Aggregate claims
                relevant_claims = []
                for sq_id in linked_sqs:
                    if str(sq_id) in claims_by_sq:
                        relevant_claims.extend(claims_by_sq[str(sq_id)])
                
                # Deduplicate by claim_id
                unique_claims_map = {}
                for c in relevant_claims:
                    unique_claims_map[c.get("claim_id")] = c
                unique_claims = list(unique_claims_map.values())
                
                # Sort: Accepted first, then high confidence
                sorted_claims = sorted(
                    unique_claims, 
                    key=lambda c: (c.get("status") == "ACCEPTED", float(c.get("confidence", 0))), 
                    reverse=True
                )

                try:
                    section = await self._draft_chapter_content(
                        chapter=chapter,
                        claims=sorted_claims,
                        llm_logs=llm_logs,
                    )
                    # Pass through the claims for the assembler to build tables
                    section["claims"] = sorted_claims 
                    
                    completed_chapters += 1
                    if progress_cb and run_dir:
                        await self._emit(
                            progress_cb,
                            run_dir,
                            "writing",
                            f"Writing chapter {completed_chapters}/{total_chapters}: {chapter.get('title', 'Untitled')}",
                        )
                    return section
                except Exception as exc:  # noqa: BLE001
                    print(f"Error writing chapter {chapter_id}: {exc}")
                    return {
                        "id": chapter_id,
                        "title": chapter.get("title"),
                        "markdown": f"*(Erreur de rédaction pour le chapitre {chapter_id}: {exc})*",
                        "claims": [],
                        "error": str(exc)
                    }

        tasks = [write_one_chapter(chapter) for chapter in master_outline]
        results = await asyncio.gather(*tasks)
        sections = [s for s in results if s]

        return {
            "sections": sections,
            "stats": {
                "chapters_planned": total_chapters,
                "chapters_written": len(sections),
            },
            "generated_at": int(time.time()),
        }

    async def _draft_chapter_content(
        self,
        chapter: dict[str, Any],
        claims: list[dict[str, Any]],
        llm_logs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        chapter_title = str(chapter.get("title") or "Chapitre Sans Titre")
        chapter_goal = str(chapter.get("goal") or "")
        sub_sections = chapter.get("sub_sections", [])
        
        # Format claims for the prompt (limit to top 60 to fit context)
        claims_text = ""
        # Prefer accepted claims
        priority_claims = [c for c in claims if c.get("status") == "ACCEPTED"]
        # Then uncertain ones if we need filler
        uncertain_claims = [c for c in claims if c.get("status") == "UNCERTAIN"]
        
        selection = priority_claims[:50]
        if len(selection) < 20:
            selection.extend(uncertain_claims[:(30 - len(selection))])
            
        for i, claim in enumerate(selection, 1):
            source_info = f"Source: {claim.get('source_title', 'Unknown')}"
            claims_text += f"[{claim.get('claim_id')}] {claim.get('claim_text') or claim.get('text')} ({source_info})\n"

        prompt = (
            f"Tu es un expert rédacteur technique. Tu dois rédiger un chapitre complet d'un livre de référence.\n\n"
            f"TITRE DU CHAPITRE : {chapter_title}\n"
            f"OBJECTIF : {chapter_goal}\n\n"
            f"PLAN IMPÉRATIF DES SOUS-SECTIONS (Respecte cet ordre) :\n"
            + "\n".join([f"- {sub}" for sub in sub_sections]) + "\n\n"
            f"BASE DE CONNAISSANCE (FAITS VALIDÉS) :\n"
            f"{claims_text}\n\n"
            f"CONSIGNES DE RÉDACTION :\n"
            f"1. Rédige un contenu DENSE, structuré et très technique.\n"
            f"2. Utilise le format Markdown. Utilise des titres de niveau 3 (###) pour les sous-sections demandées.\n"
            f"3. NE METS PAS de titre de niveau 1 (#) ou 2 (##) pour le titre du chapitre (ce sera fait automatiquement).\n"
            f"4. Cite tes sources en utilisant les ID de claims entre crochets, ex: [CLM-123ab]. C'est CRITIQUE.\n"
            f"5. Si tu manques d'information pour une sous-section, indique-le honnêtement mais synthétise ce que tu sais.\n"
            f"6. Fais des transitions fluides.\n"
            f"7. Écris en Français professionnel.\n"
            f"8. Longueur cible : Environ 800 à 1500 mots pour ce chapitre.\n"
        )

        content = await self._ask_text_model(
            model=self.config.writer_model,
            system_prompt="Tu es un rédacteur de livres techniques de haute précision.",
            user_prompt=prompt,
            llm_logs=llm_logs,
            stage="writing-chapter",
        )

        return {
            "id": chapter.get("id"),
            "title": chapter_title,
            "markdown": content,
            "claims_used": [c.get("claim_id") for c in selection],
        }

    async def _assemble_report(
        self,
        planner: dict[str, Any],
        outline_seed: dict[str, Any],
        corpus: dict[str, Any],
        claims_payload: dict[str, Any],
        verdicts_payload: dict[str, Any],
        sections_payload: dict[str, Any],
        llm_logs: list[dict[str, Any]],
    ) -> str:
        sections = sections_payload.get("sections", [])
        verdict_stats = verdicts_payload.get("stats", {})
        source_count = len(corpus.get("sources", []))
        claim_count = len(claims_payload.get("claims", []))
        
        master_outline = planner.get("master_outline", [])
        if not isinstance(master_outline, list):
            master_outline = []

        # 1. Generate Executive Summary
        summary_prompt = (
            "You are the finalizer of a research dossier. Write a concise executive summary in French. "
            "Focus on the key findings and conclusions."
        )
        # Use first 3 chapters intro for summary context to avoid token overflow
        summary_context = "\n".join([s.get("markdown", "")[:800] for s in sections[:3]])
        
        summary_user = json.dumps(
            {
                "question": planner.get("question_reformulated", ""),
                "context_extract": summary_context,
                "stats": {"sources": source_count, "claims": claim_count},
            },
            ensure_ascii=False,
        )
        executive_summary = await self._ask_text_model(
            model=self.config.writer_model,
            system_prompt=summary_prompt,
            user_prompt=summary_user,
            llm_logs=llm_logs,
            stage="finalizer",
        )

        # 2. Build Table of Contents (Nested) & Body
        book_body: list[str] = []
        toc_lines: list[str] = ["- [1. Résumé Exécutif](#1-resume-executif)"]
        
        # Start numbering chapters at 2
        chapter_num = 2
        
        # Map sections by ID for easy access
        sections_map = {str(s.get("id")): s for s in sections}

        for chapter in master_outline:
            c_id = str(chapter.get("id") or "").strip()
            c_title = str(chapter.get("title") or "Chapitre").strip()
            # Clean title
            c_title = re.sub(r"^\d+(\.\d+)*\s*", "", c_title)
            c_subsections = chapter.get("sub_sections", [])
            
            # TOC Entry: Chapter
            anchor_chap = self._markdown_anchor(f"{chapter_num} {c_title}")
            toc_lines.append(f"- [{chapter_num}. {c_title}](#{anchor_chap})")
            
            # TOC Entries: Sub-sections
            for sub_idx, sub in enumerate(c_subsections, 1):
                sub_clean = re.sub(r"^\d+(\.\d+)*\s*", "", str(sub))
                toc_lines.append(f"  - {chapter_num}.{sub_idx}. {sub_clean}")

            # Body: Header
            book_body.append(f"## {chapter_num}. {c_title}")
            if chapter.get("goal"):
                book_body.append(f"> *Objectif : {chapter.get('goal')}*")
            book_body.append("")

            # Body: Content
            section_data = sections_map.get(c_id)
            if section_data:
                raw_content = section_data.get("markdown", "").strip()
                
                # 1. Clean Title: Remove the first line if it duplicates the title (H1 or H2)
                raw_content = re.sub(r"^#+\s*" + re.escape(c_title) + r".*?\n", "", raw_content, flags=re.IGNORECASE).strip()
                
                # 2. Add Links: Convert [CLM-xxx] to links
                raw_content = re.sub(r"\[(CLM-[a-f0-9]+)\](?!\()", r"[\1](#\1)", raw_content)

                # 3. Apply Visual Indentation (Stair-step effect)
                # Logic:
                # - Intro text (before first ###) -> Indent 1 level ("> ")
                # - Sub-sections (### and following text) -> Indent 2 levels (">> ")
                
                lines = raw_content.split('\n')
                processed_lines = []
                current_level = 1 # Start at level 1 (Intro)
                
                for line in lines:
                    stripped = line.strip()
                    
                    # Detect Sub-section header
                    if stripped.startswith("### "):
                        current_level = 2
                        processed_lines.append("") # Spacing
                        processed_lines.append(f">> {line}") # Indent header
                        continue
                        
                    # Apply indentation based on level
                    if stripped == "":
                        # Empty lines need the quote char to maintain the block
                        prefix = ">" if current_level == 1 else ">>"
                        processed_lines.append(prefix)
                    else:
                        prefix = "> " if current_level == 1 else ">> "
                        processed_lines.append(f"{prefix}{line}")

                formatted_content = "\n".join(processed_lines)
                book_body.append(formatted_content)
            else:
                book_body.append("*(Contenu non disponible)*")
            
            book_body.append("\n---\n") # Page break equivalent
            chapter_num += 1

        # 3. Build Annexes
        toc_lines.append(f"- [{chapter_num}. Annexes Techniques](#{self._markdown_anchor(f'{chapter_num} Annexes Techniques')})")
        
        # Enrich claims with verdicts
        all_verdicts = {v.get("claim_id"): v for v in verdicts_payload.get("verdicts", [])}
        enriched_claims = []
        for claim in claims_payload.get("claims", []):
            c = dict(claim)
            v = all_verdicts.get(c.get("claim_id"))
            if v:
                c["status"] = v.get("status", "UNCERTAIN")
            else:
                c["status"] = "UNCERTAIN"
            enriched_claims.append(c)

        # Accepted Claims Index
        accepted_claims = [c for c in enriched_claims if c.get("status") == "ACCEPTED"]
        accepted_lines = []
        source_by_id = {source["source_id"]: source for source in corpus.get("sources", [])}
        
        for claim in accepted_claims:
            cid = claim.get("claim_id")
            text = str(claim.get("text") or claim.get("claim_text") or "").replace("\n", " ").strip()
            sid = claim.get("source_id")
            source_url = source_by_id.get(sid, {}).get("canonical_url", "")
            source_link = f"[{sid}]({source_url})" if source_url else f"`{sid}`"
            accepted_lines.append(f"- <a name=\"{cid}\"></a>**{cid}** : {text} (Source: {source_link})")
            
        accepted_md = "\n".join(accepted_lines) if accepted_lines else "*Aucun fait validé.*"

        rejected_rows = [c for c in enriched_claims if c.get("status") == "REJECTED"]
        uncertain_rows = [c for c in enriched_claims if c.get("status") == "UNCERTAIN"]

        rejected_md = self._claims_table_markdown(rejected_rows[:150], title="Faits Rejetés")
        uncertain_md = self._claims_table_markdown(uncertain_rows[:150], title="Faits Incertains")

        sources_annex = "\n".join(
            [
                f"- `{source.get('source_id')}` [{source.get('title') or source.get('canonical_url')}]({source.get('canonical_url')}) "
                f"- domaine: `{source.get('domain')}`"
                for source in corpus.get("sources", [])
            ]
        )
        
        methodology_md = (
            f"### Méthodologie et Périmètre\n\n"
            f"**Question Initiale** : {planner.get('question_reformulated', '')}\n\n"
            f"**Note de Portée** : {planner.get('scope_notes', '')}\n\n"
            f"**Statistiques** :\n"
            f"- Sources analysées : {source_count}\n"
            f"- Faits extraits : {claim_count}\n"
            f"- Faits validés : {verdict_stats.get('ACCEPTED', 0)}\n"
        )
        
        annexes_body = (
            f"## {chapter_num}. Annexes Techniques\n\n"
            f"{methodology_md}\n\n"
            f"### Index des Faits Validés (Cités)\n{accepted_md}\n\n"
            f"### Sources Consultées\n{sources_annex}\n\n"
            f"{uncertain_md}\n\n"
            f"{rejected_md}\n"
        )

        # 4. Final Assembly
        toc_md = "\n".join(toc_lines)
        body_md = "\n".join(book_body)
        
        report = (
            f"# {planner.get('question_reformulated', 'Dossier de Recherche')}\n\n"
            f"## Table des matières\n\n{toc_md}\n\n"
            f"## 1. Résumé Exécutif\n\n{executive_summary.strip()}\n\n"
            f"{body_md}\n"
            f"{annexes_body}"
        )
        return report
    async def _draft_chapter_content(
        self,
        chapter: dict[str, Any],
        claims: list[dict[str, Any]],
        llm_logs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        chapter_title = str(chapter.get("title") or "Chapitre Sans Titre")
        chapter_goal = str(chapter.get("goal") or "")
        sub_sections = chapter.get("sub_sections", [])
        
        # Format claims
        claims_text = ""
        accepted_claims = [c for c in claims if c.get("status") == "ACCEPTED"]
        # Selection logic (Top 60)
        selection = accepted_claims[:50]
        if len(selection) < 20:
            uncertain = [c for c in claims if c.get("status") == "UNCERTAIN"]
            selection.extend(uncertain[:(30 - len(selection))])
            
        for i, claim in enumerate(selection, 1):
            source_info = f"Source: {claim.get('source_title', 'Unknown')}"
            claims_text += f"[{claim.get('claim_id')}] {claim.get('claim_text') or claim.get('text')} ({source_info})\n"

        prompt = (
            f"Tu es un expert rédacteur technique. Tu rédiges le chapitre '{chapter_title}' d'un livre de référence.\n"

            f"OBJECTIF DU CHAPITRE : {chapter_goal}\n\n"
            f"PLAN STRICT DES SOUS-SECTIONS (Utilise ces titres exacts en H3 '###') :\n"
            + "\n".join([f"- {sub}" for sub in sub_sections]) + "\n\n"
            f"BASE DE CONNAISSANCE (Citations obligatoires) :\n"
            f"{claims_text}\n\n"
            f"RÈGLES DE FORMATAGE (CRITIQUES) :\n"

                                f"1. NE METS PAS LE TITRE DU CHAPITRE (ni # ni ##) au début. Commence directement par une phrase d'intro ou la première sous-section.\n"

                                f"2. Utilise '### Titre Sous-Section' pour chaque point du plan.\n"

                                f"3. Le texte doit être dense, technique et fluide.\n"

                                f"4. Cite tes sources [CLM-xxx] à chaque affirmation.\n"

                                f"5. Écris en Français.\n"

                            )

        content = await self._ask_text_model(
            model=self.config.writer_model,
            system_prompt="Tu es un rédacteur technique de précision.",
            user_prompt=prompt,
            llm_logs=llm_logs,
            stage="writing-chapter",
        )

        return {
            "id": chapter.get("id"),
            "title": chapter_title,
            "markdown": content,
            "claims_used": [c.get("claim_id") for c in selection],
        }
        accepted = [row for row in claim_rows if row.get("status") == "ACCEPTED"]
        uncertain = [row for row in claim_rows if row.get("status") == "UNCERTAIN"]
        rejected = [row for row in claim_rows if row.get("status") == "REJECTED"]
        def confidence_value(row: dict[str, Any]) -> float:
            try:
                return float(row.get("confidence", 0.0))
            except Exception:  # noqa: BLE001
                return 0.0
        ranked_claims = sorted(
            [*accepted, *uncertain, *rejected],
            key=confidence_value,
            reverse=True,
        )
        if not ranked_claims:
            raise RuntimeError(
                f"Writer cannot draft sub-question {sub_question.get('id')}: no claim rows available"
            )

        sq_id = str(sub_question.get("id") or "SQ")
        sq_question = str(sub_question.get("question") or "")
        planner_topics: list[dict[str, str]] = []
        planner_chapters: list[dict[str, Any]] = []
        for raw_chapter in planned_chapters if isinstance(planned_chapters, list) else []:
            if not isinstance(raw_chapter, dict):
                continue
            chapter_title = str(raw_chapter.get("title") or "").strip()
            if not chapter_title:
                continue
            chapter_goal = str(raw_chapter.get("goal") or "").strip()
            chapter_sub_sections = self._normalize_subsections(
                raw_chapter.get("sub_sections", []),
                chapter_title=chapter_title,
            )
            planner_chapters.append(
                {
                    "id": str(raw_chapter.get("id") or "").strip(),
                    "title": chapter_title,
                    "goal": chapter_goal,
                    "sub_sections": chapter_sub_sections,
                }
            )
            planner_topics.append(
                {
                    "title": chapter_title,
                    "why_relevant": chapter_goal or "Chapitre planifie par le sommaire maitre.",
                    "viewpoint": "both",
                }
            )

        initial_topics: list[dict[str, str]] = []
        for raw_topic in topic_seed.get("topics", []) if isinstance(topic_seed, dict) else []:
            normalized = self._normalize_topic(raw_topic)
            if normalized:
                initial_topics.append(normalized)

        if planner_topics:
            initial_topics = self._dedupe_topics([*planner_topics, *initial_topics], max_items=24)
        if not initial_topics:
            initial_topics = [
                {
                    "title": sq_question or sq_id,
                    "why_relevant": "Topic derive directement de la sous-question faute de topic seed.",
                    "viewpoint": "both",
                }
            ]
        initial_topics = self._dedupe_topics(initial_topics, max_items=24)

        topic_queue = list(initial_topics)
        seen_topic_titles = {topic["title"].casefold() for topic in initial_topics}
        added_topics: list[dict[str, str]] = []
        rejected_topics: list[dict[str, str]] = []
        writer_warnings: list[str] = []
        for raw_topic in topic_seed.get("rejected_topics", []) if isinstance(topic_seed, dict) else []:
            title = str(raw_topic.get("title") or "").strip() if isinstance(raw_topic, dict) else str(raw_topic).strip()
            if not title:
                continue
            reason = str(raw_topic.get("reason") or "").strip() if isinstance(raw_topic, dict) else ""
            rejected_topics.append({"title": title, "reason": reason or "Hors perimetre detecte au sommaire."})

        topic_target_words = max(
            260,
            int(self.config.writer_target_words_per_section / max(1, len(initial_topics))),
        )
        max_topics = max(10, len(initial_topics) + 16)
        per_topic_iterations = min(2, max(1, self.config.writer_iterations))
        chapters: list[str] = []
        topic_index = 0
        while topic_index < len(topic_queue) and topic_index < max_topics:
            topic = topic_queue[topic_index]
            topic_index += 1
            chapter_parts: list[str] = []
            for iteration in range(1, per_topic_iterations + 1):
                claim_batch = self._pick_claims_for_topic(
                    topic_title=topic["title"],
                    claim_rows=ranked_claims,
                    limit=self.config.writer_batch_claims,
                    offset=(iteration - 1) * self.config.writer_batch_claims,
                )
                if not claim_batch:
                    claim_batch = ranked_claims[: self.config.writer_batch_claims]
                if not claim_batch:
                    break

                outline_hint = self._pick_best_planned_chapter_for_topic(
                    topic_title=topic["title"],
                    planned_chapters=planner_chapters,
                )
                writer_payload = {
                    "sub_question": {
                        "id": sq_id,
                        "question": sq_question,
                    },
                    "topic": topic,
                    "chapter_outline_hint": outline_hint,
                    "iteration": iteration,
                    "existing_chapter_excerpt": "\n\n".join(chapter_parts)[-4000:],
                    "claim_batch": claim_batch,
                    "constraints": {
                        "language": "fr",
                        "no_new_facts": True,
                        "must_cite_claim_ids_inline": True,
                        "cover_positive_and_negative_points": True,
                        "follow_chapter_outline_hint_if_available": True,
                        "focus": [
                            "benefices et points forts",
                            "limites et risques",
                            "configuration et mise en oeuvre",
                            "exemples concrets",
                            "contradictions et incertitudes",
                        ],
                    },
                    "output_schema": {
                        "chapter_markdown": "string",
                        "new_related_topics": [
                            {"title": "string", "reason": "string", "viewpoint": "benefits|limitations|both"}
                        ],
                        "out_of_scope_topics": [{"title": "string", "reason": "string"}],
                    },
                }
                response = await self._ask_text_model(
                    model=self.config.writer_model,
                    system_prompt=(
                        "You write long-form research chapters in French from verified claims only. "
                        "When chapter_outline_hint is provided, align the structure with its sub_sections. "
                        "Return strict JSON only. "
                        "Do not invent facts and do not omit negative findings."
                    ),
                    user_prompt=json.dumps(writer_payload, ensure_ascii=False),
                    llm_logs=llm_logs,
                    stage=f"writer-{sq_id}",
                )
                parsed = await self._parse_or_repair_json(
                    raw_text=response,
                    model=self.config.writer_model,
                    stage=f"writer-{sq_id}",
                    llm_logs=llm_logs,
                    expected_type="dict",
                    schema_hint=writer_payload["output_schema"],
                )

                chapter_markdown = str(parsed.get("chapter_markdown") or "").strip()
                if len(chapter_markdown) < 180:
                    chapter_markdown = await self._repair_short_writer_chapter(
                        sq_id=sq_id,
                        sq_question=sq_question,
                        topic=topic,
                        claim_batch=claim_batch,
                        current_text=chapter_markdown,
                        llm_logs=llm_logs,
                        stage_label=f"writer-{sq_id}-retry",
                    )
                if len(chapter_markdown) < 180:
                    writer_warnings.append(
                        f"Chapitre trop court ignore pour le topic '{topic['title']}' (iteration {iteration})."
                    )
                    continue
                chapter_markdown = self._sanitize_writer_markdown(chapter_markdown)
                if self._is_duplicate_markdown_block(chapter_markdown, chapter_parts):
                    writer_warnings.append(
                        f"Bloc duplique ignore pour le topic '{topic['title']}' (iteration {iteration})."
                    )
                    continue
                chapter_parts.append(chapter_markdown)

                for raw_new in parsed.get("new_related_topics", []) if isinstance(parsed.get("new_related_topics", []), list) else []:
                    normalized_new = self._normalize_topic(raw_new)
                    if not normalized_new:
                        continue
                    key = normalized_new["title"].casefold()
                    if key in seen_topic_titles:
                        continue
                    seen_topic_titles.add(key)
                    topic_queue.append(normalized_new)
                    added_topics.append(normalized_new)

                for raw_rejected in parsed.get("out_of_scope_topics", []) if isinstance(parsed.get("out_of_scope_topics", []), list) else []:
                    if isinstance(raw_rejected, dict):
                        title = str(raw_rejected.get("title") or "").strip()
                        reason = str(raw_rejected.get("reason") or "").strip()
                    else:
                        title = str(raw_rejected).strip()
                        reason = ""
                    if not title:
                        continue
                    rejected_topics.append(
                        {
                            "title": title,
                            "reason": reason or "Hors du sujet principal.",
                        }
                    )

                if self._word_count("\n\n".join(chapter_parts)) >= topic_target_words:
                    break

            if chapter_parts:
                topic_block = self._dedupe_markdown_paragraphs("\n\n".join(chapter_parts))
                if topic_block.strip():
                    chapters.append(f"### {topic['title']}\n\n{topic_block}")

        development_markdown = self._dedupe_markdown_paragraphs("\n\n".join(chapters))
        if not development_markdown.strip():
            raise RuntimeError(f"Writer produced an empty detailed section for sub-question {sq_id}")

        effective_min_words = min(
            self.config.writer_min_words_per_section,
            max(350, int(85 * len(ranked_claims))),
        )
        while self._word_count(development_markdown) < effective_min_words:
            if len(chapters) >= self.config.writer_iterations + max(2, len(initial_topics)):
                break
            extension_payload = {
                "sub_question": {"id": sq_id, "question": sq_question},
                "planner_outline_hints": planner_chapters[:6],
                "existing_development_excerpt": development_markdown[-6000:],
                "remaining_claims": ranked_claims[: self.config.writer_batch_claims],
                "constraints": {
                    "language": "fr",
                    "no_new_facts": True,
                    "expand_with_concrete_detail": True,
                    "must_include_limits_and_risks": True,
                    "must_include_configuration_or_usage_examples_if_claims_allow": True,
                    "follow_planner_outline_if_available": True,
                },
                "output_schema": {"chapter_markdown": "string"},
            }
            extension_response = await self._ask_text_model(
                model=self.config.writer_model,
                system_prompt=(
                    "Expand a technical dossier section with concrete detail while staying strictly within provided claims. "
                    "When available, use planner_outline_hints as the target chapter/subchapter map. "
                    "Return strict JSON only."
                ),
                user_prompt=json.dumps(extension_payload, ensure_ascii=False),
                llm_logs=llm_logs,
                stage=f"writer-{sq_id}-expand",
            )
            parsed_extension = await self._parse_or_repair_json(
                raw_text=extension_response,
                model=self.config.writer_model,
                stage=f"writer-{sq_id}-expand",
                llm_logs=llm_logs,
                expected_type="dict",
                schema_hint=extension_payload["output_schema"],
            )
            added_markdown = str(parsed_extension.get("chapter_markdown") or "").strip()
            if len(added_markdown) < 160:
                added_markdown = await self._repair_short_writer_chapter(
                    sq_id=sq_id,
                    sq_question=sq_question,
                    topic={"title": "Extension", "why_relevant": "Extension de section", "viewpoint": "both"},
                    claim_batch=self._pick_claims_for_topic(
                        topic_title=sq_question,
                        claim_rows=ranked_claims,
                        limit=self.config.writer_batch_claims,
                    ),
                    current_text=added_markdown,
                    llm_logs=llm_logs,
                    stage_label=f"writer-{sq_id}-expand-retry",
                )
            if len(added_markdown) < 160:
                writer_warnings.append(
                    f"Extension trop courte ignoree pour la sous-question {sq_id}."
                )
                break
            added_markdown = self._sanitize_writer_markdown(added_markdown)
            if self._is_duplicate_markdown_block(added_markdown, chapters):
                writer_warnings.append(
                    f"Extension ignoree car tres proche d'un texte deja redige ({sq_id})."
                )
                break
            chapters.append(f"### Complements techniques {len(chapters) + 1}\n\n{added_markdown}")
            development_markdown = self._dedupe_markdown_paragraphs("\n\n".join(chapters))

        minimum_acceptable = max(240, int(effective_min_words * 0.7))
        if self._word_count(development_markdown) < minimum_acceptable:
            explicit_digest = self._build_explicit_claim_digest(
                accepted=accepted,
                uncertain=uncertain,
                rejected=rejected,
            )
            if explicit_digest:
                writer_warnings.append(
                    "Section completee par digest explicite des claims (writer insuffisant)."
                )
                development_markdown = (
                    f"{development_markdown}\n\n"
                    f"#### Complements automatiques explicites\n"
                    f"{explicit_digest}"
                )
                development_markdown = self._dedupe_markdown_paragraphs(development_markdown)

        development_markdown = self._sanitize_writer_markdown(development_markdown)
        development_markdown = self._dedupe_markdown_paragraphs(development_markdown)

        short_lines: list[str] = []
        if accepted:
            short_lines.append(
                "Les preuves les plus solides convergent sur les points suivants:"
            )
            for row in accepted[:5]:
                short_lines.append(f"- `{row.get('claim_id')}` {str(row.get('text') or '')[:220]}")
        else:
            short_lines.append(
                "Aucun claim n'est suffisamment corroboré en statut ACCEPTED sur cette sous-question."
            )
        if uncertain:
            short_lines.append(
                "Les points ci-dessous restent incertains et doivent etre lus avec prudence:"
            )
            for row in uncertain[:3]:
                short_lines.append(f"- `{row.get('claim_id')}` {str(row.get('text') or '')[:200]}")
        short_answer_markdown = "\n".join(short_lines)

        evidence_table = self._claims_table_markdown(
            [*accepted[:25], *uncertain[:15]],
            title="Tableau preuves / contradictions",
        )
        rejected_lines = (
            "\n".join(f"- `{row.get('claim_id')}` {str(row.get('text') or '')[:220]}" for row in rejected[:16])
            or "- Aucun claim rejete."
        )
        uncertain_lines = (
            "\n".join(
                f"- `{row.get('claim_id')}` confiance={float(row.get('confidence', 0.0)):.2f} - {str(row.get('text') or '')[:220]}"
                for row in uncertain[:16]
            )
            or "- Aucun claim incertain."
        )
        added_topics = self._dedupe_topics(added_topics, max_items=32)
        rejected_topics = self._dedupe_rejected_topics(rejected_topics, max_items=48)
        added_topics_md = (
            "\n".join(
                f"- **{topic.get('title')}** - {topic.get('why_relevant') or 'Ajoute pendant la redaction.'}"
                for topic in added_topics
            )
            or "- Aucun."
        )
        rejected_topics_md = (
            "\n".join(
                f"- **{topic.get('title')}** - {topic.get('reason') or 'Hors perimetre.'}"
                for topic in rejected_topics
            )
            or "- Aucun."
        )
        warnings_md = "\n".join(f"- {item}" for item in writer_warnings) or "- Aucune."
        topic_plan_md = (
            "\n".join(
                f"- **{topic.get('title')}** - {topic.get('why_relevant') or 'Topic principal'}"
                for topic in initial_topics
            )
            or "- Aucun."
        )
        planner_outline_md = self._planned_chapters_markdown(planner_chapters)
        section_markdown = (
            f"### {sq_id} - {sq_question}\n\n"
            f"#### Sommaire cible du planner\n{planner_outline_md}\n\n"
            f"#### Sommaire de la sous-question\n{topic_plan_md}\n\n"
            f"#### Reponse courte\n{short_answer_markdown}\n\n"
            f"#### Developpement detaille\n{development_markdown}\n\n"
            f"{evidence_table}\n\n"
            f"#### Incertitudes\n{uncertain_lines}\n\n"
            f"#### Claims rejetes (resume)\n{rejected_lines}\n\n"
            f"#### Sujets ajoutes pendant l'analyse\n{added_topics_md}\n\n"
            f"#### Sujets ejectes (hors perimetre)\n{rejected_topics_md}\n\n"
            f"#### Alertes de redaction\n{warnings_md}"
        )

        return {
            "markdown": section_markdown,
            "topic_plan_initial": initial_topics,
            "topic_plan_added": added_topics,
            "topic_plan_rejected": rejected_topics,
            "writer_warnings": writer_warnings,
            "chapter_count": len(chapters),
            "word_count": self._word_count(section_markdown),
        }

    async def _repair_short_writer_chapter(
        self,
        sq_id: str,
        sq_question: str,
        topic: dict[str, Any],
        claim_batch: list[dict[str, Any]],
        current_text: str,
        llm_logs: list[dict[str, Any]],
        stage_label: str,
    ) -> str:
        if not claim_batch:
            return ""

        repair_payload = {
            "sub_question": {"id": sq_id, "question": sq_question},
            "topic": topic,
            "too_short_text": str(current_text or "")[:3000],
            "claim_batch": claim_batch,
            "constraints": {
                "language": "fr",
                "no_new_facts": True,
                "must_cite_claim_ids_inline": True,
                "minimum_words": 180,
                "cover_positive_and_negative_points": True,
            },
            "output_schema": {"chapter_markdown": "string"},
        }
        repaired_text = await self._ask_text_model(
            model=self.config.writer_model,
            system_prompt=(
                "Rewrite the draft as a richer chapter using only provided claims. "
                "Return strict JSON only."
            ),
            user_prompt=json.dumps(repair_payload, ensure_ascii=False),
            llm_logs=llm_logs,
            stage=stage_label,
        )
        repaired = await self._parse_or_repair_json(
            raw_text=repaired_text,
            model=self.config.writer_model,
            stage=stage_label,
            llm_logs=llm_logs,
            expected_type="dict",
            schema_hint=repair_payload["output_schema"],
        )
        return str(repaired.get("chapter_markdown") or "").strip()

    def _build_explicit_claim_digest(
        self,
        accepted: list[dict[str, Any]],
        uncertain: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
    ) -> str:
        def _safe_confidence(row: dict[str, Any]) -> float:
            try:
                return float(row.get("confidence", 0.0))
            except Exception:  # noqa: BLE001
                return 0.0

        lines: list[str] = []
        lines.append("##### Points corroborés")
        if accepted:
            for row in accepted[:20]:
                lines.append(
                    f"- `{row.get('claim_id')}` (confiance={_safe_confidence(row):.2f}) {str(row.get('text') or '')[:320]}"
                )
        else:
            lines.append("- Aucun claim ACCEPTED disponible.")

        lines.append("\n##### Points incertains")
        if uncertain:
            for row in uncertain[:20]:
                lines.append(
                    f"- `{row.get('claim_id')}` (confiance={_safe_confidence(row):.2f}) {str(row.get('text') or '')[:320]}"
                )
        else:
            lines.append("- Aucun claim UNCERTAIN.")

        lines.append("\n##### Points rejetés")
        if rejected:
            for row in rejected[:20]:
                lines.append(
                    f"- `{row.get('claim_id')}` (confiance={_safe_confidence(row):.2f}) {str(row.get('text') or '')[:320]}"
                )
        else:
            lines.append("- Aucun claim REJECTED.")

        return "\n".join(lines)

    async def _extract_claims_from_source(
        self,
        sub_question: dict[str, Any],
        source: dict[str, Any],
        llm_logs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        chunks = source.get("chunks", [])[:3]
        if not chunks:
            raise RuntimeError(
                f"Claims extraction failed for source {source.get('source_id')}: no text chunks available"
            )
        claims: list[dict[str, Any]] = []

        for chunk in chunks:
            prompt_payload = {
                "sub_question": sub_question,
                "source": {
                    "source_id": source.get("source_id"),
                    "title": source.get("title"),
                    "url": source.get("canonical_url"),
                    "domain": source.get("domain"),
                    "published_at": source.get("published_at"),
                },
                "chunk": {
                    "chunk_id": chunk.get("chunk_id"),
                    "text": chunk.get("text", "")[:3500],
                },
                "format": {
                    "claims": [
                        {
                            "text": "string",
                            "type": "fact|number|causal|opinion",
                            "evidence_quote": "short quote from chunk",
                        }
                    ]
                },
            }

            response_text = await self._ask_text_model(
                model=self.config.extract_model,
                system_prompt=(
                    "Extract atomic verifiable claims from the chunk. "
                    "Return JSON only with key 'claims'."
                ),
                user_prompt=json.dumps(prompt_payload, ensure_ascii=False),
                llm_logs=llm_logs,
                stage="claims",
            )
            parsed = await self._parse_or_repair_json(
                raw_text=response_text,
                model=self.config.extract_model,
                stage="claims",
                llm_logs=llm_logs,
                expected_type="dict",
                schema_hint={"claims": [{"text": "string", "type": "fact|number|causal|opinion", "evidence_quote": "string"}]},
            )
            raw_claims = parsed.get("claims", [])
            if not isinstance(raw_claims, list):
                raise RuntimeError(
                    "Claims model JSON missing list 'claims' "
                    f"for source {source.get('source_id')} chunk {chunk.get('chunk_id')}"
                )

            for item in raw_claims[: self.config.max_claims_per_source]:
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                claims.append(
                    {
                        "claim_id": f"CLM-{uuid.uuid4().hex[:12]}",
                        "sub_question_id": sub_question.get("id"),
                        "sub_question": sub_question.get("question"),
                        "source_id": source.get("source_id"),
                        "chunk_id": chunk.get("chunk_id"),
                        "claim_text": text,
                        "claim_type": str(item.get("type") or "fact"),
                        "evidence_quote": str(item.get("evidence_quote") or "")[:500],
                    }
                )

        return claims

    async def _verify_single_claim(
        self,
        claim: dict[str, Any],
        related_claims: list[dict[str, Any]],
        llm_logs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        verify_payload = {
            "claim": claim,
            "related_claims": related_claims[:8],
            "expected_output": {
                "status": "ACCEPTED|UNCERTAIN|REJECTED",
                "confidence": "0..1",
                "justification": "short explanation",
                "supporting_claim_ids": ["..."],
                "contradicting_claim_ids": ["..."],
            },
        }

        response = await self._ask_text_model(
            model=self.config.verify_model,
            system_prompt=(
                "You are a strict verifier. Assess whether the claim is corroborated, uncertain, or contradicted. "
                "Return JSON only."
            ),
            user_prompt=json.dumps(verify_payload, ensure_ascii=False),
            llm_logs=llm_logs,
            stage="verification",
        )
        parsed = await self._parse_or_repair_json(
            raw_text=response,
            model=self.config.verify_model,
            stage="verification",
            llm_logs=llm_logs,
            expected_type="dict",
            schema_hint=verify_payload["expected_output"],
        )

        status = str(parsed.get("status") or "UNCERTAIN").upper().strip()
        if status not in {"ACCEPTED", "UNCERTAIN", "REJECTED"}:
            status = "UNCERTAIN"

        confidence = parsed.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except Exception:  # noqa: BLE001
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        supporting_ids = parsed.get("supporting_claim_ids", [])
        contradicting_ids = parsed.get("contradicting_claim_ids", [])
        if not isinstance(supporting_ids, list):
            supporting_ids = []
        if not isinstance(contradicting_ids, list):
            contradicting_ids = []

        if not supporting_ids and status == "ACCEPTED":
            status = "UNCERTAIN"
            confidence = min(confidence, 0.49)

        return {
            "claim_id": claim.get("claim_id"),
            "status": status,
            "confidence": confidence,
            "justification": str(parsed.get("justification") or "")[:1000],
            "supporting_claim_ids": [str(v) for v in supporting_ids[:8]],
            "contradicting_claim_ids": [str(v) for v in contradicting_ids[:8]],
        }

    def _find_related_claims(self, target_claim: dict[str, Any], all_claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
        target_id = target_claim.get("claim_id")
        target_tokens = set(self._simple_tokens(target_claim.get("claim_text", "")))
        if not target_tokens:
            return []

        related: list[tuple[float, dict[str, Any]]] = []
        for claim in all_claims:
            if claim.get("claim_id") == target_id:
                continue
            if claim.get("sub_question_id") != target_claim.get("sub_question_id"):
                continue
            other_tokens = set(self._simple_tokens(claim.get("claim_text", "")))
            if not other_tokens:
                continue
            overlap = len(target_tokens & other_tokens) / max(len(target_tokens | other_tokens), 1)
            if overlap < 0.10:
                continue
            related.append(
                (
                    overlap,
                    {
                        "claim_id": claim.get("claim_id"),
                        "text": claim.get("claim_text"),
                        "source_id": claim.get("source_id"),
                        "type": claim.get("claim_type"),
                    },
                )
            )

        related.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in related[:10]]

    async def _llm_rerank_sources(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        llm_logs: list[dict[str, Any]],
        progress_cb: ProgressCallback | None = None,
        run_dir: Path | None = None,
        context_label: str = "",
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        all_ranked: list[dict[str, Any]] = []
        batch_size = 25
        
        # Process in batches to avoid context overflow
        for i in range(0, len(candidates), batch_size):
            if progress_cb and run_dir:
                label = f" ({context_label})" if context_label else ""
                await self._emit(progress_cb, run_dir, "ranking", f"Ranking sources{label} {i}/{len(candidates)}")
            
            batch = candidates[i : i + batch_size]
            
            payload = {
                "question": question,
                "sources": [
                    {
                        "source_id": item.get("source_id"),
                        "title": item.get("title"),
                        "domain": item.get("domain"),
                        "published_at": item.get("published_at"),
                        "url": item.get("url"),
                    }
                    for item in batch
                ],
                "output": {
                    "ranked": [
                        {
                            "source_id": "...",
                            "score": "0..1",
                            "reason": "brief reason",
                        }
                    ]
                },
            }

            try:
                response = await self._ask_text_model(
                    model=self.config.judge_model,
                    system_prompt=(
                        "Rank sources for factual usefulness and reliability for answering the question. "
                        "Return strict JSON only."
                    ),
                    user_prompt=json.dumps(payload, ensure_ascii=False),
                    llm_logs=llm_logs,
                    stage="ranking",
                )

                parsed = await self._parse_or_repair_json(
                    raw_text=response,
                    model=self.config.judge_model,
                    stage="ranking",
                    llm_logs=llm_logs,
                    expected_type="dict",
                    schema_hint=payload["output"],
                )
                
                ranked = parsed.get("ranked", [])
                if isinstance(ranked, list):
                    for item in ranked:
                        source_id = str(item.get("source_id") or "")
                        if not source_id: continue
                        try: score = float(item.get("score", 0.0))
                        except Exception: score = 0.0
                        all_ranked.append({
                            "source_id": source_id,
                            "score": max(0.0, min(1.0, score)),
                            "reason": str(item.get("reason") or "")[:200],
                        })
            except Exception as e:
                print(f"Warning: Batch ranking failed: {e}")
                # Fallback: assume neutral score for failed batch items
                for item in batch:
                    all_ranked.append({"source_id": item.get("source_id"), "score": 0.5, "reason": "Batch failed"})

        if not all_ranked:
            # If everything failed, return neutral scores for everyone rather than crashing
            print("Warning: All ranking batches failed. Using fallback scores.")
            return [{"source_id": c["source_id"], "score": 0.5, "reason": "Ranking failure"} for c in candidates]
            
        return all_ranked

    async def _fetch_source(self, url: str, run_dir: Path) -> dict[str, Any] | None:
        timeout = httpx.Timeout(self.config.web_timeout_seconds)
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 CodexDossier/1.0",
            "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
        }

        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
                response = await client.get(url)
        except Exception:  # noqa: BLE001
            return None

        if response.status_code >= 400:
            return None

        try:
            content_type = str(response.headers.get("content-type") or "").lower()
            final_url = self._canonicalize_url(str(response.url))
            if not final_url:
                return None

            source_hash = hashlib.sha1(final_url.encode("utf-8")).hexdigest()[:14]
            source_id = f"SRC-{source_hash}"

            text = ""
            title = ""
            author = ""
            language = ""
            published_at = ""
            raw_suffix = "txt"

            body = response.content
            if "application/pdf" in content_type or final_url.lower().endswith(".pdf"):
                raw_suffix = "pdf"
                text, title, author, published_at = self._extract_pdf(body)
            else:
                raw_suffix = "html"
                text, title, author, language, published_at = self._extract_html(response.text)

            text = self._clean_text(text)
            if len(text) < 200:
                return None

            raw_path = run_dir / "raw" / f"{source_id}.{raw_suffix}"
            clean_path = run_dir / "clean" / f"{source_id}.txt"

            try:
                raw_path.write_bytes(body[:5_000_000])
            except Exception:  # noqa: BLE001
                pass
            clean_path.write_text(text, encoding="utf-8")

            chunks = self._chunk_text(text)
            return {
                "source_id": source_id,
                "canonical_url": final_url,
                "domain": self._extract_domain(final_url),
                "title": title,
                "author": author,
                "language": language,
                "published_at": published_at,
                "retrieved_at": int(time.time()),
                "content_type": content_type,
                "raw_path": str(raw_path),
                "clean_path": str(clean_path),
                "text_length": len(text),
                "text": text,
                "chunks": chunks,
            }
        except Exception:  # noqa: BLE001
            return None

    def _extract_html(self, html: str) -> tuple[str, str, str, str, str]:
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(["script", "style", "noscript", "svg", "footer", "nav", "aside"]):
            tag.decompose()

        title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
        author = ""
        language = ""
        published_at = ""

        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            language = str(html_tag.get("lang") or "").strip()

        for meta in soup.find_all("meta"):
            key = (meta.get("name") or meta.get("property") or "").strip().lower()
            value = (meta.get("content") or "").strip()
            if not value:
                continue
            if key in {"author", "article:author"} and not author:
                author = value
            if key in {
                "article:published_time",
                "og:published_time",
                "publishdate",
                "date",
                "dc.date",
                "dc.date.issued",
                "parsely-pub-date",
            } and not published_at:
                published_at = value

        main = soup.find("main") or soup.find("article")
        if main:
            text = main.get_text(" ", strip=True)
        else:
            text = soup.get_text(" ", strip=True)

        return text, title, author, language, published_at

    def _extract_pdf(self, content: bytes) -> tuple[str, str, str, str]:
        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception:  # noqa: BLE001
            return "", "", "", ""

        pages = []
        for page in reader.pages[:80]:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001
                continue

        text = "\n".join(pages)
        try:
            meta = reader.metadata or {}
            title = str(meta.get("/Title") or "")
            author = str(meta.get("/Author") or "")
            published_at = str(meta.get("/CreationDate") or "")
        except Exception:  # noqa: BLE001
            title = ""
            author = ""
            published_at = ""
        return text, title, author, published_at

    def _chunk_text(self, text: str) -> list[dict[str, Any]]:
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        chunks: list[dict[str, Any]] = []

        start = 0
        idx = 1
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": f"CHK-{idx:04d}",
                        "start": start,
                        "end": end,
                        "text": chunk_text,
                    }
                )
                idx += 1
            if end >= len(text):
                break
            start = max(0, end - overlap)

        return chunks

    def _heuristic_source_score(self, query: str, source: dict[str, Any]) -> dict[str, float]:
        title = str(source.get("title") or "")
        text = str(source.get("text") or "")
        domain = str(source.get("domain") or "")
        published_at = str(source.get("published_at") or "")

        q_tokens = set(self._simple_tokens(query))
        doc_tokens = set(self._simple_tokens(title + " " + text[:2500]))

        overlap = 0.0
        if q_tokens:
            overlap = len(q_tokens & doc_tokens) / len(q_tokens)

        length_score = min(float(len(text)) / 6000.0, 1.0)

        domain_score = 0.4
        trusted_hints = [".gov", ".edu", "who.int", "oecd", "europa.eu", "nature.com", "arxiv.org"]
        if any(hint in domain for hint in trusted_hints):
            domain_score = 1.0
        elif domain.endswith(".org"):
            domain_score = 0.7

        freshness = 0.5
        year_match = re.search(r"(19|20)\\d{2}", published_at)
        if year_match:
            year = int(year_match.group(0))
            current_year = int(time.strftime("%Y"))
            age = current_year - year
            if age <= 1:
                freshness = 1.0
            elif age <= 3:
                freshness = 0.8
            elif age <= 6:
                freshness = 0.6
            else:
                freshness = 0.35

        final = 0.42 * overlap + 0.23 * length_score + 0.20 * domain_score + 0.15 * freshness
        return {
            "overlap": round(overlap, 4),
            "length": round(length_score, 4),
            "domain": round(domain_score, 4),
            "freshness": round(freshness, 4),
            "final": round(final, 4),
        }

    async def _ask_text_model(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        llm_logs: list[dict[str, Any]],
        stage: str,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.config.llm_retry_attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=self.config.llm_timeout_seconds) as client:
                    response = await client.post(f"{self.base_url}/api/chat", json=payload)
                    response.raise_for_status()
                    data = response.json()
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.config.llm_retry_attempts:
                    detail = str(exc).strip() or repr(exc)
                    raise RuntimeError(
                        f"LLM call failed at stage={stage}, model={model}, attempt={attempt}: {detail}"
                    ) from exc
                await asyncio.sleep(1.5 * attempt)
        else:
            detail = str(last_error).strip() if last_error else "unknown error"
            raise RuntimeError(f"LLM call failed at stage={stage}, model={model}: {detail}")

        content = str(data.get("message", {}).get("content", "")).strip()
        llm_logs.append(
            {
                "timestamp": int(time.time()),
                "stage": stage,
                "model": model,
                "request": {"system": system_prompt, "user": user_prompt[:16000]},
                "response": content[:32000],
            }
        )
        return content

    async def _searx_search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        base_url = self.config.searxng_base_url.rstrip("/")
        results: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        if not base_url:
            return results

        page = 1
        per_page_limit = 50
        timeout = httpx.Timeout(self.config.web_timeout_seconds)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            while len(results) < max_results and page <= 10:
                params = {
                    "q": query,
                    "format": "json",
                    "language": "all",
                    "safesearch": "1",
                    "pageno": page,
                }
                try:
                    response = await client.get(f"{base_url}/search", params=params)
                    response.raise_for_status()
                    data = response.json()
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"SearxNG query failed for '{query}' on page {page}: {str(exc) or repr(exc)}"
                    ) from exc

                batch = data.get("results", []) if isinstance(data, dict) else []
                if not batch:
                    break

                for item in batch[:per_page_limit]:
                    url = self._canonicalize_url(str(item.get("url") or ""))
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    results.append(
                        {
                            "url": url,
                            "title": str(item.get("title") or "").strip(),
                            "snippet": str(item.get("content") or "").strip(),
                            "published_at": str(item.get("publishedDate") or "").strip(),
                        }
                    )
                    if len(results) >= max_results:
                        break

                page += 1

        return results

    async def _ddg_search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        def run() -> list[dict[str, Any]]:
            with DDGS(timeout=self.config.web_timeout_seconds) as ddgs:
                return list(
                    ddgs.text(
                        query,
                        max_results=max_results,
                        region=self.config.web_region,
                        safesearch=self.config.web_safesearch,
                    )
                )

        try:
            raw = await asyncio.to_thread(run)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"DuckDuckGo query failed for '{query}': {str(exc) or repr(exc)}") from exc

        results: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for item in raw:
            url = self._canonicalize_url(str(item.get("href") or item.get("url") or ""))
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(
                {
                    "url": url,
                    "title": str(item.get("title") or "").strip(),
                    "snippet": str(item.get("body") or item.get("snippet") or "").strip(),
                    "published_at": "",
                }
            )
            if len(results) >= max_results:
                break

        return results

    async def _probe_searxng(self) -> bool:
        base_url = self.config.searxng_base_url.rstrip("/")
        if not base_url:
            return False
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                response = await client.get(f"{base_url}/search", params={"q": "test", "format": "json"})
                if response.status_code == 200:
                    return True
        except Exception:  # noqa: BLE001
            return False
        return False

    def _parse_json_payload(self, text: str) -> Any:
        text = text.strip()
        if not text:
            return None

        try:
            return json.loads(text)
        except Exception:  # noqa: BLE001
            pass

        fragment = self._find_json_fragment(text)
        if not fragment:
            return None
        try:
            return json.loads(fragment)
        except Exception:  # noqa: BLE001
            return None

    def _find_json_fragment(self, text: str) -> str | None:
        starts = [i for i, ch in enumerate(text) if ch in "[{"]
        for start in starts:
            opener = text[start]
            closer = "]" if opener == "[" else "}"
            depth = 0
            in_string = False
            escaped = False
            for idx in range(start, len(text)):
                char = text[idx]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue

                if char == '"':
                    in_string = True
                elif char == opener:
                    depth += 1
                elif char == closer:
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1]
        return None

    async def _parse_or_repair_json(
        self,
        raw_text: str,
        model: str,
        stage: str,
        llm_logs: list[dict[str, Any]],
        expected_type: str,
        schema_hint: Any,
    ) -> Any:
        parsed = self._parse_json_payload(raw_text)
        if expected_type == "dict" and isinstance(parsed, dict):
            return parsed
        if expected_type == "list" and isinstance(parsed, list):
            return parsed

        latest_invalid = raw_text
        for repair_attempt in range(1, 3):
            repair_payload = {
                "expected_type": expected_type,
                "schema_hint": schema_hint,
                "invalid_output": latest_invalid[:16000],
                "rules": [
                    "Return JSON only",
                    "No markdown",
                    "No prose outside JSON",
                    "Preserve meaning from the original output",
                ],
            }
            repair_text = await self._ask_text_model(
                model=model,
                system_prompt=(
                    "You are a strict JSON repair assistant. "
                    "Convert the input into valid JSON that matches the expected type and schema hint. "
                    "Output JSON only."
                ),
                user_prompt=json.dumps(repair_payload, ensure_ascii=False),
                llm_logs=llm_logs,
                stage=f"{stage}-json-repair-{repair_attempt}",
            )
            repaired = self._parse_json_payload(repair_text)
            if expected_type == "dict" and isinstance(repaired, dict):
                return repaired
            if expected_type == "list" and isinstance(repaired, list):
                return repaired
            latest_invalid = repair_text

        coerced = self._coerce_json_from_text(
            raw_text=raw_text,
            stage=stage,
            expected_type=expected_type,
            schema_hint=schema_hint,
            llm_logs=llm_logs,
        )
        if coerced is not None:
            return coerced

        raise RuntimeError(
            f"{stage}: JSON repair failed (expected {expected_type}). "
            f"Original excerpt: {raw_text[:320]}"
        )

    def _coerce_json_from_text(
        self,
        raw_text: str,
        stage: str,
        expected_type: str,
        schema_hint: Any,
        llm_logs: list[dict[str, Any]],
    ) -> Any | None:
        # Explicit writer-only coercion: keep run alive when chapter text is valid
        # but model ignored JSON wrapper. This is logged to preserve traceability.
        if expected_type != "dict":
            return None
        if not stage.startswith("writer-"):
            return None
        if not isinstance(schema_hint, dict):
            return None
        if "chapter_markdown" not in schema_hint:
            return None

        chapter_markdown = str(raw_text or "").strip()
        if len(chapter_markdown) < 180:
            return None

        payload: dict[str, Any] = {"chapter_markdown": chapter_markdown}
        if "new_related_topics" in schema_hint:
            payload["new_related_topics"] = []
        if "out_of_scope_topics" in schema_hint:
            payload["out_of_scope_topics"] = []

        llm_logs.append(
            {
                "timestamp": int(time.time()),
                "stage": f"{stage}-json-coerce",
                "model": "coercer",
                "request": {
                    "system": "coerce-writer-text-into-json",
                    "user": chapter_markdown[:4000],
                },
                "response": json.dumps(payload, ensure_ascii=False)[:32000],
            }
        )
        return payload

    def _generate_query_variants(self, question: str, max_items: int) -> list[str]:
        base = " ".join(question.split())
        if not base:
            return []

        # Generic query expansion derived from the prompt itself (no hardcoded domain/topic suffixes).
        folded = self._fold_text(base)
        raw_tokens = [token for token in re.findall(r"[a-z0-9]{3,}", folded) if token]
        stopwords = {
            "fais",
            "faire",
            "moi",
            "sur",
            "avec",
            "pour",
            "dans",
            "des",
            "les",
            "une",
            "un",
            "du",
            "de",
            "la",
            "le",
            "est",
            "are",
            "the",
            "and",
            "how",
            "what",
            "which",
            "detaille",
            "detaillee",
            "topo",
        }
        tokens: list[str] = []
        seen_tokens: set[str] = set()
        for token in raw_tokens:
            if token in stopwords:
                continue
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            tokens.append(token)

        candidates = [base]
        if tokens:
            candidates.append(" ".join(tokens[:10]))
            candidates.append(f"\"{base}\"")
            if len(tokens) >= 2:
                candidates.append(" ".join(tokens[: min(4, len(tokens))]))
            if len(tokens) >= 4:
                candidates.append(" ".join(tokens[-4:]))
            if len(tokens) >= 5:
                candidates.append(" ".join(tokens[::2][:5]))

        deduped: list[str] = []
        seen: set[str] = set()
        for query in candidates:
            query = " ".join(str(query or "").split()).strip()
            if not query:
                continue
            key = query.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(query)
            if len(deduped) >= max_items:
                break
        return deduped

    def _canonicalize_url(self, url: str) -> str:
        url = url.strip()
        if not url:
            return ""

        try:
            parsed = urlparse(url)
        except Exception:  # noqa: BLE001
            return ""

        if not parsed.scheme or not parsed.netloc:
            return ""

        query_pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if not k.lower().startswith("utm_")]
        query = urlencode(query_pairs)

        normalized = urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path or "/",
                "",
                query,
                "",
            )
        )
        return normalized

    def _extract_domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:  # noqa: BLE001
            return ""

    def _simple_tokens(self, text: str) -> list[str]:
        return [token for token in re.findall(r"[a-zA-Z0-9]{3,}", text.lower()) if token]

    def _clean_text(self, text: str) -> str:
        text = text.replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _normalize_topic(self, raw_topic: Any) -> dict[str, str] | None:
        if isinstance(raw_topic, dict):
            title = str(raw_topic.get("title") or "").strip()
            why_relevant = str(raw_topic.get("why_relevant") or raw_topic.get("reason") or "").strip()
            viewpoint = str(raw_topic.get("viewpoint") or "both").strip().lower()
        else:
            title = str(raw_topic or "").strip()
            why_relevant = ""
            viewpoint = "both"
        if not title:
            return None
        if viewpoint not in {"benefits", "limitations", "both"}:
            viewpoint = "both"
        return {
            "title": title,
            "why_relevant": why_relevant,
            "viewpoint": viewpoint,
        }

    def _map_sub_question_id(self, candidate: str, planner_ids: list[str]) -> str:
        raw = str(candidate or "").strip()
        if not raw:
            return ""
        if raw in planner_ids:
            return raw
        for planner_id in planner_ids:
            if planner_id.casefold() == raw.casefold():
                return planner_id

        digits = "".join(char for char in raw if char.isdigit())
        if not digits:
            normalized = re.sub(r"[^a-z0-9]", "", raw.casefold())
            normalized = normalized.removeprefix("sq")
            digits = "".join(char for char in normalized if char.isdigit())

        if digits:
            for planner_id in planner_ids:
                planner_digits = "".join(char for char in planner_id if char.isdigit())
                if planner_digits and planner_digits == digits:
                    return planner_id

        return ""

    def _fold_text(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        normalized = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
        normalized = normalized.lower()
        # Join dotted short forms such as "i.mx" -> "imx" for more robust topic matching.
        normalized = re.sub(r"\b([a-z])\s*\.\s*([a-z]{2,})\b", r"\1\2", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _subject_anchor_tokens(self, text: str) -> set[str]:
        folded = self._fold_text(text)
        tokens = [token for token in self._simple_tokens(folded) if token]
        stopwords = {
            "fait",
            "faire",
            "topo",
            "detaille",
            "detail",
            "livre",
            "book",
            "sur",
            "avec",
            "pour",
            "dans",
            "what",
            "which",
            "how",
            "the",
            "and",
            "des",
            "les",
            "une",
            "un",
            "du",
            "de",
            "la",
            "le",
            "plus",
            "overview",
            "guide",
            "about",
            "topic",
            "sujet",
        }
        cleaned: set[str] = set()
        for token in tokens:
            if token in stopwords:
                continue
            cleaned.add(token)
            if any(char.isdigit() for char in token):
                letters_only = re.sub(r"\d+", "", token)
                if len(letters_only) >= 3:
                    cleaned.add(letters_only)
                for part in re.findall(r"[a-z]+", token):
                    if len(part) >= 3:
                        cleaned.add(part)
            if token.endswith("s") and len(token) >= 5:
                cleaned.add(token[:-1])
            if token.endswith("es") and len(token) >= 6:
                cleaned.add(token[:-2])
            if len(token) >= 6:
                cleaned.add(token[:5])
        if cleaned:
            return cleaned
        return set(tokens)

    def _planner_panel_models(self) -> list[str]:
        raw_csv = str(self.config.planner_panel_models_csv or "").strip()
        requested = [item.strip() for item in raw_csv.split(",") if item.strip()]

        defaults = [
            self.config.planner_model,
            self.config.writer_model,
            self.config.extract_model,
            self.config.verify_model,
            self.config.judge_model,
        ]
        backups = [
            "qwen3:32b",
            "llama3.3:70b",
            "deepseek-r1:32b",
            "qwen2.5:32b",
            "mistral-small3.2:24b",
        ]
        combined = [*requested, *defaults, *backups]
        unique: list[str] = []
        seen: set[str] = set()
        for model in combined:
            key = model.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(model)
            if len(unique) >= 3:
                break

        if len(unique) >= 3:
            return unique[:3]
        if len(unique) == 2:
            return [unique[0], unique[1], unique[0]]
        if len(unique) == 1:
            return [unique[0], unique[0], unique[0]]
        return [self.config.planner_model, self.config.planner_model, self.config.planner_model]

    def _planner_book_models(self) -> list[str]:
        configured = [
            str(self.config.planner_book_model_1 or "").strip(),
            str(self.config.planner_book_model_2 or "").strip(),
            str(self.config.planner_book_model_3 or "").strip(),
        ]
        if any(not model for model in configured):
            return []
        return configured

    def _planner_book_json_model(self) -> str:
        return str(self.config.planner_book_model_4_json or "").strip()

    async def _fetch_lightweight_web_doc(self, url: str) -> dict[str, Any] | None:
        timeout = httpx.Timeout(self.config.web_timeout_seconds)
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 CodexDossier/1.0",
            "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
        }
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
                response = await client.get(url)
        except Exception:  # noqa: BLE001
            return None

        if response.status_code >= 400:
            return None

        canonical_url = self._canonicalize_url(str(response.url))
        if not canonical_url:
            return None

        content_type = str(response.headers.get("content-type") or "").lower()
        text = ""
        title = ""
        author = ""
        language = ""
        published_at = ""
        body = response.content
        if "application/pdf" in content_type or canonical_url.lower().endswith(".pdf"):
            text, title, author, published_at = self._extract_pdf(body)
        else:
            text, title, author, language, published_at = self._extract_html(response.text)

        text = self._clean_text(text)
        if len(text) < 180:
            return None

        return {
            "url": canonical_url,
            "domain": self._extract_domain(canonical_url),
            "title": str(title or "").strip(),
            "author": str(author or "").strip(),
            "language": str(language or "").strip(),
            "published_at": str(published_at or "").strip(),
            "content_type": content_type,
            "text": text[: self.config.planner_book_page_chars],
        }

    async def _collect_planner_web_notes(
        self,
        question: str,
        selected_links: list[dict[str, Any]],
        models: list[str],
        llm_logs: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        warnings: list[str] = []
        if not selected_links:
            return [], ["planner web notes: no selected links"]
        if not models:
            return [], ["planner web notes: no models configured"]

        def _normalize_list(raw: Any, max_items: int = 10) -> list[str]:
            items = raw if isinstance(raw, list) else [raw]
            out: list[str] = []
            seen: set[str] = set()
            for item in items:
                value = str(item or "").strip()
                if not value:
                    continue
                key = self._normalize_text_for_dedupe(value)
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(value)
                if len(out) >= max_items:
                    break
            return out

        fetch_semaphore = asyncio.Semaphore(max(1, min(4, self.config.max_parallel_fetch)))
        llm_semaphore = asyncio.Semaphore(max(1, min(2, self.config.max_parallel_llm)))
        notes: list[dict[str, Any]] = []

        async def _worker(idx: int, link: dict[str, Any]) -> dict[str, Any] | None:
            model = models[(idx - 1) % len(models)]
            url = self._canonicalize_url(str(link.get("url") or ""))
            if not url:
                warnings.append(f"planner web notes: invalid url at rank {idx}")
                return None

            async with fetch_semaphore:
                doc = await self._fetch_lightweight_web_doc(url)
            if not doc:
                warnings.append(f"planner web notes: fetch/extract failed for rank {idx} ({url})")
                return None

            payload = {
                "question": question,
                "link": {
                    "rank": idx,
                    "url": doc.get("url", ""),
                    "domain": doc.get("domain", ""),
                    "title": doc.get("title", "") or link.get("title", ""),
                    "snippet": str(link.get("snippet") or "").strip()[:500],
                },
                "page_content": {
                    "text_excerpt": str(doc.get("text") or "")[: self.config.planner_book_page_chars],
                    "language": doc.get("language", ""),
                    "published_at": doc.get("published_at", ""),
                },
                "output_schema": {
                    "is_relevant": "true|false",
                    "relevance_score": "0..1",
                    "summary": "string",
                    "key_topics": ["string"],
                    "key_facts": ["string"],
                    "components": ["string"],
                    "software_stack": ["string"],
                    "configuration_examples": ["string"],
                    "limitations": ["string"],
                    "contradictions_or_uncertainties": ["string"],
                },
            }

            try:
                async with llm_semaphore:
                    raw = await self._ask_text_model(
                        model=model,
                        system_prompt=(
                            "Tu extrais des informations utiles pour preparer le sommaire d'un livre. "
                            "Retourne STRICTEMENT du JSON. "
                            "Conserve uniquement les elements pertinents pour la question."
                        ),
                        user_prompt=json.dumps(payload, ensure_ascii=False),
                        llm_logs=llm_logs,
                        stage=f"planner-web-note-{idx}",
                    )
                    parsed = await self._parse_or_repair_json(
                        raw_text=raw,
                        model=model,
                        stage=f"planner-web-note-{idx}",
                        llm_logs=llm_logs,
                        expected_type="dict",
                        schema_hint=payload["output_schema"],
                    )
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    f"planner web notes: model extraction failed rank={idx}, model={model}: {str(exc) or repr(exc)}"
                )
                return None

            relevance_score = parsed.get("relevance_score", 0.0)
            try:
                relevance_score = float(relevance_score)
            except Exception:  # noqa: BLE001
                relevance_score = 0.0
            relevance_score = max(0.0, min(1.0, relevance_score))
            is_relevant_raw = str(parsed.get("is_relevant") or "").strip().lower()
            is_relevant = is_relevant_raw in {"1", "true", "yes", "oui"}
            if relevance_score >= 0.45:
                is_relevant = True

            note = {
                "rank": idx,
                "model": model,
                "url": doc.get("url", ""),
                "domain": doc.get("domain", ""),
                "title": str(doc.get("title") or link.get("title") or "").strip(),
                "is_relevant": is_relevant,
                "relevance_score": relevance_score,
                "summary": str(parsed.get("summary") or "").strip(),
                "key_topics": _normalize_list(parsed.get("key_topics", []), max_items=12),
                "key_facts": _normalize_list(parsed.get("key_facts", []), max_items=14),
                "components": _normalize_list(parsed.get("components", []), max_items=12),
                "software_stack": _normalize_list(parsed.get("software_stack", []), max_items=12),
                "configuration_examples": _normalize_list(parsed.get("configuration_examples", []), max_items=12),
                "limitations": _normalize_list(parsed.get("limitations", []), max_items=12),
                "contradictions_or_uncertainties": _normalize_list(
                    parsed.get("contradictions_or_uncertainties", []),
                    max_items=10,
                ),
            }
            if (
                not note["key_topics"]
                and not note["key_facts"]
                and not note["components"]
                and not note["software_stack"]
            ):
                warnings.append(
                    f"planner web notes: low-content extraction rank={idx}, model={model}"
                )
            return note

        tasks = [asyncio.create_task(_worker(idx, link)) for idx, link in enumerate(selected_links, start=1)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                warnings.append(f"planner web notes: unexpected worker error: {str(result) or repr(result)}")
                continue
            if result:
                notes.append(result)

        notes.sort(key=lambda item: float(item.get("relevance_score", 0.0)), reverse=True)
        return notes, warnings

    def _aggregate_planner_topics(
        self,
        notes: list[dict[str, Any]],
        max_items: int = 24,
    ) -> list[str]:
        collected: list[str] = []
        seen: set[str] = set()
        for note in notes:
            relevance = float(note.get("relevance_score", 0.0))
            if relevance < 0.3 and not note.get("is_relevant"):
                continue
            candidates = [
                *note.get("key_topics", []),
                *note.get("components", []),
                *note.get("software_stack", []),
                str(note.get("title") or "").strip(),
            ]
            for candidate in candidates:
                value = self._localize_common_outline_label(str(candidate or "").strip())
                if len(value) < 4:
                    continue
                key = self._normalize_text_for_dedupe(value)
                if not key or key in seen:
                    continue
                seen.add(key)
                collected.append(value)
                if len(collected) >= max_items:
                    return collected
        return collected

    def _aggregate_planner_facts(
        self,
        notes: list[dict[str, Any]],
        max_items: int = 120,
    ) -> list[str]:
        facts: list[str] = []
        seen: set[str] = set()
        for note in notes:
            candidates = [
                *note.get("key_facts", []),
                *note.get("configuration_examples", []),
                *note.get("limitations", []),
                *note.get("contradictions_or_uncertainties", []),
            ]
            for candidate in candidates:
                value = str(candidate or "").strip()
                if len(value) < 8:
                    continue
                key = self._normalize_text_for_dedupe(value)
                if not key or key in seen:
                    continue
                seen.add(key)
                facts.append(value)
                if len(facts) >= max_items:
                    return facts
        return facts

    def _outline_lenses(self) -> list[str]:
        raw_csv = str(self.config.outline_lenses_csv or "").strip()
        items = [item.strip() for item in raw_csv.split(",") if item.strip()]
        if not items:
            items = [
                "definitions et perimetre",
                "faits etablis et mecanismes",
                "benefices et opportunites",
                "limites et risques",
                "cout et ressources",
                "environnement",
                "societe et politique",
                "aspects techniques ou operationnels",
                "exemples d'application",
            ]
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            key = self._normalize_text_for_dedupe(item)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:16]

    def _localize_common_outline_label(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        key = self._normalize_text_for_dedupe(raw)
        direct_map = {
            "overview and key features": "Vue d'ensemble et fonctionnalites cles",
            "applications and use cases": "Applications et cas d'usage",
            "technical specifications": "Specifications techniques",
            "advantages": "Avantages",
            "limitations and risks": "Limites et risques",
            "examples of products": "Exemples de produits",
            "resources": "Ressources",
            "processor architecture": "Architecture processeur",
            "memory": "Memoire",
            "multimedia": "Multimedia",
            "connectivity": "Connectivite",
            "industrial automation": "Automatisation industrielle",
            "embedded systems": "Systemes embarques",
            "power efficiency": "Efficacite energetique",
            "thermal range": "Plage thermique",
            "software support": "Support logiciel",
            "ai capabilities": "Capacites IA",
            "dualcore realtime control": "Controle temps reel dual-core",
            "multimedia performance": "Performance multimedia",
            "reliability": "Fiabilite",
            "complexity": "Complexite",
            "power consumption": "Consommation energetique",
            "cost": "Cout",
            "datasheets": "Fiches techniques",
            "developer guides": "Guides developpeur",
        }
        mapped = direct_map.get(key)
        if mapped:
            return mapped
        return raw

    def _outline_topic_token_coverage(
        self,
        master_outline: list[dict[str, Any]],
        topic_titles: list[str],
    ) -> float:
        if not topic_titles or not master_outline:
            return 0.0

        topic_tokens: set[str] = set()
        for title in topic_titles:
            topic_tokens |= self._subject_anchor_tokens(title)
        if not topic_tokens:
            return 0.0

        outline_tokens: set[str] = set()
        for chapter in master_outline:
            if not isinstance(chapter, dict):
                continue
            outline_tokens |= self._subject_anchor_tokens(str(chapter.get("title") or ""))
            for sub in chapter.get("sub_sections", []) if isinstance(chapter.get("sub_sections", []), list) else []:
                outline_tokens |= self._subject_anchor_tokens(str(sub or ""))

        if not outline_tokens:
            return 0.0
        return len(topic_tokens & outline_tokens) / max(1, len(topic_tokens))

    def _outline_has_generic_shape(
        self,
        master_outline: list[dict[str, Any]],
    ) -> bool:
        if not master_outline:
            return True
        generic_markers = {
            "overview",
            "key features",
            "applications",
            "use cases",
            "technical specifications",
            "advantages",
            "limitations",
            "risks",
            "resources",
            "introduction",
            "conclusion",
        }
        generic_hits = 0
        for chapter in master_outline:
            title = self._fold_text(str(chapter.get("title") or ""))
            if not title:
                continue
            if any(marker in title for marker in generic_markers):
                generic_hits += 1
        return generic_hits >= max(3, int(len(master_outline) * 0.5))

    def _enrich_outline_with_presearch_topics(
        self,
        question: str,
        master_outline: list[dict[str, Any]],
        topic_titles: list[str],
        sub_questions: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        warnings: list[str] = []
        if not topic_titles:
            return master_outline, warnings

        desired_chapters = max(
            len(master_outline),
            len(sub_questions),
            len(topic_titles),
        )
        if self.config.web_max_sub_questions > 0:
            desired_chapters = min(desired_chapters, self.config.web_max_sub_questions)
        if len(master_outline) >= desired_chapters:
            return master_outline, warnings

        sq_ids = [
            str(item.get("id") or "").strip()
            for item in sub_questions
            if str(item.get("id") or "").strip()
        ]
        seen_titles: set[str] = {
            self._normalize_text_for_dedupe(str(item.get("title") or ""))
            for item in master_outline
            if isinstance(item, dict)
        }

        for topic in topic_titles:
            if len(master_outline) >= desired_chapters:
                break
            topic_clean = self._localize_common_outline_label(str(topic or "").strip())
            if not topic_clean:
                continue
            topic_key = self._normalize_text_for_dedupe(topic_clean)
            if not topic_key or topic_key in seen_titles:
                continue
            seen_titles.add(topic_key)
            linked = self._match_sub_questions_for_topic(topic_clean, sub_questions, limit=2)
            if not linked and sq_ids:
                linked = [sq_ids[len(master_outline) % len(sq_ids)]]
            master_outline.append(
                {
                    "id": f"CH{len(master_outline) + 1}",
                    "title": topic_clean,
                    "goal": (
                        f"Analyser en profondeur le theme '{topic_clean}' "
                        f"dans le cadre de '{question}'."
                    ),
                    "linked_sub_questions": linked,
                    "status": "planned",
                    "reason": "presearch_topic_enrichment",
                    "sub_sections": self._default_subsections_for_chapter(
                        chapter_title=topic_clean,
                        lenses=self._outline_lenses(),
                        question=question,
                    ),
                }
            )
            warnings.append(f"planner enrichment: added presearch topic chapter '{topic_clean}'")

        return master_outline, warnings

    def _normalize_subsections(
        self,
        raw_subsections: Any,
        chapter_title: str,
        fill_missing: bool = True,
    ) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        raw_values: list[str] = []

        if isinstance(raw_subsections, list):
            for item in raw_subsections:
                if isinstance(item, dict):
                    value = str(item.get("title") or item.get("name") or item.get("section") or "").strip()
                else:
                    value = str(item or "").strip()
                if value:
                    raw_values.append(value)
        elif isinstance(raw_subsections, str):
            raw_values.append(raw_subsections)

        candidates: list[str] = []
        for raw_value in raw_values:
            parts = re.split(r"\n+|[;|]+", raw_value)
            for part in parts:
                cleaned = re.sub(r"^\s*(?:[-*•]+|\d+[\.\)\-])\s*", "", str(part or "").strip())
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                cleaned = self._localize_common_outline_label(cleaned)
                if cleaned:
                    candidates.append(cleaned)

        for value in candidates:
            key = self._normalize_text_for_dedupe(value)
            if not key or key in seen:
                continue
            seen.add(key)
            normalized.append(value)
            if self.config.outline_max_subsections > 0 and len(normalized) >= self.config.outline_max_subsections:
                break
        if (
            fill_missing
            and self.config.outline_min_subsections > 0
            and len(normalized) < self.config.outline_min_subsections
        ):
            auto = self._default_subsections_for_chapter(
                chapter_title=chapter_title,
                lenses=self._outline_lenses(),
                question=chapter_title,
            )
            for item in auto:
                key = self._normalize_text_for_dedupe(item)
                if key in seen:
                    continue
                seen.add(key)
                normalized.append(item)
                # Complete only to the minimum floor; avoid force-filling every chapter to max.
                if self.config.outline_min_subsections > 0 and len(normalized) >= self.config.outline_min_subsections:
                    break
        if self.config.outline_max_subsections > 0:
            return normalized[: self.config.outline_max_subsections]
        return normalized

    def _default_subsections_for_chapter(
        self,
        chapter_title: str,
        lenses: list[str],
        question: str,
    ) -> list[str]:
        chapter = str(chapter_title or "").strip()
        question_clean = str(question or "").strip()
        if not chapter:
            chapter = question_clean or "Chapitre"

        subsections: list[str] = [
            f"Contexte, definitions et delimitation de {chapter}",
            f"Composants, acteurs et ecosysteme de {chapter}",
            f"Mecanismes de fonctionnement et points clefs de {chapter}",
            f"Mise en oeuvre pratique: methodes, configuration et workflow pour {chapter}",
            f"Exemples d'application et retours d'experience autour de {chapter}",
            f"Evaluation: performance, qualite, couts et ressources pour {chapter}",
            f"Limites, risques, controverses et points faibles de {chapter}",
            f"Enjeux environnementaux, societaux, juridiques et politiques de {chapter}",
            f"Comparaison des alternatives et criteres de decision pour {chapter}",
            f"Perspectives d'evolution et recommandations sur {chapter}",
        ]

        for lens in lenses:
            lens_clean = str(lens or "").strip()
            if not lens_clean:
                continue
            subsections.append(f"Analyse ciblee '{lens_clean}' appliquee a {chapter}")
            if self.config.outline_max_subsections > 0 and len(subsections) >= self.config.outline_max_subsections:
                break

        deduped: list[str] = []
        seen: set[str] = set()
        for item in subsections:
            key = self._normalize_text_for_dedupe(item)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if self.config.outline_max_subsections > 0 and len(deduped) >= self.config.outline_max_subsections:
                break
        if self.config.outline_max_subsections > 0:
            return deduped[: self.config.outline_max_subsections]
        return deduped

    async def _refine_master_outline_iteratively(
        self,
        question: str,
        presearch_context: dict[str, Any],
        sub_questions: list[dict[str, Any]],
        master_outline: list[dict[str, Any]],
        llm_logs: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        warnings: list[str] = []
        rounds = max(0, int(self.config.outline_refinement_rounds))
        if rounds <= 0:
            return master_outline, warnings

        current = master_outline
        model = self.config.planner_synth_model or self.config.judge_model
        schema_hint = {
            "master_outline": [
                {
                    "id": "CH1",
                    "title": "string",
                    "goal": "string",
                    "linked_sub_questions": ["SQ1"],
                    "status": "planned|rejected",
                    "reason": "string",
                    "sub_sections": ["string"],
                }
            ]
        }

        for round_idx in range(1, rounds + 1):
            payload = {
                "question": question,
                "presearch_context": presearch_context,
                "sub_questions": sub_questions,
                "outline_lenses": self._outline_lenses(),
                "book_coverage_axes": [
                    "avantages et opportunites",
                    "inconvenients, limites et risques",
                    "couts, ressources et contraintes",
                    "environnement, societe et politique",
                    "aspects techniques ou operationnels",
                    "exemples d'application et cas concrets",
                ],
                "current_master_outline": current,
                "constraints": {
                    "language": "fr",
                    "reject_out_of_scope_topics": True,
                    "allow_new_related_chapters": True,
                    "no_duplicate_chapters": True,
                    "no_duplicate_subsections": True,
                },
                "output_schema": schema_hint,
            }
            if self.config.outline_min_subsections > 0:
                payload["constraints"]["min_subsections_per_chapter"] = self.config.outline_min_subsections
            if self.config.outline_max_subsections > 0:
                payload["constraints"]["max_subsections_per_chapter"] = self.config.outline_max_subsections
            try:
                raw = await self._ask_text_model(
                    model=model,
                    system_prompt=(
                        "You refine a book outline for deep analysis. "
                        "Expand and improve chapter sub-sections while keeping only relevant topics. "
                        "Each chapter must have concrete and non-redundant sub-sections directly linked to the chapter title. "
                        "When relevant, ensure the full outline covers both positive and negative angles, costs, "
                        "environmental/political aspects, technical details, and application examples. "
                        "Return strict JSON only, in French."
                    ),
                    user_prompt=json.dumps(payload, ensure_ascii=False),
                    llm_logs=llm_logs,
                    stage=f"planner-outline-refine-{round_idx}",
                )
                parsed = await self._parse_or_repair_json(
                    raw_text=raw,
                    model=model,
                    stage=f"planner-outline-refine-{round_idx}",
                    llm_logs=llm_logs,
                    expected_type="dict",
                    schema_hint=schema_hint,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    f"outline refinement round {round_idx} failed: {str(exc) or repr(exc)}"
                )
                break

            refined = parsed.get("master_outline", [])
            if not isinstance(refined, list) or not refined:
                warnings.append(
                    f"outline refinement round {round_idx} returned empty outline; keeping previous version"
                )
                break

            sq_ids = [
                str(item.get("id") or "").strip()
                for item in sub_questions
                if str(item.get("id") or "").strip()
            ]
            next_outline: list[dict[str, Any]] = []
            seen_titles: set[str] = set()
            for idx, item in enumerate(refined, start=1):
                title = str(item.get("title") or "").strip()
                if not title:
                    continue
                key = self._normalize_text_for_dedupe(title)
                if not key or key in seen_titles:
                    continue
                seen_titles.add(key)
                linked = []
                for sq_id in item.get("linked_sub_questions", []) if isinstance(item.get("linked_sub_questions"), list) else []:
                    mapped = self._map_sub_question_id(str(sq_id or ""), sq_ids)
                    if mapped and mapped not in linked:
                        linked.append(mapped)
                if not linked and sq_ids:
                    linked = [sq_ids[min(idx - 1, len(sq_ids) - 1)]]
                status = str(item.get("status") or "planned").strip().lower()
                if status not in {"planned", "rejected"}:
                    status = "planned"
                next_outline.append(
                    {
                        "id": str(item.get("id") or f"CH{idx}").strip() or f"CH{idx}",
                        "title": title,
                        "goal": str(item.get("goal") or "").strip(),
                        "linked_sub_questions": linked,
                        "status": status,
                        "reason": str(item.get("reason") or "").strip(),
                        "sub_sections": self._normalize_subsections(
                            item.get("sub_sections", []),
                            chapter_title=title,
                        ),
                    }
                )
            if not next_outline:
                warnings.append(
                    f"outline refinement round {round_idx} produced no valid chapters; keeping previous version"
                )
                break
            current = next_outline

        return current, warnings

    def _build_question_anchor_terms(self, question: str) -> set[str]:
        anchors = self._subject_anchor_tokens(question)

        strong: set[str] = set()
        for token in anchors:
            token_clean = str(token).strip().lower()
            if len(token_clean) < 3:
                continue
            if (
                any(char.isdigit() for char in token_clean)
                or len(token_clean) >= 5
            ):
                strong.add(token_clean)

        if not strong:
            for token in sorted(anchors):
                if len(token) >= 4:
                    strong.add(token)
        return strong

    def _domain_quality_score(self, domain: str) -> float:
        folded = self._fold_text(domain)
        if not folded:
            return 0.35
        if any(item in folded for item in [".gov", ".edu"]):
            return 0.9
        if folded.endswith(".org"):
            return 0.65
        if folded.endswith(".com"):
            return 0.5
        if folded.endswith(".net"):
            return 0.45
        if folded.endswith(".io"):
            return 0.45
        if folded.endswith(".fr") or folded.endswith(".de") or folded.endswith(".uk"):
            return 0.75
        return 0.45

    def _doc_type_score(self, text: str) -> float:
        folded = self._fold_text(text)
        if not folded:
            return 0.0
        strong = [
            "systematic review",
            "peer reviewed",
            "official report",
            "guideline",
            "standard",
            "regulation",
            "dataset",
            "methodology",
            "evidence",
            "white paper",
            "whitepaper",
            "rapport",
            "etude",
            "publication",
        ]
        weak = [
            "documentation",
            "guide",
            "tutorial",
            "example",
            "overview",
            "article",
            "blog",
            "news",
            "faq",
        ]
        if any(keyword in folded for keyword in strong):
            return 1.0
        if any(keyword in folded for keyword in weak):
            return 0.6
        return 0.2

    def _evaluate_presearch_link(
        self,
        question: str,
        strong_anchors: set[str],
        queries: list[str],
        link: dict[str, Any],
    ) -> dict[str, Any]:
        title = str(link.get("title") or "").strip()
        snippet = str(link.get("snippet") or "").strip()
        url = str(link.get("url") or "").strip()
        domain = str(link.get("domain") or "").strip()
        folded_blob = self._fold_text(f"{title} {snippet} {url}")
        if not folded_blob:
            return {"keep": False, "reason": "empty_text"}

        anchor_hits = [anchor for anchor in strong_anchors if anchor and anchor in folded_blob]
        question_tokens = self._subject_anchor_tokens(question)
        link_tokens = self._subject_anchor_tokens(f"{title} {snippet} {domain}")
        overlap = 0.0
        if question_tokens and link_tokens:
            overlap = len(question_tokens & link_tokens) / max(1, len(question_tokens))

        query_overlap = 0.0
        for query in queries:
            query_tokens = self._subject_anchor_tokens(query)
            if not query_tokens or not link_tokens:
                continue
            score = len(query_tokens & link_tokens) / max(1, len(query_tokens))
            if score > query_overlap:
                query_overlap = score

        anchor_overlap = 0.0
        if strong_anchors and link_tokens:
            anchor_overlap = len(set(strong_anchors) & link_tokens) / max(1, len(strong_anchors))

        if not anchor_hits and anchor_overlap < 0.12 and query_overlap < 0.22:
            return {
                "keep": False,
                "reason": "missing_anchor",
                "signals": {
                    "anchor_overlap": round(anchor_overlap, 4),
                    "query_overlap": round(query_overlap, 4),
                },
            }

        specificity_score = 0.0
        if link_tokens:
            long_tokens = [token for token in link_tokens if len(token) >= 6]
            specificity_score = min(1.0, len(long_tokens) / max(1, len(link_tokens)))

        anchor_score = max(
            min(1.0, len(anchor_hits) / max(1, min(3, len(strong_anchors)))),
            min(1.0, anchor_overlap * 1.8),
        )
        domain_score = self._domain_quality_score(domain)
        doc_score = self._doc_type_score(f"{title} {snippet}")
        final_score = (
            0.45 * anchor_score
            + 0.20 * overlap
            + 0.15 * query_overlap
            + 0.10 * domain_score
            + 0.10 * max(doc_score, specificity_score * 0.8)
        )
        if final_score < 0.45:
            return {
                "keep": False,
                "reason": "low_score",
                "signals": {
                    "anchor_hits": anchor_hits[:6],
                    "anchor_score": round(anchor_score, 4),
                    "anchor_overlap": round(anchor_overlap, 4),
                    "overlap": round(overlap, 4),
                    "query_overlap": round(query_overlap, 4),
                    "domain_score": round(domain_score, 4),
                    "doc_score": round(doc_score, 4),
                    "specificity_score": round(specificity_score, 4),
                    "final_score": round(final_score, 4),
                },
            }

        return {
            "keep": True,
            "score": final_score,
            "signals": {
                "anchor_hits": anchor_hits[:6],
                "anchor_score": round(anchor_score, 4),
                "anchor_overlap": round(anchor_overlap, 4),
                "overlap": round(overlap, 4),
                "query_overlap": round(query_overlap, 4),
                "domain_score": round(domain_score, 4),
                "doc_score": round(doc_score, 4),
                "specificity_score": round(specificity_score, 4),
                "final_score": round(final_score, 4),
            },
        }

    def _subquestion_has_theme(self, text: str, theme: str) -> bool:
        folded = self._fold_text(text)
        if not folded:
            return False
        if theme == "implementation":
            keywords = [
                "mise en oeuvre",
                "implementation",
                "configuration",
                "configurer",
                "exemple",
                "example",
                "code",
                "integration",
                "deploiement",
                "setup",
                "usage",
                "application pratique",
                "protocole",
                "procedure",
                "workflow",
                "methodes",
            ]
        elif theme == "limitations":
            keywords = [
                "limite",
                "limitations",
                "risque",
                "risks",
                "tradeoff",
                "trade-off",
                "compromis",
                "contraintes",
                "point faible",
                "weakness",
                "inconvenient",
            ]
        else:
            keywords = [theme]
        return any(keyword in folded for keyword in keywords)

    def _collect_presearch_topic_titles(
        self,
        presearch_payload: dict[str, Any],
        max_items: int = 20,
    ) -> list[str]:
        candidates: list[tuple[int, str]] = []
        priority_rank = {"high": 0, "medium": 1, "low": 2}
        question_anchors = self._subject_anchor_tokens(str(presearch_payload.get("question") or ""))
        for item in presearch_payload.get("topic_candidates", []) if isinstance(presearch_payload.get("topic_candidates"), list) else []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            priority = str(item.get("priority") or "medium").strip().lower()
            candidates.append((priority_rank.get(priority, 2), title))

        for link in presearch_payload.get("links", []) if isinstance(presearch_payload.get("links"), list) else []:
            title = str(link.get("title") or "").strip()
            if not title:
                continue
            title = re.sub(r"\s*[|•·]\s*[^|•·]+$", "", title).strip()
            candidates.append((3, title))

        def _topic_relevance_score(title: str) -> int:
            if not question_anchors:
                return 1
            folded_title = self._fold_text(title)
            title_tokens = set(self._simple_tokens(folded_title))
            token_overlap = len(title_tokens & question_anchors) if title_tokens else 0
            title_compact = re.sub(r"[^a-z0-9]", "", folded_title)
            compact_overlap = 0
            for anchor in question_anchors:
                anchor_compact = re.sub(r"[^a-z0-9]", "", str(anchor or "").lower())
                if len(anchor_compact) >= 4 and anchor_compact in title_compact:
                    compact_overlap += 1
            return token_overlap + compact_overlap

        deduped: list[str] = []
        seen: set[str] = set()
        sorted_candidates = sorted(candidates, key=lambda item: item[0])
        for strict_mode in (True, False):
            for rank, title in sorted_candidates:
                # Link titles are useful but should not dominate chapter planning.
                if rank >= 3 and len(deduped) >= 10:
                    continue
                cleaned = re.sub(r"\s+", " ", title).strip()
                if len(cleaned) < 4:
                    continue
                if len(cleaned) > 140:
                    cleaned = cleaned[:140].rstrip()
                if strict_mode and question_anchors and _topic_relevance_score(cleaned) <= 0:
                    continue
                key = self._normalize_text_for_dedupe(cleaned)
                if not key or key in seen:
                    continue
                seen.add(key)
                deduped.append(cleaned)
                if len(deduped) >= max_items:
                    break
            if len(deduped) >= max_items:
                break

        return deduped

    def _build_fallback_subquestions(
        self,
        question: str,
        topic_titles: list[str],
    ) -> list[dict[str, Any]]:
        question_clean = " ".join(str(question or "").split()).strip()
        if not question_clean:
            return []

        def _mk_sq(question_text: str, theme: str, extra_queries: list[str], proof: list[str]) -> dict[str, Any]:
            queries: list[str] = []
            seen_queries: set[str] = set()
            for value in extra_queries:
                query = " ".join(str(value or "").split()).strip()
                if not query:
                    continue
                key = query.casefold()
                if key in seen_queries:
                    continue
                seen_queries.add(key)
                queries.append(query)
            seed_for_variants = queries[0] if queries else question_clean
            if len(queries) < self.config.web_query_variants:
                for query in self._generate_query_variants(seed_for_variants, self.config.web_query_variants):
                    key = query.casefold()
                    if key in seen_queries:
                        continue
                    seen_queries.add(key)
                    queries.append(query)
                    if len(queries) >= self.config.web_query_variants:
                        break
            return {
                "question": question_text,
                "proof_criteria": proof,
                "search_queries": queries[: self.config.web_query_variants],
                "theme": theme,
            }

        output: list[dict[str, Any]] = []
        output.append(
            _mk_sq(
                question_text=(
                    f"Quel est le perimetre exact de '{question_clean}' et quels sont les axes "
                    "et sous-sujets majeurs a analyser ?"
                ),
                theme="overview",
                extra_queries=[
                    question_clean,
                    f"{question_clean} overview",
                    f"{question_clean} scope",
                    f"{question_clean} key topics",
                ],
                proof=[
                    "Definition claire du sujet principal",
                    "Liste structuree des axes/sous-sujets majeurs",
                    "Sources de reference officielles ou techniques",
                ],
            )
        )

        detail_topics = [title for title in topic_titles if str(title).strip()]
        fallback_sq_cap = self.config.web_max_sub_questions

        seen_detail_topics: set[str] = set()
        for title in detail_topics:
            title_clean = str(title).strip()
            if not title_clean:
                continue
            key = self._normalize_text_for_dedupe(title_clean)
            if not key or key in seen_detail_topics:
                continue
            seen_detail_topics.add(key)
            output.append(
                _mk_sq(
                    question_text=(
                        f"Analyse detaillee du theme '{title_clean}' pour '{question_clean}' : "
                        "faits etablis, mecanismes, applications, controverses et limites."
                    ),
                    theme="deep-dive",
                    extra_queries=[
                        f"{question_clean} {title_clean}",
                        f"{title_clean} {question_clean} evidence",
                        f"{title_clean} {question_clean} analysis",
                    ],
                    proof=[
                        "Elements de preuve clairs et exploitables",
                        "Contexte et mecanismes ou methodes de mise en oeuvre",
                        "Limites documentees et contradictions eventuelles",
                    ],
                )
            )
            if fallback_sq_cap > 0 and len(output) >= fallback_sq_cap:
                break

        output.append(
            _mk_sq(
                question_text=(
                    f"Comment appliquer '{question_clean}' en pratique : methodes, protocole, "
                    "workflow, exemples operationnels, validation ?"
                ),
                theme="implementation",
                extra_queries=[
                    f"{question_clean} implementation",
                    f"{question_clean} practical guide",
                    f"{question_clean} methods",
                    f"{question_clean} examples",
                ],
                proof=[
                    "Exemples concrets reproductibles ou comparables",
                    "Etapes et parametres de mise en oeuvre explicites",
                    "Critere de validation/verification",
                ],
            )
        )

        output.append(
            _mk_sq(
                question_text=(
                    f"Quelles sont les limites, risques, compromis et points faibles de '{question_clean}', "
                    "et dans quels cas ces limites deviennent bloquantes ?"
                ),
                theme="limitations",
                extra_queries=[
                    f"{question_clean} limitations",
                    f"{question_clean} tradeoffs",
                    f"{question_clean} risks",
                    f"{question_clean} benchmark",
                ],
                proof=[
                    "Limites techniques explicites et contextuelles",
                    "Compromis performance/cout/complexite",
                    "Exemples de cas defavorables",
                ],
            )
        )
        if fallback_sq_cap > 0:
            return output[:fallback_sq_cap]
        return output

    def _match_sub_questions_for_topic(
        self,
        topic_title: str,
        sub_questions: list[dict[str, Any]],
        limit: int = 0,
    ) -> list[str]:
        topic_tokens = set(self._simple_tokens(self._fold_text(topic_title)))
        if not topic_tokens:
            return []
        scored: list[tuple[float, str]] = []
        for sq in sub_questions:
            sq_id = str(sq.get("id") or "").strip()
            if not sq_id:
                continue
            sq_tokens = set(self._simple_tokens(self._fold_text(sq.get("question", ""))))
            if not sq_tokens:
                continue
            overlap = len(topic_tokens & sq_tokens) / max(1, len(topic_tokens | sq_tokens))
            if overlap <= 0:
                continue
            scored.append((overlap, sq_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        output: list[str] = []
        selected = scored if limit <= 0 else scored[:limit]
        for _, sq_id in selected:
            if sq_id not in output:
                output.append(sq_id)
        return output

    def _build_master_outline_from_topics(
        self,
        question: str,
        topic_titles: list[str],
        sub_questions: list[dict[str, Any]],
        target_count: int,
    ) -> list[dict[str, Any]]:
        outline: list[dict[str, Any]] = []
        sq_ids = [str(item.get("id") or "").strip() for item in sub_questions if str(item.get("id") or "").strip()]
        topics = [item for item in topic_titles if str(item).strip()]
        if not topics:
            topics = [str(item.get("question") or "").strip() for item in sub_questions if str(item.get("question") or "").strip()]

        for idx, topic in enumerate(topics, start=1):
            topic_label = self._localize_common_outline_label(str(topic or "").strip())
            if not topic_label:
                continue
            linked = self._match_sub_questions_for_topic(topic_label, sub_questions, limit=2)
            if not linked and sq_ids:
                linked = [sq_ids[min(idx - 1, len(sq_ids) - 1)]]
            outline.append(
                {
                    "id": f"CH{idx}",
                    "title": topic_label,
                    "goal": f"Analyser en profondeur le theme '{topic_label}' dans le cadre de '{question}'.",
                    "linked_sub_questions": linked,
                    "status": "planned",
                    "reason": "derived_from_presearch",
                    "sub_sections": self._default_subsections_for_chapter(
                        chapter_title=topic_label,
                        lenses=self._outline_lenses(),
                        question=question,
                    ),
                }
            )
            if len(outline) >= target_count:
                break

        covered = {
            sq_id
            for item in outline
            for sq_id in item.get("linked_sub_questions", [])
            if sq_id
        }
        for sq in sub_questions:
            sq_id = str(sq.get("id") or "").strip()
            sq_question = str(sq.get("question") or "").strip()
            if not sq_id or sq_id in covered:
                continue
            outline.append(
                {
                    "id": f"CH{len(outline) + 1}",
                    "title": sq_question,
                    "goal": "Chapitre ajoute pour couvrir une sous-question non reliee.",
                    "linked_sub_questions": [sq_id],
                    "status": "planned",
                    "reason": "coverage_completion",
                    "sub_sections": self._default_subsections_for_chapter(
                        chapter_title=sq_question,
                        lenses=self._outline_lenses(),
                        question=question,
                    ),
                }
            )

        return outline

    def _markdown_anchor(self, text: str) -> str:
        value = str(text or "").strip().lower()
        value = re.sub(r"`+", "", value)
        value = re.sub(r"[^a-z0-9\s-]", "", value)
        value = re.sub(r"\s+", "-", value).strip("-")
        value = re.sub(r"-{2,}", "-", value)
        return value or "section"

    def _dedupe_topics(self, topics: list[Any], max_items: int = 20) -> list[dict[str, str]]:
        deduped: list[dict[str, str]] = []
        seen: set[str] = set()
        for raw_topic in topics:
            normalized = self._normalize_topic(raw_topic)
            if not normalized:
                continue
            key = normalized["title"].casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
            if len(deduped) >= max_items:
                break
        return deduped

    def _dedupe_rejected_topics(self, topics: list[Any], max_items: int = 30) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []
        seen: set[str] = set()
        for raw_topic in topics:
            if isinstance(raw_topic, dict):
                title = str(raw_topic.get("title") or "").strip()
                reason = str(raw_topic.get("reason") or "").strip()
            else:
                title = str(raw_topic or "").strip()
                reason = ""
            if not title:
                continue
            key = title.casefold()
            if key in seen:
                continue
            seen.add(key)
            output.append({"title": title, "reason": reason})
            if len(output) >= max_items:
                break
        return output

    def _dedupe_global_topics(self, topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen_titles: set[str] = set()
        for idx, topic in enumerate(topics, start=1):
            title = str(topic.get("title") or "").strip()
            if not title:
                continue
            key = title.casefold()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            status = str(topic.get("status") or "planned").strip().lower()
            if status not in {"planned", "rejected"}:
                status = "planned"
            related_sq = [
                str(value).strip()
                for value in topic.get("related_sub_question_ids", [])
                if str(value).strip()
            ]
            output.append(
                {
                    "id": str(topic.get("id") or f"TOP-{idx}").strip() or f"TOP-{idx}",
                    "title": title,
                    "scope": str(topic.get("scope") or "").strip(),
                    "related_sub_question_ids": list(dict.fromkeys(related_sq)),
                    "status": status,
                    "reason": str(topic.get("reason") or "").strip(),
                }
            )
        return output

    def _normalize_text_for_dedupe(self, text: str) -> str:
        normalized = str(text or "").strip().lower()
        normalized = re.sub(r"https?://\S+", " ", normalized)
        normalized = re.sub(r"`+", "", normalized)
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _is_duplicate_markdown_block(self, candidate: str, existing_blocks: list[str]) -> bool:
        candidate_norm = self._normalize_text_for_dedupe(candidate)
        if len(candidate_norm) < 120:
            return False
        candidate_tokens = set(self._simple_tokens(candidate_norm))
        for block in existing_blocks:
            block_norm = self._normalize_text_for_dedupe(block)
            if not block_norm:
                continue
            if candidate_norm == block_norm:
                return True
            if candidate_norm in block_norm or block_norm in candidate_norm:
                return True
            block_tokens = set(self._simple_tokens(block_norm))
            if not candidate_tokens or not block_tokens:
                continue
            overlap = len(candidate_tokens & block_tokens) / max(1, min(len(candidate_tokens), len(block_tokens)))
            if overlap >= 0.92 and abs(len(candidate_tokens) - len(block_tokens)) <= 30:
                return True
        return False

    def _sanitize_writer_markdown(self, text: str) -> str:
        cleaned = str(text or "").strip()
        cleaned = re.sub(r"^\s*```(?:markdown|md|text)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        cleaned = re.sub(r"(?im)^\s*###\s*\*{0,2}chapitre[^\n]*\n?", "", cleaned)
        cleaned = re.sub(r"(?im)^\s*---+\s*$", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _dedupe_markdown_paragraphs(self, text: str) -> str:
        chunks = [item.strip() for item in re.split(r"\n\s*\n", str(text or "").strip()) if item.strip()]
        kept: list[str] = []
        kept_norms: list[str] = []
        for chunk in chunks:
            chunk_norm = self._normalize_text_for_dedupe(chunk)
            if not chunk_norm:
                continue
            is_duplicate = False
            for existing_norm in kept_norms:
                if chunk_norm == existing_norm:
                    is_duplicate = True
                    break
                if len(chunk_norm) > 140 and (chunk_norm in existing_norm or existing_norm in chunk_norm):
                    is_duplicate = True
                    break
                chunk_tokens = set(self._simple_tokens(chunk_norm))
                existing_tokens = set(self._simple_tokens(existing_norm))
                if chunk_tokens and existing_tokens:
                    overlap = len(chunk_tokens & existing_tokens) / max(1, min(len(chunk_tokens), len(existing_tokens)))
                    if overlap >= 0.93 and abs(len(chunk_tokens) - len(existing_tokens)) <= 28:
                        is_duplicate = True
                        break
            if is_duplicate:
                continue
            kept.append(chunk)
            kept_norms.append(chunk_norm)
        return "\n\n".join(kept)

    def _pick_best_planned_chapter_for_topic(
        self,
        topic_title: str,
        planned_chapters: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not planned_chapters:
            return {"title": "", "goal": "", "sub_sections": []}

        topic_tokens = set(self._simple_tokens(self._fold_text(topic_title)))
        best_score = -1.0
        best: dict[str, Any] | None = None
        for chapter in planned_chapters:
            chapter_title = str(chapter.get("title") or "").strip()
            chapter_tokens = set(self._simple_tokens(self._fold_text(chapter_title)))
            if not chapter_tokens:
                continue
            overlap = len(topic_tokens & chapter_tokens) / max(1, len(topic_tokens | chapter_tokens))
            if overlap > best_score:
                best_score = overlap
                best = chapter

        if not best:
            best = planned_chapters[0]

        return {
            "title": str(best.get("title") or "").strip(),
            "goal": str(best.get("goal") or "").strip(),
            "sub_sections": self._normalize_subsections(
                best.get("sub_sections", []),
                chapter_title=str(best.get("title") or topic_title).strip() or topic_title,
            ),
        }

    def _planned_chapters_markdown(self, planned_chapters: list[dict[str, Any]]) -> str:
        if not planned_chapters:
            return "- Aucun chapitre planner relie a cette sous-question."

        lines: list[str] = []
        for chapter in planned_chapters[:10]:
            chapter_id = str(chapter.get("id") or "").strip()
            chapter_title = str(chapter.get("title") or "").strip()
            chapter_goal = str(chapter.get("goal") or "").strip()
            prefix = f"`{chapter_id}` " if chapter_id else ""
            if chapter_title:
                lines.append(f"- {prefix}**{chapter_title}** - {chapter_goal or 'Objectif non precise.'}")
            sub_sections = self._normalize_subsections(
                chapter.get("sub_sections", []),
                chapter_title=chapter_title or "chapitre",
            )
            for idx, sub_section in enumerate(sub_sections, start=1):
                lines.append(f"- {chapter_id or 'CH'}.{idx} {sub_section}")
        return "\n".join(lines) if lines else "- Aucun chapitre planner relie a cette sous-question."

    def _master_outline_markdown(self, master_outline: list[dict[str, Any]]) -> str:
        if not master_outline:
            return "- Aucun chapitre propose."

        lines: list[str] = []
        for chapter in master_outline:
            if not isinstance(chapter, dict):
                continue
            chapter_id = str(chapter.get("id") or "").strip()
            chapter_title = str(chapter.get("title") or "").strip()
            status = str(chapter.get("status") or "planned").strip()
            goal = str(chapter.get("goal") or "").strip()
            reason = str(chapter.get("reason") or "").strip()
            linked_sq = chapter.get("linked_sub_questions", [])
            if not isinstance(linked_sq, list):
                linked_sq = []
            sub_sections = self._normalize_subsections(
                chapter.get("sub_sections", []),
                chapter_title=chapter_title or chapter_id or "chapitre",
            )

            title_cell = chapter_title or "Titre non precise"
            lines.append(
                f"### {chapter_id or 'CH'} - {title_cell}"
            )
            lines.append(
                (
                    f"- Statut: `{status}`\n"
                    f"- Objectif: {goal or 'ND'}\n"
                    f"- Sous-questions liees: {', '.join(linked_sq) or 'ND'}\n"
                    f"- Raison: {reason or 'ND'}"
                )
            )
            lines.append("- Sous-sections prevues:")
            if sub_sections:
                for idx, sub in enumerate(sub_sections, start=1):
                    lines.append(f"- {chapter_id or 'CH'}.{idx} {sub}")
            else:
                lines.append("- Aucun detail de sous-section fourni.")
            lines.append("")

        output = "\n".join(lines).strip()
        return output or "- Aucun chapitre propose."

    def _pick_claims_for_topic(
        self,
        topic_title: str,
        claim_rows: list[dict[str, Any]],
        limit: int,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if not claim_rows:
            return []

        topic_tokens = set(self._simple_tokens(topic_title))
        status_weight = {"ACCEPTED": 1.0, "UNCERTAIN": 0.7, "REJECTED": 0.45}
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in claim_rows:
            claim_text = str(row.get("text") or "")
            claim_tokens = set(self._simple_tokens(claim_text))
            overlap = 0.0
            if topic_tokens and claim_tokens:
                overlap = len(topic_tokens & claim_tokens) / max(1, len(topic_tokens | claim_tokens))
            try:
                confidence = float(row.get("confidence", 0.0))
            except Exception:  # noqa: BLE001
                confidence = 0.0
            score = overlap * 0.65 + confidence * 0.25 + status_weight.get(str(row.get("status")), 0.4) * 0.10
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [row for _, row in scored[offset : offset + limit]]
        return [
            {
                "claim_id": row.get("claim_id"),
                "text": str(row.get("text") or "")[:360],
                "status": row.get("status"),
                "confidence": row.get("confidence"),
                "justification": str(row.get("justification") or "")[:320],
                "source_id": row.get("source_id"),
                "source_url": row.get("source_url"),
                "supporting_claim_ids": row.get("supporting_claim_ids", []),
                "contradicting_claim_ids": row.get("contradicting_claim_ids", []),
            }
            for row in selected
        ]

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"\b[\w-]+\b", text))

    def _claims_table_markdown(self, rows: list[dict[str, Any]], title: str) -> str:
        if not rows:
            return f"#### {title}\n\nAucun element."

        lines = [f"#### {title}", "", "| Claim ID | Statut | Confiance | Source | Claim |", "|---|---:|---:|---|---|"]
        for row in rows:
            confidence = row.get("confidence", 0)
            try:
                confidence = float(confidence)
            except Exception:  # noqa: BLE001
                confidence = 0.0
            source_url = row.get("source_url") or ""
            source_cell = f"[{row.get('source_id')}]({source_url})" if source_url else str(row.get("source_id") or "")
            claim_text = str(row.get("text") or "").replace("|", "\\|")
            lines.append(
                f"| `{row.get('claim_id')}` | {row.get('status')} | {confidence:.2f} | {source_cell} | {claim_text[:220]} |"
            )
        return "\n".join(lines)

    async def _emit(
        self,
        progress_cb: ProgressCallback | None,
        run_dir: Path,
        stage: str,
        message: str,
        **extra: Any,
    ) -> None:
        payload = {
            "timestamp": int(time.time()),
            "stage": stage,
            "message": message,
            **extra,
        }

        status = self._load_status(run_dir)
        status["updated_at"] = int(time.time())
        status["stage"] = stage
        events = status.get("events", [])
        events.append(payload)
        status["events"] = events[-300:]
        self._write_status(run_dir, status)

        if progress_cb:
            maybe_awaitable = progress_cb(payload)
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable

    def _new_run_id(self, question: str) -> str:
        short_hash = hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]
        return f"run-{int(time.time())}-{short_hash}"

    def _status_path(self, run_dir: Path) -> Path:
        return run_dir / "status.json"

    def _load_status(self, run_dir: Path) -> dict[str, Any]:
        status_path = self._status_path(run_dir)
        if not status_path.exists():
            return {
                "run_id": run_dir.name,
                "state": "created",
                "started_at": int(time.time()),
                "updated_at": int(time.time()),
                "question": "",
                "stage": "created",
                "events": [],
                "error": None,
            }
        try:
            return json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {
                "run_id": run_dir.name,
                "state": "corrupted",
                "started_at": int(time.time()),
                "updated_at": int(time.time()),
                "question": "",
                "stage": "corrupted",
                "events": [],
                "error": "status.json unreadable",
            }

    def _write_status(self, run_dir: Path, payload: dict[str, Any]) -> None:
        self._status_path(run_dir).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
