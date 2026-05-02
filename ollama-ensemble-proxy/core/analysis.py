import json
import asyncio
import uuid
from datetime import date
from typing import Any, Callable
from pathlib import Path
from .config import DossierConfig
from .llm import LLMClient
from .utils import emit_progress
from .logging_config import get_logger

log = get_logger("aidocgen.analysis")

def _date_instruction() -> str:
    today = date.today().strftime("%d/%m/%Y")
    return f"Nous sommes le {today}. Priorise les informations les plus récentes."

class Analyst:
    def __init__(self, config: DossierConfig, llm: LLMClient):
        self.config = config
        self.llm = llm

    async def rank_sources(
        self,
        planner: dict[str, Any],
        corpus: dict[str, Any],
        llm_logs: list[dict[str, Any]],
        progress_cb: Callable | None,
        run_dir: Path | None,
    ) -> dict[str, Any]:
        """Rank sources using real LLM multi-criteria scoring."""
        sources = corpus.get("sources", [])
        question = planner.get("question_reformulated", "")

        # Build source summaries with actual content snippets
        source_summaries = []
        for s in sources:
            content_preview = ""
            if run_dir:
                content_path = run_dir / s.get("content_path", "")
                if content_path.exists():
                    content_preview = content_path.read_text(encoding="utf-8", errors="ignore")[:2000]

            source_summaries.append({
                "source_id": s["source_id"],
                "url": s.get("url", ""),
                "title": s.get("title", ""),
                "domain": s.get("domain", ""),
                "content_preview": content_preview[:1500],
            })

        # Batch processing with real LLM scoring
        batch_size = 10
        ranked_results = []

        for i in range(0, len(source_summaries), batch_size):
            batch = source_summaries[i : i + batch_size]
            if progress_cb and run_dir:
                await emit_progress(progress_cb, run_dir, "ranking",
                    f"Ranking sources {i+1}-{min(i+batch_size, len(source_summaries))}/{len(source_summaries)}")

            # Build detailed prompt for LLM ranking
            sources_desc = []
            for idx, s in enumerate(batch, 1):
                sources_desc.append(
                    f"--- Source {idx} (ID: {s['source_id']}) ---\n"
                    f"URL: {s['url']}\nTitle: {s['title']}\nDomain: {s['domain']}\n"
                    f"Content preview:\n{s['content_preview'][:800]}\n"
                )

            system_prompt = (
                f"Tu es un évaluateur de sources pour la recherche documentaire. {_date_instruction()}\n\n"
                f"Évalue chaque source selon ces 4 critères (chaque critère de 0.0 à 1.0):\n"
                f"1. **pertinence**: La source traite-t-elle directement du sujet ?\n"
                f"2. **fraicheur**: Les informations sont-elles récentes et à jour ?\n"
                f"3. **autorite**: Le domaine/auteur est-il fiable et reconnu ?\n"
                f"4. **profondeur**: La source offre-t-elle des détails techniques ou juste du survol ?\n\n"
                f"Retourne un JSON strict:\n"
                f'{{"rankings": [{{"source_id": "...", "pertinence": 0.0, "fraicheur": 0.0, "autorite": 0.0, "profondeur": 0.0, "score_global": 0.0, "justification": "..."}}]}}'
            )

            user_prompt = (
                f"Question de recherche: {question}\n\n"
                f"Sources à évaluer:\n\n" + "\n\n".join(sources_desc)
            )

            try:
                resp = await self.llm.ask(self.config.judge_model, system_prompt, user_prompt, llm_logs, "ranking")
                data = await self.llm.parse_json(resp, self.config.judge_model, "ranking", llm_logs)

                rankings = data.get("rankings", [])
                # Map rankings by source_id
                rank_map = {r["source_id"]: r for r in rankings if "source_id" in r}

                for s in batch:
                    sid = s["source_id"]
                    if sid in rank_map:
                        r = rank_map[sid]
                        # Compute weighted global score if not provided
                        score = r.get("score_global")
                        if not score or score == 0:
                            score = (
                                r.get("pertinence", 0.5) * 0.4 +
                                r.get("fraicheur", 0.5) * 0.2 +
                                r.get("autorite", 0.5) * 0.2 +
                                r.get("profondeur", 0.5) * 0.2
                            )
                        ranked_results.append({
                            "source_id": sid,
                            "score": round(score, 3),
                            "pertinence": r.get("pertinence", 0.5),
                            "fraicheur": r.get("fraicheur", 0.5),
                            "autorite": r.get("autorite", 0.5),
                            "profondeur": r.get("profondeur", 0.5),
                            "justification": r.get("justification", ""),
                        })
                    else:
                        # Source not ranked by LLM — assign neutral score
                        ranked_results.append({"source_id": sid, "score": 0.5})
            except Exception as e:
                log.warning("Ranking batch  failed", extra={"i": i, "error": str(e)})
                # Graceful fallback: assign neutral scores
                for s in batch:
                    ranked_results.append({"source_id": s["source_id"], "score": 0.5})

        # Sort by score descending
        ranked_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        if run_dir:
            await emit_progress(progress_cb, run_dir, "ranking",
                f"Ranking complete: top score {ranked_results[0]['score']:.2f}, "
                f"bottom {ranked_results[-1]['score']:.2f}" if ranked_results else "No sources to rank")

        return {"shortlist": ranked_results}

    async def extract_claims(
        self,
        planner: dict[str, Any],
        shortlist: dict[str, Any],
        corpus: dict[str, Any],
        llm_logs: list[dict[str, Any]],
        progress_cb: Callable | None,
        run_dir: Path,
    ) -> dict[str, Any]:
        """Extract factual claims from shortlisted sources."""
        sources = corpus.get("sources", [])
        source_map = {s["source_id"]: s for s in sources}

        # Take top 30 sources from shortlist for extraction
        top_ids = [item["source_id"] for item in shortlist.get("shortlist", [])[:30]]
        all_claims = []

        semaphore = asyncio.Semaphore(self.config.max_parallel_llm)
        processed = 0

        async def extract_one(sid: str):
            nonlocal processed
            async with semaphore:
                source = source_map.get(sid)
                if not source: return []

                content_path = run_dir / source["content_path"]
                if not content_path.exists(): return []

                # Increased cap: 20K chars for richer extraction
                text = content_path.read_text(encoding="utf-8", errors="ignore")[:20000]

                system = (
                    f"You are an expert fact extractor. {_date_instruction()}\n"
                    f"Extract 5-15 factual, verifiable claims from the text.\n"
                    f"For each claim, classify its type: 'statistic', 'definition', 'comparison', 'specification', 'opinion', 'event', 'methodology'.\n"
                    f"Return JSON: {{\"claims\": [{{\"claim_id\": \"unique-id\", \"claim_text\": \"...\", \"claim_type\": \"...\", \"confidence\": 0.0-1.0}}]}}"
                )
                prompt = f"Source: {source.get('title', 'Unknown')} ({source.get('url', '')})\n\nText:\n{text}"

                try:
                    resp = await self.llm.ask(self.config.extract_model, system, prompt, llm_logs, "claims")
                    data = await self.llm.parse_json(resp, self.config.extract_model, "claims", llm_logs)
                    claims = data.get("claims", [])
                    for c in claims:
                        c["source_id"] = sid
                        if not c.get("claim_id"): c["claim_id"] = f"{sid}-{uuid.uuid4().hex[:6]}"
                    return claims
                except Exception:
                    return []
                finally:
                    processed += 1
                    if progress_cb:
                        await emit_progress(progress_cb, run_dir, "claims", f"Extracting claims {processed}/{len(top_ids)}")

        results = await asyncio.gather(*(extract_one(sid) for sid in top_ids))
        for res in results:
            all_claims.extend(res)

        # ── COHERENCE CHECK: cross-reference claims via embeddings ──
        coherence_summary = None
        try:
            from .coherence import CoherenceIndex
            embed_model = getattr(self.config, 'coherence_embed_model', 'mxbai-embed-large')
            index = CoherenceIndex(
                ollama_url="http://127.0.0.1:11434",
                embed_model=embed_model,
                similarity_threshold=0.80,
            )
            await emit_progress(progress_cb, run_dir, "coherence", f"Coherence check: {len(all_claims)} claims...")

            checked_claims = []
            for i, claim in enumerate(all_claims):
                enriched = await index.check_and_add(claim)
                checked_claims.append(enriched)
                if (i + 1) % 50 == 0:
                    await emit_progress(progress_cb, run_dir, "coherence",
                        f"Coherence {i+1}/{len(all_claims)}: {len(index.conflicts)} conflicts, {len(index.confirmations)} confirmed")

            all_claims = checked_claims
            coherence_summary = index.get_summary()

            await emit_progress(progress_cb, run_dir, "coherence",
                f"Coherence done: {coherence_summary['conflicts_found']} conflicts, {coherence_summary['confirmations_found']} confirmations")

            # Save coherence report
            if run_dir:
                import json
                (run_dir / "coherence.json").write_text(
                    json.dumps(coherence_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
        except Exception as e:
            log.warning("Coherence check failed (non-fatal)", extra={"error": str(e)})

        return {"claims": all_claims, "coherence": coherence_summary}

    async def verify_claims(
        self,
        claims: list[dict[str, Any]],
        llm_logs: list[dict[str, Any]],
        progress_cb: Callable | None,
        run_dir: Path | None,
    ) -> dict[str, Any]:
        # Verification with Checkpointing (Partial Save)
        verdicts = []
        partial_file = run_dir / "verdicts.partial.json" if run_dir else None

        if partial_file and partial_file.exists():
            try:
                verdicts = json.loads(partial_file.read_text(encoding="utf-8"))
            except: pass

        processed = len(verdicts)
        to_verify = claims[processed:]

        semaphore = asyncio.Semaphore(self.config.max_parallel_llm)

        async def verify_one(claim: dict[str, Any]):
            async with semaphore:
                claim_text = claim.get("claim_text") or claim.get("claim_product_image") or str(claim)
                system = (
                    f"You are a factual judge. {_date_instruction()}\n"
                    f"Verify the following claim. Consider:\n"
                    f"- Is it factually accurate based on your knowledge?\n"
                    f"- Is it verifiable or just opinion?\n"
                    f"- Is it up to date?\n"
                    f"Return JSON: {{\"status\": \"ACCEPTED|REJECTED|UNCERTAIN\", \"justification\": \"...\", \"confidence\": 0.0-1.0}}"
                )
                prompt = f"Claim to verify: {claim_text}"
                try:
                    resp = await self.llm.ask(self.config.verify_model, system, prompt, llm_logs, "verification")
                    data = await self.llm.parse_json(resp, self.config.verify_model, "verification", llm_logs)
                    return {
                        "claim_id": claim["claim_id"],
                        "status": data.get("status", "UNCERTAIN"),
                        "justification": data.get("justification", "No justification provided."),
                        "confidence": data.get("confidence", 0.5),
                    }
                except Exception:
                    return {"claim_id": claim["claim_id"], "status": "UNCERTAIN", "justification": "Error during verification."}

        # Process in small batches to allow checkpointing
        batch_size = 5
        for i in range(0, len(to_verify), batch_size):
            batch = to_verify[i : i + batch_size]
            results = await asyncio.gather(*(verify_one(c) for c in batch))
            verdicts.extend(results)

            if partial_file:
                partial_file.write_text(json.dumps(verdicts, indent=2, ensure_ascii=False), encoding="utf-8")

            if progress_cb and run_dir:
                await emit_progress(progress_cb, run_dir, "verification", f"Verified {len(verdicts)}/{len(claims)} claims")

        if partial_file: partial_file.unlink(missing_ok=True)
        return {"verdicts": verdicts}