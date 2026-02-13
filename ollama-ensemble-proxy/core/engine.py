import time
import json
from pathlib import Path
from typing import Any, Callable
from .config import DossierConfig
from .llm import LLMClient
from .research import WebResearcher
from .analysis import Analyst
from .writer import Writer
from .utils import save_json, load_json, emit_progress

class DossierEngine:
    def __init__(self, config: DossierConfig):
        self.config = config
        self.llm = LLMClient(config)
        self.researcher = WebResearcher(config)
        self.analyst = Analyst(config, self.llm)
        self.writer = Writer(config, self.llm)

    async def run(self, run_id: str, question: str, prompt_type: str = "generic", detail_level: str = "medium", resume: bool = False) -> dict[str, Any]:
        run_dir = Path(self.config.data_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Update status to running
        s_path = run_dir / "status.json"
        data = load_json(s_path) or {}
        data["state"] = "running"
        data["updated_at"] = int(time.time())
        save_json(s_path, data)
        
        llm_logs = []
        if resume:
            logs = load_json(run_dir / "llm_logs.json")
            if logs: llm_logs = logs

        try:
            # 1. PRESEARCH
            pre = await self._step(run_dir, "presearch", resume, None, lambda: self.researcher.presearch(question))
            # 2. PLANNER
            pla = await self._step(run_dir, "planner", resume, None, lambda: self.writer.plan_dossier(question, detail_level, llm_logs, prompt_type))
            
            # 3. PAUSE
            if not (run_dir / "validated.txt").exists():
                await emit_progress(None, run_dir, "awaiting_validation", "Plan ready. Awaiting approval.")
                return {"status": "paused"}

            # 4. SEARCH
            await emit_progress(None, run_dir, "search", "Démarrage de la recherche web...")
            sea = await self._step(run_dir, "search_results", resume, None, lambda: self.researcher.search_subquestions(pla, None, run_dir))
            
            # 5. CORPUS
            await emit_progress(None, run_dir, "corpus", f"Téléchargement de {len(sea.get('sub_questions', []))} sets de résultats...")
            cor = await self._step(run_dir, "corpus", resume, None, lambda: self.researcher.build_corpus(sea, None, run_dir))
            
            # Update status with source count
            s_path = run_dir / "status.json"
            status_data = load_json(s_path) or {}
            status_data["sources_count"] = cor.get("count", 0)
            save_json(s_path, status_data)

            if not cor.get("sources") or cor.get("count", 0) == 0:
                raise ValueError("Aucune source n'a pu être trouvée ou téléchargée. Recherche infructueuse.")

            # 6. RANKING
            await emit_progress(None, run_dir, "shortlist", "Classement des sources par pertinence...")
            shor = await self._step(run_dir, "shortlist", resume, None, lambda: self.analyst.rank_sources(pla, cor, llm_logs, None, run_dir))
            # 7. CLAIMS
            await emit_progress(None, run_dir, "claims", "Extraction des affirmations factuelles...")
            clm = await self._step(run_dir, "claims", resume, None, lambda: self.analyst.extract_claims(pla, shor, cor, llm_logs, None, run_dir))
            
            # Update status with claims count
            status_data = load_json(s_path) or {}
            status_data["claims_count"] = len(clm.get("claims", []))
            save_json(s_path, status_data)

            # 8. VERIFICATION
            await emit_progress(None, run_dir, "verdicts", f"Vérification de {len(clm['claims'])} affirmations...")
            ver = await self._step(run_dir, "verdicts", resume, None, lambda: self.analyst.verify_claims(clm["claims"], llm_logs, None, run_dir))
            # 9. WRITING
            await emit_progress(None, run_dir, "sections", "Rédaction des chapitres du dossier...")
            sec = await self._step(run_dir, "sections", resume, None, lambda: self.writer.write_sections(pla, clm["claims"], llm_logs, None, run_dir))

            # 10. ASSEMBLY
            report_md, annex_md = await self.writer.assemble_report(pla, sec, clm["claims"], ver, cor)
            (run_dir / "report.md").write_text(report_md, encoding="utf-8")
            (run_dir / "annexes.md").write_text(annex_md, encoding="utf-8")
            
            # 11. LATEX & PDF EXPORT
            try:
                from export_latex import generate_latex
                generate_latex(run_id, data_dir=self.config.data_dir)
            except Exception as e:
                print(f"Latex/PDF export failed: {e}")
            
            # FINALIZE
            await emit_progress(None, run_dir, "completed", "Dossier completed.")
            s_path = run_dir / "status.json"
            data = load_json(s_path) or {}
            data["state"] = "completed"
            save_json(s_path, data)
            save_json(run_dir / "llm_logs.json", llm_logs)
            return {"status": "completed"}

        except Exception as e:
            await emit_progress(None, run_dir, "failed", str(e))
            s_path = run_dir / "status.json"
            data = load_json(s_path) or {}
            data["state"] = "failed"
            data["error"] = str(e)
            save_json(s_path, data)
            raise

    async def _step(self, run_dir: Path, name: str, resume: bool, cb, func):
        path = run_dir / f"{name}.json"
        if resume and path.exists():
            data = load_json(path)
            if data: return data
        result = await func()
        save_json(path, result)
        return result