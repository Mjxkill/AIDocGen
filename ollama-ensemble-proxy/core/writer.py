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

    async def plan_dossier(self, question: str, detail_level: str, llm_logs: list[dict[str, Any]], prompt_type: str = "generic") -> dict[str, Any]:
        p_path = Path("prompts") / f"planner_{prompt_type}.txt"
        if not p_path.exists(): p_path = Path("prompts") / "planner_generic.txt"
        
        sys_prompt = p_path.read_text(encoding="utf-8").replace("{question}", question).replace("{detail_level}", detail_level)
        user_prompt = f"Plan accurate structure for: {question}."
        
        raw = await self.llm.ask(self.config.planner_model, sys_prompt, user_prompt, llm_logs, "planner", temperature=0.2)
        return await self.llm.parse_json(raw, self.config.planner_model, "planner", llm_logs)

    async def write_sections(self, planner: dict[str, Any], claims: list[dict[str, Any]], llm_logs: list[dict[str, Any]], progress_cb: Callable | None, run_dir: Path | None) -> dict[str, Any]:
        outline = planner.get("master_outline", [])
        sections_content = []
        tasks = []
        for p in outline:
            for c in p.get("chapters", []):
                for s in c.get("sub_sections", []):
                    tasks.append((p.get("party_title", "Partie"), c.get("chapter_title", "Chapitre"), s))
        
        for idx, (p_title, c_title, sec) in enumerate(tasks, 1):
            s_title = sec.get("title")
            if progress_cb:
                await emit_progress(progress_cb, run_dir, "writing", f"Writing {idx}/{len(tasks)}: {s_title}")
            
            keywords = s_title.lower().split()
            relevant = [c for c in claims if any(k in c.get("claim_text", "").lower() for k in keywords[:3])]
            context = json.dumps(relevant[:30], ensure_ascii=False)

            prompt = f"Dissertation Section: {s_title}\nPartie: {p_title}\nChapitre: {c_title}\nBrief: {sec.get('brief')}\nContext: {context}\n\nRules: Technical depth, min 1000 words, cite [CLM-id]."
            content = await self.llm.ask(self.config.writer_model, "Academic Writer", prompt, llm_logs, "writing")
            sections_content.append({"type": "section", "p_title": p_title, "c_title": c_title, "s_title": s_title, "content": content})

        return {"sections": sections_content}

    async def assemble_report(self, planner: dict[str, Any], sections_payload: dict[str, Any], claims: list[dict[str, Any]], verdicts: dict[str, Any], corpus: dict[str, Any]) -> tuple[str, str]:
        title = planner.get("question_reformulated", "Dossier")
        sections = sections_payload.get("sections", [])
        
        # 1. Table of Contents
        toc = ["## Table des Matières\n"]
        current_p, current_c = "", ""
        for s in sections:
            if s["p_title"] != current_p:
                current_p = s["p_title"]
                toc.append(f"- [{current_p}](#{markdown_anchor(current_p)})")
            if s["c_title"] != current_c:
                current_c = s["c_title"]
                toc.append(f"  - [{current_c}](#{markdown_anchor(current_c)})")
            toc.append(f"    - [{s['s_title']}](#{markdown_anchor(s['s_title'])})")

        # 2. Main Report
        body = [f"# {title}\n", "\n".join(toc), "\n---\n"]
        current_p, current_c = "", ""
        for s in sections:
            if s["p_title"] != current_p:
                current_p = s["p_title"]
                body.append(f"\n# {current_p}\n<a name='{markdown_anchor(current_p)}'></a>")
            if s["c_title"] != current_c:
                current_c = s["c_title"]
                body.append(f"\n## {current_c}\n<a name='{markdown_anchor(current_c)}'></a>")
            
            content = s["content"]
            # REFINEMENT: Citations [CLM-xxx] -> [[*]](annexes.md#CLM-xxx)
            content = re.sub(r"\[CLM-([a-f0-9-]+)\]", r"[[*]](annexes.md#CLM-\1)", content)
            body.append(f"\n### {s['s_title']}\n<a name='{markdown_anchor(s['s_title'])}'></a>\n\n{content}\n")

        # 3. Annexes
        annex = ["# Annexes Techniques : Preuves\n", "> Retrouvez ici les sources et validations détaillées.\n\n"]
        v_dict = {v["claim_id"]: v for v in verdicts.get("verdicts", [])}
        src_dict = {s["source_id"]: s for s in corpus.get("sources", [])}
        
        for c in claims:
            cid = c["claim_id"]
            v = v_dict.get(cid, {})
            src = src_dict.get(c.get("source_id"), {})
            annex.append(f"<a name='CLM-{cid}'></a>\n### Preuve {cid}\n- **Fait :** {c['claim_text']}\n- **Status :** {v.get('status', 'UNCERTAIN')}\n- **Source :** [{src.get('title') or 'Lien'}]({src.get('canonical_url', '#')})\n")
            if v.get("justification"): annex.append(f"- **Analyse :** {v['justification']}\n")
            annex.append("\n---\n")

        return "\n".join(body), "\n".join(annex)