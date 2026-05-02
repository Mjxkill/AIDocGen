import json
import sys
import re
from pathlib import Path

def rebuild(run_id):
    path = Path(f"/root/codex/ollama-ensemble-proxy/data/dossiers/{run_id}")
    draft = (path / "planner_draft.txt").read_text(encoding="utf-8")
    
    chapters = []
    current_chapter = None
    chap_regex = r'^(?:#{1,4}|\*\*)\s*(?:Chapitre|Chapter)\s*[\d\.]*\s*[\:\-\.\s—]*\s*(.*?)(?:\*\*|#|$)'
    
    for line in draft.split('\n'):
        line = line.strip()
        if not line: continue
        
        head_match = re.search(chap_regex, line, re.IGNORECASE)
        if head_match:
            title = head_match.group(1).strip()
            if len(title) > 3:
                current_chapter = {"chapter_title": title, "sub_sections": []}
                chapters.append(current_chapter)
            continue
            
        if current_chapter:
            sec_match = re.search(r'^[*-]\s*(?:[\d\.]+\s*)?(.*)', line)
            if sec_match:
                sec_title = sec_match.group(1).replace('**', '').strip()
                if len(sec_title) > 3:
                    current_chapter["sub_sections"].append({"title": sec_title, "brief": "Analyse détaillée."})
    
    final_chapters = [c for c in chapters if len(c["sub_sections"]) > 0]
    res = {
        "question_reformulated": "Plan d'Enquête Scientifique : Qualité des Vaccins en France",
        "master_outline": final_chapters,
        "sub_questions": []
    }
    (path / "planner.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"SUCCESS: Rebuilt {run_id} with {len(final_chapters)} chapters.")

if __name__ == "__main__":
    rebuild("run-1771148267-6b7b35631e")