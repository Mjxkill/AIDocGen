import asyncio
import os
import sys
import json
import re
from pathlib import Path

# Fix path for core modules
base_dir = "/root/codex/ollama-ensemble-proxy"
sys.path.append(base_dir)

from core.config import DossierConfig
from core.llm import LLMClient

async def reformat_report(run_id: str):
    config = DossierConfig.from_env()
    llm = LLMClient(config)
    run_dir = Path(config.data_dir) / run_id
    report_path = run_dir / "report.md"
    status_path = run_dir / "status.json"
    
    if not report_path.exists(): return

    # Detect target language
    target_lang = "fr"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
            target_lang = status.get("language", "fr")
        except: pass
    
    lang_name = {"fr": "Français", "en": "English", "es": "Español", "de": "Deutsch"}.get(target_lang, "Français")
    content = report_path.read_text(encoding="utf-8")
    
    # 1. CLEAN UP CURLY BRACES (often LLM trying to do LaTeX or bolding)
    content = content.replace("{", "").replace("}", "")
    
    # 2. FIX TITLE HIERARCHY
    lines = content.split("\n")
    new_lines = []
    first_h1_found = False
    
    for line in lines:
        if line.startswith("# "):
            if not first_h1_found:
                new_lines.append(line) # Keep document title
                first_h1_found = True
            else:
                new_lines.append("## " + line[2:]) # Downgrade other H1 to H2
        else:
            new_lines.append(line)
    
    content = "\n".join(new_lines)
    
    # 3. TRANSLATE & CLEAN SECTIONS
    sections = re.split(r'\n## ', content)
    reformatted_sections = []
    
    for i, section in enumerate(sections):
        if not section.strip(): continue
        if i > 0: section = "## " + section
        
        # English detection
        english_indicators = [" the ", " and ", " of ", " is ", " in ", " with "]
        is_english = sum(1 for word in english_indicators if word in section.lower()) > 10
        
        if is_english and target_lang != "en":
            print(f"Translating section...")
            prompt = f"Traduis cette section en {lang_name} professionnel. Garde le formatage Markdown. Ne change pas les faits.\n\n{section}"
            translated = await llm.ask(config.writer_model, f"Traducteur expert {lang_name}", prompt)
            section = translated
            
        # Cleanup citations
        section = re.sub(r'\[\[\*\]\]\(annexes\.md#CLM-(\d+)\)', r'[\1]', section)
        section = re.sub(r'\[CLM-C(\d+)\]', r'[C\1]', section)
        
        # Cleanup stray title markers in the content itself
        section = re.sub(r'^#+ (.*)$', r'\1', section, flags=re.MULTILINE) # Remove # from start of lines
        
        if "Table des Matières" in section and i < 2: continue
        reformatted_sections.append(section)

    final_content = "\n\n".join(reformatted_sections)
    report_path.write_text(final_content, encoding="utf-8")
    print(f"Reformatting and cleaning complete for {run_id}")

if __name__ == "__main__":
    if len(sys.argv) > 1: asyncio.run(reformat_report(sys.argv[1]))