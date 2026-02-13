import json
import re
from pathlib import Path

def markdown_to_latex(text):
    if not text: return ""
    # Basic escapes
    text = text.replace("\\", r"\textbackslash{}")
    text = text.replace("_", r"\_")
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("$", r"\$")
    text = text.replace("#", r"\#")
    text = text.replace("{", r"\{")
    text = text.replace("}", r"\}")
    text = text.replace("~", r"\textasciitilde{}")
    text = text.replace("^", r"\textasciicircum{}")
    
    # Bold / Italic
    text = re.sub(r"\*\*(.*?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"\*(.*?)\*", r"\\textit{\1}", text)
    
    # Links
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1 (\\url{\2})", text)
    
    return text

def generate_latex(run_id, data_dir="data/dossiers"):
    base_path = Path(data_dir) / run_id
    planner = json.loads((base_path / "planner.json").read_text())
    sections_data = json.loads((base_path / "sections.json").read_text())
    sections = sections_data.get("sections", [])
    
    claims_data = json.loads((base_path / "claims.json").read_text())
    claims = claims_data.get("claims", [])
    
    verdicts_data = json.loads((base_path / "verdicts.json").read_text())
    verdicts_list = verdicts_data.get("verdicts", [])
    verdicts = {v["claim_id"]: v for v in verdicts_list}
    
    corpus_data = json.loads((base_path / "corpus.json").read_text())
    corpus_sources = corpus_data.get("sources", []) if isinstance(corpus_data, dict) else []
    sources = {s["source_id"]: s for s in corpus_sources}

    title = planner.get("question_reformulated", "Dossier de Recherche")
    
    latex = [
        r"\documentclass[11pt,a4paper]{report}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[french]{babel}",
        r"\usepackage{hyperref}",
        r"\usepackage{geometry}",
        r"\usepackage{url}",
        r"\geometry{margin=2.5cm}",
        f"\\title{{{markdown_to_latex(title)}}}",
        r"\author{Ollama Ensemble Proxy}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage"
    ]

    for s in sections:
        # Utiliser 'title' ou 's_title' selon ce qui existe
        s_title = s.get("title") or s.get("s_title") or "Section"
        latex.append(f"\\chapter{{{markdown_to_latex(s_title)}}}")
        
        # Utiliser 'markdown' ou 'content'
        content = s.get("markdown") or s.get("content") or ""
        # Remplacer les citations par des notes de bas de page
        content = re.sub(r"\[CLM-([a-f0-9-]+)\]", r"\\footnote{Preuve CLM-\1}", content)
        latex.append(markdown_to_latex(content))
        latex.append(r"\newpage")

    # Annexes
    latex.append(r"\appendix")
    latex.append(r"\chapter{Annexes Techniques : Preuves et Sources}")
    latex.append(r"\begin{description}")
    
    # On limite pour éviter un document trop massif, mais on en met quand même pas mal
    for c in claims[:1500]: 
        cid = c["claim_id"]
        v = verdicts.get(cid, {})
        src = sources.get(c.get("source_id"), {})
        
        status = v.get("status", "INCERTAIN")
        
        latex.append(f"\\item[Preuve {cid}] \\hfill \\\\")
        latex.append(f"\\textbf{{Fait :}} {markdown_to_latex(c['claim_text'])} \\\\")
        latex.append(f"\\textbf{{Statut :}} {status} \\\\")
        latex.append(f"\\textbf{{Source :}} {markdown_to_latex(src.get('title', 'N/A'))} (\\url{{{src.get('canonical_url', '#')}}})")
        if v.get("justification"):
            latex.append(f"\\\\ \\textbf{{Analyse :}} {markdown_to_latex(v['justification'])}")
        latex.append(r"\hrulefill")

    latex.append(r"\end{description}")
    latex.append(r"\end{document}")

    output_path = base_path / "report.tex"
    output_path.write_text("\n".join(latex), encoding="utf-8")
    print(f"LaTeX generated: {output_path}")

    # Automatic PDF conversion
    try:
        import subprocess
        print(f"Converting to PDF: {run_id}...")
        # Run pdflatex twice for TOC and references
        for _ in range(2):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "report.tex"],
                cwd=base_path,
                capture_output=True,
                text=True
            )
        if result.returncode == 0:
            print(f"PDF successfully generated: {base_path / 'report.pdf'}")
        else:
            print(f"PDF conversion failed with return code {result.returncode}")
            # print(result.stdout) # Optional: for debugging
    except Exception as e:
        print(f"Error during PDF conversion: {e}")

if __name__ == "__main__":
    import sys
    run_id = sys.argv[1] if len(sys.argv) > 1 else "run-1770937767-4f34fbbd4e"
    generate_latex(run_id)
