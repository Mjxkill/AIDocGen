import json
import re
import subprocess
from pathlib import Path

def escape_latex_special_chars(text):
    """Escape LaTeX special characters (minimal set for regular text)."""
    if not text: return ""
    # Order matters: & must be escaped first
    replacements = [
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    # Escape underscores not already escaped
    text = re.sub(r'(?<!\\)_', r'\_', text)
    return text

def convert_markdown_inline(text):
    """Convert markdown inline formatting to LaTeX."""
    if not text: return ""
    
    # Remove markdown links: [text](url) -> text
    text = re.sub(r'\[\[?\*?\]?\]\([^)]+\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Handle bold and italic: ***text***
    text = re.sub(r'\*\*\*([^*]+?)\*\*\*', r'\\textbf{\\textit{\1}}', text)
    # Handle bold: **text**
    text = re.sub(r'\*\*([^*]+?)\*\*', r'\\textbf{\1}', text)
    # Handle italic: *text*
    text = re.sub(r'\*([^*]+?)\*', r'\\textit{\1}', text)
    
    return text

def process_inline_text(text):
    """Full processing: markdown + escape special chars."""
    if not text: return ""
    text = convert_markdown_inline(text)
    text = escape_latex_special_chars(text)
    return text

def estimate_text_width(text):
    """Estimate text width in cm."""
    if not text: return 0
    return len(text) * 0.22

def convert_table_cell(cell):
    """Process a table cell content."""
    if not cell: return ""
    
    # Handle HTML tags first
    cell = cell.replace('<br>', r' \\ ')
    cell = cell.replace('<BR>', r' \\ ')
    cell = re.sub(r'<[^>]+>', '', cell)  # Remove other HTML tags
    
    # Remove markdown links
    cell = re.sub(r'\[\[?\*?\]?\]\([^)]+\)', '', cell)
    cell = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cell)
    
    # Convert markdown formatting
    cell = re.sub(r'\*\*\*([^*]+?)\*\*\*', r'\\textbf{\\textit{\1}}', cell)
    cell = re.sub(r'\*\*([^*]+?)\*\*', r'\\textbf{\1}', cell)
    cell = re.sub(r'\*([^*]+?)\*', r'\\textit{\1}', cell)
    
    # Escape special chars
    cell = cell.replace('&', r'\&')
    cell = cell.replace('%', r'\%')
    cell = cell.replace('$', r'\$')
    cell = cell.replace('#', r'\#')
    cell = re.sub(r'(?<!\\)_', r'\_', cell)
    
    return cell

def convert_markdown_table_to_latex(table_lines, page_width=15.0):
    """Convert Markdown table lines to LaTeX tabular environment."""
    if not table_lines or len(table_lines) < 2:
        return None
    
    rows = []
    for line in table_lines:
        line = line.strip()
        if not line or not line.startswith('|'):
            continue
        # Skip separator lines
        if re.match(r'^\|[\s\-:|]+\|\s*$', line):
            continue
        cells = line.split('|')
        cells = [c.strip() for c in cells]
        cells = [c for c in cells if c]
        if cells:
            rows.append(cells)
    
    if not rows:
        return None
    
    num_cols = max(len(row) for row in rows)
    
    # Pad rows to same length
    for row in rows:
        while len(row) < num_cols:
            row.append('')
    
    # Calculate column widths
    col_widths = []
    for col_idx in range(num_cols):
        max_width = max(estimate_text_width(row[col_idx]) for row in rows)
        col_widths.append(max_width)
    
    total_width = sum(col_widths)
    use_tabularx = total_width > page_width
    col_spec = 'X' * num_cols if use_tabularx else 'l' * num_cols
    
    latex_lines = []
    latex_lines.append(r'\begin{table}[htbp]')
    latex_lines.append(r'\centering')
    latex_lines.append(r'\small')
    
    if use_tabularx:
        latex_lines.append(f'\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}')
    else:
        latex_lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    
    latex_lines.append(r'\toprule')
    
    for i, row in enumerate(rows):
        cells = [convert_table_cell(cell) for cell in row]
        latex_lines.append(' & '.join(cells) + r' \\')
        if i == 0:
            latex_lines.append(r'\midrule')
    
    latex_lines.append(r'\bottomrule')
    
    if use_tabularx:
        latex_lines.append(r'\end{tabularx}')
    else:
        latex_lines.append(r'\end{tabular}')
    
    latex_lines.append(r'\end{table}')
    latex_lines.append('')
    
    return '\n'.join(latex_lines)

def process_body(text):
    """Process body text, converting markdown to LaTeX."""
    if not text: return ""
    
    lines = text.split('\n')
    result_lines = []
    table_buffer = []
    in_table = False
    in_list = False
    in_code = False
    code_lang = ""
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # === CODE BLOCKS ===
        if line.strip().startswith('```'):
            if not in_code:
                # Start code block
                if in_list:
                    result_lines.append(r'\end{itemize}')
                    result_lines.append('')
                    in_list = False
                if in_table and table_buffer:
                    latex_table = convert_markdown_table_to_latex(table_buffer)
                    if latex_table:
                        result_lines.append(latex_table)
                    table_buffer = []
                    in_table = False
                
                in_code = True
                code_lang = line.strip()[3:].strip()  # Extract language
                result_lines.append(r'\begin{lstlisting}[basicstyle=\ttfamily\small]')
            else:
                # End code block
                in_code = False
                result_lines.append(r'\end{lstlisting}')
                result_lines.append('')
            i += 1
            continue
        
        if in_code:
            # Inside code block - output raw, only escape minimal chars
            code_line = line
            # Escape backslash first, then other special chars
            code_line = code_line.replace('\\', r'\textbackslash{}')
            code_line = code_line.replace('{', r'\{')
            code_line = code_line.replace('}', r'\}')
            # Don't escape _ & % $ # in code (they should appear as-is in lstlisting)
            result_lines.append(code_line)
            i += 1
            continue
        
        # === TABLES ===
        is_table_line = line.strip().startswith('|') and line.count('|') >= 2
        
        if is_table_line:
            if in_list:
                result_lines.append(r'\end{itemize}')
                result_lines.append('')
                in_list = False
            in_table = True
            table_buffer.append(line)
            i += 1
            continue
        else:
            if in_table and table_buffer:
                latex_table = convert_markdown_table_to_latex(table_buffer)
                if latex_table:
                    result_lines.append(latex_table)
                else:
                    # Fallback: output raw lines
                    for tbl in table_buffer:
                        result_lines.append(process_inline_text(tbl))
                table_buffer = []
                in_table = False
        
        # Skip markdown headers
        if re.match(r'^\s*#{1,6}\s', line):
            i += 1
            continue
        
        # === LISTS ===
        is_list_item = line.strip().startswith('- ')
        is_empty = line.strip() == ''
        
        if is_list_item:
            if not in_list:
                result_lines.append(r'\begin{itemize}')
                in_list = True
            item_text = line.strip()[2:]  # Remove '- '
            item_text = process_inline_text(item_text)
            result_lines.append(f'  \\item {item_text}')
        elif is_empty:
            if in_list:
                result_lines.append(r'\end{itemize}')
                result_lines.append('')
                in_list = False
            else:
                result_lines.append('')
        else:
            if in_list:
                # Check if continuation (indented)
                if line.startswith('  ') and line.strip():
                    cont_text = process_inline_text(line.strip())
                    result_lines.append(f'  {cont_text}')
                else:
                    result_lines.append(r'\end{itemize}')
                    result_lines.append('')
                    in_list = False
                    result_lines.append(process_inline_text(line))
            else:
                result_lines.append(process_inline_text(line))
        
        i += 1
    
    # Close any remaining open environments
    if in_list:
        result_lines.append(r'\end{itemize}')
    if in_code:
        result_lines.append(r'\end{lstlisting}')
    if in_table and table_buffer:
        latex_table = convert_markdown_table_to_latex(table_buffer)
        if latex_table:
            result_lines.append(latex_table)
    
    return '\n'.join(result_lines)

def clean_title(text):
    if not text: return ""
    text = re.sub(r'#+', '', text)
    text = text.replace('{', '').replace('}', '')
    text = text.replace('_', r'\_')
    return text.strip()

def generate_latex(run_id, data_dir="data/dossiers"):
    base_path = Path(data_dir) / run_id
    planner = json.loads((base_path / "planner.json").read_text())
    sections = json.loads((base_path / "sections.json").read_text())["sections"]
    
    doc_title = clean_title(planner.get("question_reformulated", "Dossier de Recherche"))
    
    latex = [
        r"\documentclass[11pt,a4paper]{report}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[french]{babel}",
        r"\usepackage{hyperref}",
        r"\usepackage{geometry}",
        r"\usepackage{url}",
        r"\usepackage{longtable}",
        r"\usepackage{booktabs}",
        r"\usepackage{tabularx}",
        r"\usepackage{amsmath}",
        r"\usepackage{listings}",
        r"\lstset{breaklines=true,frame=single,backgroundcolor=\color{gray!10}}",
        r"\usepackage{xcolor}",
        r"\geometry{margin=2.5cm}",
        f"\\title{{{clean_title(doc_title)}}}",
        r"\author{AIDocGen}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\newpage"
    ]

    current_party = ""
    current_chapter = ""

    for s in sections:
        p_title = clean_title(s.get("p_title", ""))
        c_title = clean_title(s.get("c_title", "Chapitre"))
        s_title = clean_title(s.get("s_title", "Section"))
        
        if p_title and p_title != current_party:
            latex.append(f"\\chapter{{{clean_title(p_title)}}}")
            current_party = p_title
            current_chapter = ""
            
        if c_title and c_title != current_chapter:
            if not p_title:
                latex.append(f"\\chapter{{{clean_title(c_title)}}}")
            else:
                latex.append(f"\\section{{{clean_title(c_title)}}}")
            current_chapter = c_title
            
        if not p_title:
            latex.append(f"\\section{{{clean_title(s_title)}}}")
        else:
            latex.append(f"\\subsection{{{clean_title(s_title)}}}")
            
        latex.append(process_body(s.get("content", "")))
        latex.append("\n")

    latex.append(r"\end{document}")
    (base_path / "report.tex").write_text("\n".join(latex), encoding="utf-8")
    
    for _ in range(3):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], cwd=base_path, capture_output=True)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1: generate_latex(sys.argv[1])
