import json
import sys
import re
from pathlib import Path

base_dir = "/root/codex/ollama-ensemble-proxy"
sys.path.append(base_dir)
from core.writer import Writer

# THE REAL MARKDOWN THAT CAUSED TRUNCATION (Sample)
broken_markdown = """
# Plan d'Enquête
## Introduction
*   Contexte.
*   Méthodologie.

### Chapitre 1 : Définition
*   Normes.
*   Standards.

### Chapitre 12 : Pharmacovigilance
*   Organisation.
"""

def test_structural_extraction():
    writer = Writer(None, None)
    # Testing the Python parser
    res = writer.parse_markdown_outline(broken_markdown)
    print("--- STRUCTURAL EXTRACTION TEST ---")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    
    if len(res) == 3: # Intro + Chap 1 + Chap 12
        print("\nSUCCESS: Python extracted all 3 chapters perfectly!")
    else:
        print(f"\nFAILED: Found {len(res)} chapters instead of 3.")

if __name__ == "__main__":
    test_structural_extraction()