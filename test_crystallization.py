
import asyncio
import os
import sys
import json
from pathlib import Path

base_dir = "/root/codex/ollama-ensemble-proxy"
sys.path.append(base_dir)

from core.config import DossierConfig
from core.llm import LLMClient

async def run_ttu():
    print("--- DEBUGGING RAW RESPONSE ---")
    config = DossierConfig.from_env()
    config.ollama_base_url = "https://ollama.com"
    config.ollama_api_key = "1594e30e14c346e4bfba0b6d4857d82a.NGTObxs0oirsi0AkwHWYSzmP"
    llm = LLMClient(config)
    
    draft_path = Path("/root/codex/temp_draft.txt")
    draft_text = draft_path.read_text(encoding="utf-8")
    
    sys_p = "Tu es un Ingenieur JSON. NE RESUME PAS."
    user_p = f"Convertis en JSON Master Outline :\n{draft_text}"
    
    raw = await llm.ask("qwen3-coder:480b", sys_p, user_p, stage="planner_crystallize", temperature=0.1)
    print("--- RAW OUTPUT START ---")
    print(raw)
    print("--- RAW OUTPUT END ---")

if __name__ == "__main__":
    asyncio.run(run_ttu())
