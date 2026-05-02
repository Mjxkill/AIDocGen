import asyncio
import os
import sys
from pathlib import Path

base_dir = "/root/codex/ollama-ensemble-proxy"
sys.path.append(base_dir)

from core.config import DossierConfig
from core.llm import LLMClient

async def test_official_cloud():
    print("--- TESTING OFFICIAL CLOUD NAMES ---")
    config = DossierConfig.from_env()
    config.ollama_base_url = "https://ollama.com"
    config.ollama_api_key = "1594e30e14c346e4bfba0b6d4857d82a.NGTObxs0oirsi0AkwHWYSzmP"
    llm = LLMClient(config)
    
    # TESTING glm-5
    try:
        print("Calling 'glm-5'...")
        res = await llm.ask("glm-5", "Expert scientifique.", "Bonjour.", stage="test")
        print(f"SUCCESS: {res[:50]}...")
    except Exception as e:
        print(f"FAILED 'glm-5': {e}")

if __name__ == "__main__":
    asyncio.run(test_official_cloud())