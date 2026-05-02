import json
import time
import asyncio
import httpx
import re
from typing import Any
from .config import DossierConfig
from .logging_config import get_logger

log = get_logger("aidocgen.llm")

class LLMClient:
    def __init__(self, config: DossierConfig):
        self.config = config

    async def ask(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        llm_logs: list[dict[str, Any]] | None = None,
        stage: str = "unknown",
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> str:
        url = f"{self.config.ollama_base_url.rstrip('/')}/api/chat"
        
        headers = {}
        if self.config.ollama_api_key:
            headers["Authorization"] = f"Bearer {self.config.ollama_api_key}"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        
        # --- BLINDAGE CLOUD : AUCUNE OPTION SI OLLAMA.COM ---
        # Le relais cloud ollama.com renvoie 404 si on envoie 'options' (temperature, num_ctx, etc)
        if "ollama.com" not in url and ":cloud" not in model:
            payload["options"] = {
                "temperature": temperature,
                "num_ctx": min(self.config.context_window, 32000),
            }

        timeout_val = self.config.planner_timeout_seconds
        if stage in ["writing", "verification", "ranking", "critique", "rewrite", "executive_summary"]:
            timeout_val = self.config.writer_timeout_seconds

        last_error = None
        for attempt in range(max_retries):
            try:
                log.info("llm call", extra={"model": model, "stage": stage,
                                            "attempt": attempt + 1, "max": max_retries})
                start_ts = time.time()
                async with httpx.AsyncClient(timeout=timeout_val) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    content = data.get("message", {}).get("content", "")
                    duration = round(time.time() - start_ts, 2)
                    log.info("llm ok", extra={"model": model, "stage": stage,
                                              "duration_s": duration,
                                              "output_chars": len(content)})

                    if llm_logs is not None:
                        llm_logs.append({
                            "timestamp": int(time.time()),
                            "stage": stage,
                            "model": model,
                            "duration": duration,
                            "input_len": len(system_prompt) + len(user_prompt),
                            "output_len": len(content),
                        })
                    return content
            except Exception as e:
                last_error = e
                log.warning("llm attempt failed",
                            extra={"model": model, "stage": stage,
                                   "attempt": attempt + 1, "error": str(e)})
                await asyncio.sleep(2 * (attempt + 1))

        raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")

    async def parse_json(
        self,
        raw_text: str,
        model: str,
        stage: str,
        llm_logs: list[dict[str, Any]] | None = None,
        schema_hint: Any = None,
    ) -> Any:
        cleaned = raw_text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
            
        repair_prompt = f"You are a JSON fixer. Output ONLY corrected JSON. Original: {raw_text[:4000]}"
        try:
            repaired = await self.ask(model, "Fix JSON", repair_prompt, llm_logs, f"{stage}_repair", temperature=0.1)
            cleaned = repaired.strip()
            if "```json" in cleaned: cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            return json.loads(cleaned)
        except Exception:
            raise ValueError(f"Failed to parse JSON in stage {stage}")
