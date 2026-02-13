import json
import time
import asyncio
import httpx
import re
from typing import Any
from .config import DossierConfig

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
        ctx_size = min(self.config.context_window, 16384)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": ctx_size,
            },
        }

        # Select timeout based on stage
        timeout_val = self.config.planner_timeout_seconds
        if stage in ["writing", "verification", "ranking"]:
            timeout_val = self.config.writer_timeout_seconds

        last_error = None
        for attempt in range(max_retries):
            try:
                print(f"DEBUG: LLM Call to {model} (ctx: {ctx_size}, stage: {stage}, attempt: {attempt+1})")
                start_ts = time.time()
                async with httpx.AsyncClient(timeout=timeout_val) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    content = data.get("message", {}).get("content", "")
                    duration = round(time.time() - start_ts, 2)
                    print(f"DEBUG: LLM Success in {duration}s")
                    
                    if llm_logs is not None:
                        llm_logs.append({
                            "timestamp": int(time.time()),
                            "stage": stage,
                            "model": model,
                            "duration": round(time.time() - start_ts, 2),
                            "input_len": len(system_prompt) + len(user_prompt),
                            "output_len": len(content),
                        })
                    return content
            except Exception as e:
                last_error = e
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
        # Try pure extraction
        cleaned = raw_text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
            
        # Repair attempt
        repair_prompt = f"""You are a JSON fixer. The following text contains invalid JSON. 
Output ONLY the corrected JSON. No markdown, no comments.
Original Text:
{raw_text[:4000]}
"""
        try:
            repaired = await self.ask(model, "Fix JSON", repair_prompt, llm_logs, f"{stage}_repair", temperature=0.1)
            cleaned = repaired.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            return json.loads(cleaned)
        except Exception:
            raise ValueError(f"Failed to parse JSON in stage {stage}")
