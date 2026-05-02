
import asyncio
import httpx
import json

models = ["glm-5:cloud", "deepseek-v3.1:cloud", "mistral-large-3:cloud", "qwen3-coder:cloud", "gemini-3-flash-preview:cloud"]
url = "http://127.0.0.1:11434/api/chat"

async def test_config(model, name, options=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Ping. Respond with 'Pong' only."}],
        "stream": False
    }
    if options:
        payload["options"] = options
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload)
            return resp.status_code
    except Exception as e:
        return str(e)

async def run_matrix():
    print(f"{'Model':<30} | {'Minimal':<8} | {'Temp Only':<8} | {'Full Ops':<8}")
    print("-" * 65)
    
    for m in models:
        # 1. Minimal
        s1 = await test_config(m, "Min")
        # 2. Temp only
        s2 = await test_config(m, "Temp", {"temperature": 0.7})
        # 3. Full (Temp + CTX)
        s3 = await test_config(m, "Full", {"temperature": 0.7, "num_ctx": 16384})
        
        print(f"{m:<30} | {s1:<8} | {s2:<8} | {s3:<8}")

if __name__ == "__main__":
    asyncio.run(run_matrix())
