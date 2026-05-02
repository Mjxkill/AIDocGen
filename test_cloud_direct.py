
import asyncio
import httpx

async def test_cloud():
    model = "glm-5:cloud"
    print(f"--- TESTING CLOUD ACCESS: {model} ---")
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Es-tu opérationnel sur Ollama Cloud Max ? Réponds brièvement en français."}],
        "stream": False
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                print("SUCCESS!")
                print(resp.json()["message"]["content"])
            else:
                print(f"FAILED: {resp.status_code}")
                print(resp.text)
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_cloud())
