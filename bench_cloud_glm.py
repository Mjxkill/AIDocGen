import asyncio
import httpx
import json

API_KEY = "1594e30e14c346e4bfba0b6d4857d82a.NGTObxs0oirsi0AkwHWYSzmP"
BASE_URL = "https://ollama.com/api/chat"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

PROMPT_SUJET = "Qualité des vaccins en france comparé au autres pays. Avantage et risque des vacins"

async def call_cloud(model, system, user, temp=0.7):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "stream": False
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(BASE_URL, json=payload, headers=HEADERS)
        return resp.json()["message"]["content"]

async def run_bench():
    print("--- 1. BROUILLON (GLM-5) ---")
    sys_p = "Tu es un Redacteur en Chef. Fais un sommaire exhaustif et tres long."
    user_p = f"Sujet : {PROMPT_SUJET}"
    draft = await call_cloud("glm-5", sys_p, user_p, temp=0.7)
    print(f"LONGUEUR : {len(draft)}")

    print("\n--- 2. JSON (GLM-5) ---")
    sys_c = "Tu es un expert JSON. Ne resume rien. Garde tout."
    user_c = f"Convertis en JSON (master_outline) :\n{draft}"
    res = await call_cloud("glm-5", sys_c, user_c, temp=0.1)
    
    try:
        clean = res.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        chapters = data.get("master_outline", [])
        print(f"RESULTAT : {len(chapters)} chapitres.")
        for i, c in enumerate(chapters):
            print(f"Ch {i+1}: {c.get('chapter_title')} ({len(c.get('sub_sections', []))} secs)")
    except Exception as e:
        print(f"ERR : {e}")

if __name__ == "__main__":
    asyncio.run(run_bench())