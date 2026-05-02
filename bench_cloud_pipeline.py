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
        "stream": False,
        "options": {"temperature": temp}
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(BASE_URL, json=payload, headers=HEADERS)
        return resp.json()["message"]["content"]

async def run_bench():
    print("--- 1. GENERATION DU BROUILLON (GLM-5) ---")
    sys_planner = "Tu es un Rédacteur en Chef. Fais un sommaire de livre exhaustif, sans limite de taille, très détaillé."
    user_planner = f"Sujet : {PROMPT_SUJET}. Utilise une structure organique et asymétrique."
    
    draft = await call_cloud("glm-5", sys_planner, user_planner, temp=0.7)
    print(f"LONGUEUR BROUILLON : {len(draft)} caracteres")

    print("\n--- 2. CRISTALLISATION JSON (QWEN3-CODER:480B) ---")
    sys_coder = "Tu es un expert JSON. Transforme le texte en JSON STRICT. NE RESUME PAS. Garde TOUTES les sous-sections."
    user_coder = f"Convertis ce texte en JSON (schéma: master_outline[chapter_title, sub_sections[title, brief]]):\n\n{draft}"
    
    result_json = await call_cloud("qwen3-coder:480b", sys_coder, user_coder, temp=0.1)
    
    try:
        # Nettoyage manuel du JSON si markdown
        cleaned = result_json.strip()
        if "```json" in cleaned: cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned: cleaned = cleaned.split("```")[1].split("```")[0]
        
        parsed = json.loads(cleaned)
        nb_chapitres = len(parsed.get("master_outline", []))
        print(f"RESULTAT : {nb_chapitres} chapitres trouves dans le JSON.")
        
        for i, chap in enumerate(parsed.get("master_outline", [])[:5]):
            print(f"Chapitre {i+1}: {chap.get('chapter_title')} -> {len(chap.get('sub_sections', []))} sections")
            
    except Exception as e:
        print(f"Erreur de parsing JSON : {e}")
        print(result_json[:500])

if __name__ == "__main__":
    asyncio.run(run_bench())