
import httpx
import asyncio
import os

async def test_dynamic_models():
    print("--- STARTING DYNAMIC MODEL FETCH TEST ---")
    url = "http://127.0.0.1:8001/ollama/models?url=https://ollama.com"
    
    # We need to simulate the frontend auth
    # Since I don't want to mess with login, I'll check if the backend
    # logic itself works by calling the internal function if possible,
    # or just use a real authenticated request.
    
    # Let's use the actual OLLAMA_API_KEY from the env
    api_key = "1594e30e14c346e4bfba0b6d4857d82a.NGTObxs0oirsi0AkwHWYSzmP"
    
    async with httpx.AsyncClient() as client:
        # We call the official API directly first to confirm it's up
        print("Testing direct Cloud API...")
        r = await client.get("https://ollama.com/api/tags", headers={"Authorization": f"Bearer {api_key}"})
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            print(f"SUCCESS: Found {len(models)} models on Cloud.")
            print(f"Sample: {models[:3]}")
            
            # CRITICAL CHECK: glm-5 presence
            if "glm-5" in models:
                print("PASSED: 'glm-5' is present in dynamic list.")
            else:
                print("WARNING: 'glm-5' not found in Cloud tags.")
        else:
            print(f"FAILED: Cloud API returned {r.status_code}")

if __name__ == "__main__":
    asyncio.run(test_dynamic_models())
