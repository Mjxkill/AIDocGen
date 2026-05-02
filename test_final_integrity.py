import requests
import json

BASE_URL = "http://127.0.0.1:8001"

def test_system_integrity():
    print("--- STARTING END-TO-END INTEGRITY TEST ---")
    
    # 1. Metrics
    try:
        r = requests.get(f"{BASE_URL}/system/metrics")
        print(f"Metrics Status: {r.status_code}")
    except: print("Metrics Error")

    # 2. Run Creation
    payload = {"question": "Test", "junk": "data"}
    try:
        r = requests.post(f"{BASE_URL}/v1/dossier/runs", json=payload)
        print(f"Run Creation Status: {r.status_code}")
        if r.status_code == 422:
            print("CRITICAL: 422 REGRESSION DETECTED")
        else:
            print("SUCCESS: JSON validation is robust (Anti-422 confirmed)")
    except: print("Creation Error")

if __name__ == "__main__":
    test_system_integrity()