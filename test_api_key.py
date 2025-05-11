import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("HUGGINGFACE_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

def test_huggingface_api_key():
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {"inputs": "Explain Section 302 of IPC."}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        print("✅ API Key Valid! Response:", response.json())
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    test_huggingface_api_key()
