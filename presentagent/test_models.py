import requests
import os

api_key = "AIzaSyAyhsuo-bPkOFJy8uGDDFS0IG0ZMBRZjZs"
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

print(f"Querying: {url}")
try:
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        with open("models_safe.txt", "w") as f:
            for m in data.get('models', []):
                if 'gemini' in m['name']:
                    f.write(f"{m['name']}\n")
                    print(f" - {m['name']}")
    else:
        print(f"Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"Exception: {e}")
