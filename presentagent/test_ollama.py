import os
import requests
import json

base_url = "http://localhost:11434/api/tags"

print(f"Testing Ollama connection at: {base_url}")

try:
    response = requests.get(base_url)
    if response.status_code == 200:
        models = [m['name'] for m in response.json()['models']]
        print("Success! Ollama is running.")
        print(f"Available models: {models}")
        
        required = ['llama3.2', 'llava', 'nomic-embed-text']
        missing = [r for r in required if not any(r in m for m in models)]
        
        if missing:
            print(f"WARNING: Missing required models: {missing}")
            print("Please run: ollama pull <model_name>")
        else:
            print("All required models are present! Config looks good.")
            
    else:
        print(f"Error: Ollama returned status {response.status_code}")
except Exception as e:
    print(f"Connection Failed: {e}")
    print("Is Ollama running? (Start 'ollama serve' in a terminal)")
