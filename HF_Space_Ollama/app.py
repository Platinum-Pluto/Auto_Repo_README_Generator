from fastapi import FastAPI
import requests, subprocess, json, re

app = FastAPI()

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:1.5b"

def ensure_model():
    """Check if the model is available; if not, pull it."""
    try:
        resp = requests.get("http://localhost:11434/api/tags")
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if any(m["name"] == MODEL_NAME for m in models):
                return
        subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
    except Exception as e:
        print("Error ensuring model:", e)

def clean_response(text: str) -> str:
    """Remove <think>...</think> sections and extra whitespace."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()
    
ensure_model()

@app.get("/")
def root():
    return {"message": "Ollama + FastAPI running in HuggingFace Space"}

@app.get("/chat")
def chat(prompt: str):
    payload = {"model": MODEL_NAME, "prompt": prompt}
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
        output = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        output += data["response"]
                except Exception:
                    pass
        return {"response": clean_response(output)}
    except Exception as e:
        return {"error": str(e)}