from dotenv import load_dotenv
import os 
from openai import OpenAI
from mistralai import Mistral
import anthropic
from google import genai
import requests 

load_dotenv()
hf = os.getenv('HF')
openai_api = os.getenv('OPENAI')
mistral = os.getenv('MISTRAL')
claude = os.getenv('CLAUDE')
grok = os.getenv('GROK')
gemini = os.getenv('GEMINI')
open_router = os.getenv('OPEN_ROUTER')
BASE_URL = os.getenv('BASE_URL')

def summarizer_message(prompt):
    messages=[
        {
            "role": "system", 
            "content": "You are a concise code summarizer. Provide key points, structure, and purpose without unnecessary details."
        },
        {
            "role": "user", 
            "content": prompt
        }
    ]
    return messages 

def readme_gen_message(prompt):
    messages=[
        {
            "role": "system", 
            "content": "You are a helpful assistant that generates repository documentation from file summaries."
        },
        {
            "role": "user", 
            "content": prompt
        }
    ]
    return messages 

def available_client():
    try:
        if openai_api:
            client = OpenAI(api_key=openai_api)
            return client, "openai"
        elif gemini:
            client = genai.Client(api_key=gemini)
            return client, "gemini"
        elif mistral:
            client = Mistral(api_key=mistral)
            return client, "mistral"
        elif claude:
            client = anthropic.Anthropic(api_key = claude)
            return client, "claude"
        elif open_router:
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=open_router)
            return client, "openrouter"
        elif grok:
            client = OpenAI(api_key=grok, base_url="https://api.x.ai/v1",)
            return client, "grok"
        elif hf:
            client = ""
            return client, "HF"
    except Exception as e:
        print(e)




def api_call(client, provider, prompt, messages):
    match provider:
        case "openai":
            response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages(prompt),
                        max_tokens=1000,  
                        temperature=0.5
            )  
            return response.choices[0].message.content.strip()
        case "gemini":
            content = f"You are a concise code summarizer. Provide key points, structure, and purpose without unnecessary details.\n {prompt}"
            contents = [
                {"type": "input_text", "text": content}
            ]
            response = client.models.generate_content(
                model = "gemini-2.5-flash",
                contents = contents
            )
            return response.text
        case "mistral":
            response = client.chat.complete(
                model = "mistral-large-latest",
                messages = messages(prompt)
            )
            return response.choices[0].message.content 
        case "claude":
            response = client.messages.create(
                model = "claude-sonnet-4-20250514",
                max_tokens = 1000,
                messages = messages(prompt)
            )
            return response.content
        case "openrouter":
            response = client.chat.completions.create(
                model = "deepseek/deepseek-r1:free",
                messages = messages(prompt)
            )
            return response.choices[0].message.content 
        
        case "grok":
            response = client.chat.completions.create(
                model = "grok-beta",
                messages = messages(prompt)
            )
            return response.choices[0].message.content 
        case "HF":
            headers = {"Authorization": f"Bearer {hf}"}
            content = f"You are a concise code summarizer. Provide key points, structure, and purpose without unnecessary details.\n {prompt}"
            r = requests.get(f"{BASE_URL}/chat", params={"prompt": content}, headers=headers)
            response = r.json()
            return response["response"]
        