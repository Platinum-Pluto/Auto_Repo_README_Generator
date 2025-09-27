import os 
from openai import OpenAI
from mistralai import Mistral
import anthropic
from google import genai
import requests 



hf = os.environ["HF"]
openai_api = os.environ["OPENAI"]
mistral = os.environ["MISTRAL"]
claude = os.environ["CLAUDE"]
grok = os.environ["GROK"] 
gemini = os.environ["GEMINI"]
open_router = os.environ["OPEN_ROUTER"] 
BASE_URL = os.environ["BASE_URL"]


def summarizer_message(prompt, provider):
    same_message = ["openai", "mistral", "claude", "openrouter", "grok"]
    if provider in same_message:
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
    else:
        return f"You are a concise code summarizer. Provide key points, structure, and purpose without unnecessary details.\n {prompt}"



def readme_gen_message(prompt, provider):
    same_message = ["openai", "mistral", "claude", "openrouter", "grok"]
    if provider in same_message:
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
    else:
        return "You are a helpful assistant that generates repository documentation from file summaries."

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
                        messages=messages(prompt, provider),
                        max_tokens=1000,  
                        temperature=0.5
            )  
            return response.choices[0].message.content.strip()
        case "gemini":
            content = messages(prompt, provider)
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
                messages = messages(prompt, provider)
            )
            return response.choices[0].message.content 
        case "claude":
            response = client.messages.create(
                model = "claude-sonnet-4-20250514",
                max_tokens = 1000,
                messages = messages(prompt, provider)
            )
            return response.content
        case "openrouter":
            response = client.chat.completions.create(
                model = "deepseek/deepseek-r1:free",
                messages = messages(prompt, provider)
            )
            return response.choices[0].message.content 
        
        case "grok":
            response = client.chat.completions.create(
                model = "grok-beta",
                messages = messages(prompt, provider)
            )
            return response.choices[0].message.content 
        case "HF":
            headers = {"Authorization": f"Bearer {hf}"}
            content = messages(prompt, provider)
            r = requests.get(f"{BASE_URL}/chat", params={"prompt": content}, headers=headers)
            response = r.json()
            return response["response"]
        