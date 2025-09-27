import os
from openai import OpenAI
from mistralai import Mistral
import anthropic
from google import genai
import requests 
import time

EXCLUDE_DIRS = {'.git', '__pycache__', 'node_modules', 'venv', 'env', '.idea', 'build', 'dist'}
EXCLUDE_FILES = {'README.md', 'LICENSE', '.gitignore'}
EXTENSIONS = (".py", ".js", ".ts", ".java", ".cpp", ".h", ".md", ".html", ".css", ".go", ".rs") 

MAX_TOKENS = 3000  
CHUNK_SIZE = 2000  


hf = os.environ["HF"]
openai_api = os.environ['OPENAI']
mistral = os.environ['MISTRAL']
claude = os.environ['CLAUDE']
grok = os.environ['GROK'] 
gemini = os.environ['GEMINI']
open_router = os.environ['OPEN_ROUTER'] 
BASE_URL = os.environ["BASE_URL"]

if not hf or not BASE_URL or not openai_api or not mistral or not claude or not grok or not gemini or not open_router:
    raise RuntimeError("❌ Missing environment variable. Check GitHub secrets.")

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
        return f"You are a concise code summarizer. Provide key points, structure, and purpose without unnecessary details.\n\n {prompt}"



def readme_gen_message(prompt, provider):
    same_message = ["openai", "mistral", "claude", "openrouter", "grok"]
    if provider in same_message:
        messages=[
            {
                "role": "system", 
                "content": "You are a helpful assistant that generates repository documentation from file summaries. And you will write and reply with nothing but the documentation"
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
        print(f"Error initializing client: {e}")




def api_call(client, provider, prompt, message_formatter):
    formatted_message = message_formatter(prompt, provider)
    match provider:
        case "openai":
            response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=formatted_message,
                        max_tokens=1000,  
                        temperature=0.5
            )  
            return response.choices[0].message.content.strip()
        case "gemini":
            content = formatted_message
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
                messages = formatted_message,
                max_tokens = 1000
            )
            return response.choices[0].message.content 
        case "claude":
            response = client.messages.create(
                model = "claude-sonnet-4-20250514",
                max_tokens = 1000,
                messages = formatted_message
            )

            if isinstance(response.content, list):
                return response.content[0].text
            
            return response.content
        case "openrouter":
            response = client.chat.completions.create(
                model = "deepseek/deepseek-r1:free",
                messages = formatted_message,
                max_tokens = 1000
            )
            return response.choices[0].message.content 
        
        case "grok":
            response = client.chat.completions.create(
                model = "grok-beta",
                messages = formatted_message,
                max_tokens = 1000
            )
            return response.choices[0].message.content 
        case "HF":
            headers = {"Authorization": f"Bearer {hf}"}
            content = formatted_message
            r = requests.get(f"{BASE_URL}/chat", params={"prompt": content}, headers=headers)
            response = r.json()
            return response["response"]
        


client, provider = available_client()

def estimate_tokens(text):
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4 + 1

def collect_file_paths(root_dir):
    """Collect list of relevant file paths without loading content yet."""
    paths = []
    for subdir, _, files in os.walk(root_dir):
        if any(excl in subdir for excl in EXCLUDE_DIRS):
            continue
        for file in files:
            if file.startswith('.') or file in EXCLUDE_FILES or not file.endswith(EXTENSIONS):
                continue
            paths.append(os.path.join(subdir, file))
    return paths



def summarize_file(file_path):
    """Summarize a single file, chunking if too large."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return "Error reading file."

    relative_path = os.path.relpath(file_path, os.getcwd())
    
    # If file is small enough, summarize directly
    if estimate_tokens(content) <= MAX_TOKENS // 2:  
        prompt = f"Summarize this code file '{relative_path}':\n\n{content[:10000]}"  # Limit content
        return summarize_text(prompt)

    # For large files, chunk and summarize
    chunks = [content[i:i + CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
    chunk_summaries = []
    
    print(f"  Processing {len(chunks)} chunks for {relative_path}")
    
    for i, chunk in enumerate(chunks[:5], 1):  # Limit to first 5 chunks
        prompt = f"Summarize chunk {i} of file '{relative_path}':\n\n{chunk}"
        summary = summarize_text(prompt)
        chunk_summaries.append(summary)
        
        # Add small delay to avoid rate limits
        if provider in ["openai", "claude", "mistral"]:
            time.sleep(0.5)

    # Combine chunk summaries
    combined_chunks = "\n".join(chunk_summaries)
    final_prompt = f"Combine these summaries into one cohesive summary for '{relative_path}':\n\n{combined_chunks}"
    return summarize_text(final_prompt)

def summarize_text(prompt):
    """Call LLM to summarize text, with robust error handling."""
    try:
        response = api_call(client, provider, prompt, summarizer_message)
        return response
    except Exception as e:
        print(f"Summarization Error: {e}")



def generate_readme_content(file_summaries):
    """Generate README using summaries of files."""
    # Build the prompt with file summaries
    prompt = """Generate a comprehensive README.md for this repository based on the following file summaries.
    
Include these sections:
- Project Title and Description
- Features
- Installation
- Usage
- File Structure
- Contributing
- License

File Summaries:
"""
    
    # Add summaries, limiting size
    total_tokens = estimate_tokens(prompt)
    for path, summary in file_summaries.items():
        file_section = f"\n### {path}\n{summary}\n"
        section_tokens = estimate_tokens(file_section)
        
        if total_tokens + section_tokens > MAX_TOKENS - 500:  # Leave room for response
            prompt += "\n[Additional files omitted due to length]\n"
            break
        
        prompt += file_section
        total_tokens += section_tokens
    
    # Generate README
    readme_content = api_call(client, provider, prompt, readme_gen_message)
    
    if not readme_content or "Error" in readme_content:
        # Fallback README if generation fails
        readme_content = f"""# Repository Documentation

## Description
This repository contains {len(file_summaries)} files.

## Files
"""
        for path in list(file_summaries.keys())[:20]:  # List first 20 files
            readme_content += f"- `{path}`\n"
        
        readme_content += "\n## Installation\nPlease refer to individual file documentation.\n"
    
    return readme_content


def main():
    """Main function to orchestrate README generation."""
    print(f"Starting README generation using {provider}...")
    
    repo_dir = os.getcwd()
    readme_path = os.path.join(repo_dir, "README.md")
    
    # Collect files
    file_paths = collect_file_paths(repo_dir)
    print(f"Found {len(file_paths)} files to summarize")
    
    if not file_paths:
        print("No files found to summarize!")
        return
    
    # Summarize files
    file_summaries = {}
    for i, path in enumerate(file_paths, 1):
        relative_path = os.path.relpath(path, repo_dir)
        print(f"[{i}/{len(file_paths)}] Summarizing {relative_path}...")
        
        summary = summarize_file(path)
        file_summaries[relative_path] = summary
        
        # Add delay to respect rate limits
        if provider in ["openai", "claude", "mistral"] and i % 5 == 0:
            print("  Pausing to respect rate limits...")
            time.sleep(2)
    
    # Generate README
    print("\nGenerating README content...")
    new_content = generate_readme_content(file_summaries)
    
    # Save or update README
    if not os.path.exists(readme_path):
        print("Creating new README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    else:
        with open(readme_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()
        
        if existing_content != new_content:
            print("Updating existing README.md")
            # Backup existing README
            backup_path = readme_path + ".backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(existing_content)
            print(f"  Backed up existing README to {backup_path}")
            
            # Write new content
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        else:
            print("README.md is already up-to-date")
    
    print("✅ README generation complete!")


if __name__ == "__main__":
    main()