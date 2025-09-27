import os
import json
import random
import time
import requests
from openai import OpenAI
from mistralai import Mistral
import anthropic
from google import genai
from pathlib import Path

EXCLUDE_DIRS = {'.git', '__pycache__', 'node_modules', 'venv', 'env', '.idea', 'build', 'dist'}
EXCLUDE_FILES = {'README.md', 'LICENSE', '.gitignore'}
EXTENSIONS = (".py", ".js", ".ts", ".java", ".cpp", ".h", ".md", ".html", ".css", ".go", ".rs")

# Token/size heuristics (tweak if needed)
MAX_TOKENS = 3000
CHUNK_SIZE = 2000
SMALL_FILE_THRESHOLD_CHARS = 2000         # files smaller than this are "small"
BATCH_CHAR_LIMIT = 8000                   # group small files up to this many chars into one request
MAX_CHUNKS_PER_FILE = 5                   # limit chunk summarization per-file
CACHE_DIR = Path(".cache")
CACHE_FILE = CACHE_DIR / "summaries.json"

# Environment keys
hf = os.environ.get("HF")
openai_api = os.environ.get("OPENAI")
mistral = os.environ.get("MISTRAL")
claude = os.environ.get("CLAUDE")
grok = os.environ.get("GROK")
gemini = os.environ.get("GEMINI")
open_router = os.environ.get("OPEN_ROUTER") or os.environ.get("OPEN_ROUTER_KEY")
BASE_URL = os.environ.get("BASE_URL")


def load_cache():
    CACHE_DIR.mkdir(exist_ok=True)
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_cache(cache):
    CACHE_DIR.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def cache_key(path, mtime):
    return f"{path}::{int(mtime)}"


def summarizer_message(prompt, provider):
    same_message = ["openai", "mistral", "claude", "openrouter", "grok"]
    if provider in same_message:
        messages = [
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
        # for retro-compat / other providers that expect text input
        return f"You are a concise code summarizer. Provide key points, structure, and purpose without unnecessary details.\n\n{prompt}"

def readme_gen_message(prompt, provider):
    same_message = ["openai", "mistral", "claude", "openrouter", "grok"]
    if provider in same_message:
        messages = [
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


def available_clients_priority_list():
    """
    Return a list of (client, provider_name) tuples in preferred fallback order.
    Order recommended: OpenAI -> Mistral -> Claude -> Gemini -> OpenRouter -> Grok -> HF
    Only includes providers for which an env key exists.
    """
    clients = []
    try:
        if openai_api:
            clients.append((OpenAI(api_key=openai_api), "openai"))
    except Exception as e:
        print("OpenAI init error:", e)

    try:
        if mistral:
            clients.append((Mistral(api_key=mistral), "mistral"))
    except Exception as e:
        print("Mistral init error:", e)

    try:
        if claude:
            clients.append((anthropic.Anthropic(api_key=claude), "claude"))
    except Exception as e:
        print("Claude init error:", e)

    try:
        if gemini:
            clients.append((genai.Client(api_key=gemini), "gemini"))
    except Exception as e:
        print("Gemini init error:", e)

    try:
        if open_router:
            clients.append((OpenAI(base_url="https://openrouter.ai/api/v1", api_key=open_router), "openrouter"))
    except Exception as e:
        print("OpenRouter init error:", e)

    try:
        if grok:
            clients.append((OpenAI(api_key=grok, base_url="https://api.x.ai/v1"), "grok"))
    except Exception as e:
        print("Grok init error:", e)

    if hf:
        clients.append((None, "HF"))

    return clients

clients_providers = available_clients_priority_list()
primary_provider = clients_providers[0][1] if clients_providers else None


def api_call(clients_list, prompt, message_formatter, max_retries=4, timeout_seconds=60):
    """
    Try providers in order. For each provider attempt `max_retries` times with exponential backoff
    on rate-limit (429) or transient errors. Move to next provider if provider exhausted.
    Returns model text output string on success, or raises last exception on failure.
    """
    last_exc = None

    for client, provider in clients_list:
        formatted_message = message_formatter(prompt, provider)
        for attempt in range(max_retries):
            try:
                # Provider-specific call logic
                if provider == "openai":
                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=formatted_message,
                        max_tokens=1000,
                        temperature=0.5,
                        timeout=timeout_seconds
                    )
                    return resp.choices[0].message.content.strip()

                elif provider == "gemini":
                    content = formatted_message if isinstance(formatted_message, str) else formatted_message[-1]["content"]
                    contents = [{"type": "input_text", "text": content}]
                    resp = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=contents
                    )
                    return getattr(resp, "text", str(resp))

                elif provider == "mistral":
                    resp = client.chat.complete(
                        model="mistral-large-latest",
                        messages=formatted_message,
                        max_tokens=1000
                    )
                    return resp.choices[0].message.content

                elif provider == "claude":
                    resp = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1000,
                        messages=formatted_message
                    )
                    if isinstance(resp.content, list):
                        return resp.content[0].text
                    return resp.content

                elif provider == "openrouter":
                    resp = client.chat.completions.create(
                        model="deepseek/deepseek-r1:free",
                        messages=formatted_message,
                        max_tokens=1000
                    )
                    return resp.choices[0].message.content

                elif provider == "grok":
                    resp = client.chat.completions.create(
                        model="grok-beta",
                        messages=formatted_message,
                        max_tokens=1000
                    )
                    return resp.choices[0].message.content

                elif provider == "HF":
                    headers = {"Authorization": f"Bearer {hf}"} if hf else {}
                    content = formatted_message if isinstance(formatted_message, str) else formatted_message[-1]["content"]
                    r = requests.get(f"{BASE_URL}/chat", params={"prompt": content}, headers=headers, timeout=timeout_seconds)
                    r.raise_for_status()
                    response = r.json()
                    return response.get("response", str(response))

                else:
                    raise RuntimeError(f"Unknown provider {provider}")

            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                # detect rate-limit or 429 - common substrings
                is_rate = ("429" in msg) or ("rate limit" in msg) or ("rate-limited" in msg) or ("too many requests" in msg)
                is_transient = ("timeout" in msg) or ("tempor" in msg) or ("503" in msg) or ("502" in msg)

                if (is_rate or is_transient) and attempt < max_retries - 1:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    print(f"[{provider}] transient/rate error: {e}. retrying attempt {attempt+1}/{max_retries} after {wait:.1f}s")
                    time.sleep(wait)
                    continue
                else:
                    print(f"[{provider}] error: {e}. moving to next provider (attempt {attempt+1}/{max_retries})")
                    break  # move to next provider

    # If we get here, all providers exhausted
    raise last_exc if last_exc else RuntimeError("No providers available")


def estimate_tokens(text):
    return len(text) // 4 + 1

def collect_file_paths(root_dir):
    paths = []
    for subdir, _, files in os.walk(root_dir):
        if any(excl in subdir for excl in EXCLUDE_DIRS):
            continue
        for file in files:
            if file.startswith('.') or file in EXCLUDE_FILES or not file.endswith(EXTENSIONS):
                continue
            paths.append(os.path.join(subdir, file))
    return paths


def summarize_text(prompt):
    """Wrapper to call the LLMs via api_call with fallback list."""
    try:
        return api_call(clients_providers, prompt, summarizer_message)
    except Exception as e:
        print("Summarization Error:", e)
        return f"Error summarizing: {e}"

def summarize_file(file_path, cache):
    """
    Summarize a single (possibly large) file. Use cache when available.
    For very large files, chunk and summarize up to MAX_CHUNKS_PER_FILE chunks.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return "Error reading file."

    mtime = os.path.getmtime(file_path)
    key = cache_key(file_path, mtime)
    if key in cache:
        return cache[key]

    # If small enough, summarize directly (limit content length to avoid passing huge files)
    if len(content) <= SMALL_FILE_THRESHOLD_CHARS or estimate_tokens(content) <= MAX_TOKENS // 2:
        prompt = f"Summarize this code file '{os.path.relpath(file_path, os.getcwd())}':\n\n{content[:10000]}"
        summary = summarize_text(prompt)
        cache[key] = summary
        save_cache(cache)
        return summary

    # Large file: chunk it
    chunks = [content[i:i + CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
    chunk_summaries = []
    print(f"  Processing {min(len(chunks), MAX_CHUNKS_PER_FILE)} chunks for {file_path}")
    for i, chunk in enumerate(chunks[:MAX_CHUNKS_PER_FILE], 1):
        prompt = f"Summarize chunk {i} of file '{os.path.relpath(file_path, os.getcwd())}':\n\n{chunk}"
        summary = summarize_text(prompt)
        chunk_summaries.append(summary)
        # small backoff to be polite
        time.sleep(0.4)

    combined_chunks = "\n".join(chunk_summaries)
    final_prompt = f"Combine these summaries into one cohesive summary for '{os.path.relpath(file_path, os.getcwd())}':\n\n{combined_chunks}"
    final_summary = summarize_text(final_prompt)
    cache[key] = final_summary
    save_cache(cache)
    return final_summary

def batch_summarize_small_files(small_files, cache):
    """
    Combine many small files into a single LLM request per batch.
    The model is instructed to return a JSON mapping filename->summary to allow splitting.
    """
    results = {}
    i = 0
    n = len(small_files)

    while i < n:
        batch = []
        batch_chars = 0
        while i < n:
            path = small_files[i]
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                results[path] = "Error reading file."
                i += 1
                continue

            if batch_chars + len(content) > BATCH_CHAR_LIMIT and batch:
                break
            batch.append((path, content))
            batch_chars += len(content)
            i += 1

        if not batch:
            # file too large for batch limit — fallback single summarize
            path, content = small_files[i], ""
            try:
                content = open(path, 'r', encoding='utf-8', errors='ignore').read()
            except Exception:
                results[path] = "Error reading file."
                i += 1
                continue
            results[path] = summarize_file(path, cache)
            i += 1
            continue

        # Build prompt asking for JSON mapping
        prompt_parts = []
        for path, content in batch:
            rel = os.path.relpath(path, os.getcwd())
            prompt_parts.append(f"---FILE_START: {rel}\n{content}\n---FILE_END")

        combined_prompt = (
            "You are given multiple small source files. For each file return a JSON object mapping "
            "the file path to a short summary (2-4 sentences). The entire response must be valid JSON. "
            "Example: {\"path/to/file.py\": \"summary...\", \"other.py\": \"summary...\"}\n\n"
            "FILES:\n" + "\n".join(prompt_parts)
        )

        resp_text = summarize_text(combined_prompt)
        # Attempt to parse JSON from model
        parsed = None
        try:
            # if the model returned text with code fences or extra text, try to extract JSON substring
            text = resp_text.strip()
            # find first { and last }
            first_brace = text.find("{")
            last_brace = text.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_candidate = text[first_brace:last_brace+1]
                parsed = json.loads(json_candidate)
        except Exception as e:
            parsed = None

        if parsed:
            for path, _ in batch:
                mtime = os.path.getmtime(path)
                key = cache_key(path, mtime)
                summary = parsed.get(os.path.relpath(path, os.getcwd()), parsed.get(path))
                if summary is None:
                    # as fallback, create a small placeholder
                    summary = "Summary not found in combined response."
                cache[key] = summary
                results[path] = summary
            save_cache(cache)
        else:
            # Fallback: if JSON couldn't be parsed, fall back to per-file summarization
            print("Failed to parse JSON from batch response — falling back to per-file calls for this batch.")
            for path, _ in batch:
                results[path] = summarize_file(path, cache)

    return results


def generate_readme_content(file_summaries):
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
    total_tokens = estimate_tokens(prompt)
    for path, summary in file_summaries.items():
        file_section = f"\n### {path}\n{summary}\n"
        section_tokens = estimate_tokens(file_section)
        if total_tokens + section_tokens > MAX_TOKENS - 500:
            prompt += "\n[Additional files omitted due to length]\n"
            break
        prompt += file_section
        total_tokens += section_tokens

    try:
        readme_content = api_call(clients_providers, prompt, readme_gen_message)
    except Exception as e:
        print("README generation failed:", e)
        # Fallback brief README
        readme_content = f"""# Repository Documentation

## Description
This repository contains {len(file_summaries)} summarized files.

## Files
"""
        for path in list(file_summaries.keys())[:50]:
            readme_content += f"- `{path}`\n"
        readme_content += "\n## Installation\nPlease refer to individual file documentation.\n"

    return readme_content


def main():
    if not clients_providers:
        print("No provider API keys found. Set OPENAI, MISTRAL, CLAUDE, OPEN_ROUTER, GROK or HF credentials.")
        return

    print(f"Starting README generation using {clients_providers[0][1]} (primary) with fallback providers {[p for _, p in clients_providers]}")

    repo_dir = os.getcwd()
    readme_path = os.path.join(repo_dir, "README.md")

    # collect file list
    file_paths = collect_file_paths(repo_dir)
    print(f"Found {len(file_paths)} files to summarize")

    if not file_paths:
        print("No files found to summarize!")
        return

    # load cache
    cache = load_cache()

    # split small vs large
    small_files = []
    large_files = []
    for p in file_paths:
        try:
            size = os.path.getsize(p)
        except Exception:
            size = 0
        if size <= SMALL_FILE_THRESHOLD_CHARS:
            small_files.append(p)
        else:
            large_files.append(p)

    file_summaries = {}

    # Summarize large files individually (may chunk)
    for idx, path in enumerate(large_files, 1):
        rel = os.path.relpath(path, repo_dir)
        print(f"[L {idx}/{len(large_files)}] Summarizing large file {rel}...")
        file_summaries[rel] = summarize_file(path, cache)
        time.sleep(0.4)

    # Summarize small files in batches
    if small_files:
        print(f"Batch summarizing {len(small_files)} small files...")
        batch_results = batch_summarize_small_files(small_files, cache)
        for path, summary in batch_results.items():
            file_summaries[os.path.relpath(path, repo_dir)] = summary

    # If no files summarized for some reason, return fallback
    if not file_summaries:
        print("No summaries produced — aborting README generation.")
        return

    # Generate README content
    print("\nGenerating README content...")
    new_content = generate_readme_content(file_summaries)

    # Write or update README
    if not os.path.exists(readme_path):
        print("Creating new README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    else:
        with open(readme_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()
        if existing_content != new_content:
            print("Updating existing README.md")
            backup_path = readme_path + ".backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(existing_content)
            print(f"  Backed up existing README to {backup_path}")
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        else:
            print("README.md is already up-to-date")

    print("✅ README generation complete!")

if __name__ == "__main__":
    main()
