import os
import time
import random
from llm import available_client, summarizer_message, readme_gen_message, api_call

EXCLUDE_DIRS = {'.git', '__pycache__', 'node_modules', 'venv', 'env', '.idea', 'build', 'dist'}
EXCLUDE_FILES = {'README.md', 'LICENSE', '.gitignore'}
EXTENSIONS = (".py", ".js", ".ts", ".java", ".cpp", ".h", ".md", ".html", ".css", ".go", ".rs") 


MAX_TOKENS = 8000  
CHUNK_SIZE = 4000  
BATCH_TOKEN_LIMIT = 6000  
LARGE_FILE_THRESHOLD = 3000  


RATE_LIMITS = {
    "openai": {"delay": 3, "max_retries": 5, "backoff_factor": 2},
    "claude": {"delay": 2, "max_retries": 5, "backoff_factor": 1.5},
    "mistral": {"delay": 2, "max_retries": 3, "backoff_factor": 2},
    "default": {"delay": 1, "max_retries": 3, "backoff_factor": 1.5}
}

client, provider = available_client()

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

def summarize_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return "Error reading file."

    relative_path = os.path.relpath(file_path, os.getcwd())
    
    if estimate_tokens(content) <= MAX_TOKENS // 2:  
        prompt = f"Summarize this code file '{relative_path}':\n\n{content[:10000]}"  # Limit content
        return summarize_text(prompt)

    chunks = [content[i:i + CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
    chunk_summaries = []
    
    print(f"  Processing {len(chunks)} chunks for {relative_path}")
    
    for i, chunk in enumerate(chunks[:5], 1): 
        prompt = f"Summarize chunk {i} of file '{relative_path}':\n\n{chunk}"
        summary = summarize_text(prompt)
        chunk_summaries.append(summary)

    combined_chunks = "\n".join(chunk_summaries)
    final_prompt = f"Combine these summaries into one cohesive summary for '{relative_path}':\n\n{combined_chunks}"
    return summarize_text(final_prompt)

def rate_limited_api_call(prompt, message_func):
    """Make API call with rate limiting and retry logic."""
    rate_config = RATE_LIMITS.get(provider, RATE_LIMITS["default"])
    max_retries = rate_config["max_retries"]
    base_delay = rate_config["delay"]
    backoff_factor = rate_config["backoff_factor"]
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = base_delay * (backoff_factor ** attempt) + random.uniform(0.5, 1.5)
                print(f"  Retry {attempt}/{max_retries} - waiting {delay:.1f}s...")
                time.sleep(delay)
            else:
                time.sleep(base_delay)
            
            response = api_call(client, provider, prompt, message_func)
            return response
            
        except Exception as e:
            error_msg = str(e).lower()

            if any(term in error_msg for term in ['rate limit', 'too many requests', '429', 'quota']):
                if attempt < max_retries - 1:
                    wait_time = base_delay * (backoff_factor ** (attempt + 1)) + random.uniform(2, 5)
                    print(f"  Rate limit hit. Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  Rate limit exceeded after {max_retries} attempts: {e}")
                    return "Rate limit exceeded - please try again later."

            elif any(term in error_msg for term in ['timeout', 'connection', 'network', 'server error', '500', '502', '503']):
                if attempt < max_retries - 1:
                    wait_time = base_delay + random.uniform(1, 3)
                    print(f"  Network error. Retrying in {wait_time:.1f}s... ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  Network error after {max_retries} attempts: {e}")
                    return "Network error - please check connection."
            else:
                print(f"  API Error: {e}")
                return f"Error during API call: {e}"
    
    return "Max retries exceeded."

def summarize_text(prompt):
    return rate_limited_api_call(prompt, summarizer_message)

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

    readme_content = rate_limited_api_call(prompt, readme_gen_message)
    
    if not readme_content or "Error" in str(readme_content):
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
    print(f"Starting README generation using {provider}...")
    print(f"Rate limiting: {RATE_LIMITS.get(provider, RATE_LIMITS['default'])}")
    
    repo_dir = os.getcwd()
    readme_path = os.path.join(repo_dir, "README.md")
    file_paths = collect_file_paths(repo_dir)
    print(f"Found {len(file_paths)} files to summarize")
    
    if not file_paths:
        print("No files found to summarize!")
        return
    
    file_summaries = {}
    small_files_batch = []
    batch_content = ""
    batch_tokens = 0
    api_call_count = 0
    
    for i, path in enumerate(file_paths, 1):
        relative_path = os.path.relpath(path, repo_dir)
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {relative_path}: {e}")
            file_summaries[relative_path] = "Error reading file."
            continue
        
        file_tokens = estimate_tokens(content)
        if file_tokens > LARGE_FILE_THRESHOLD:
            if small_files_batch:
                print(f"Processing batch of {len(small_files_batch)} small files...")
                batch_prompt = f"Summarize these code files:\n\n{batch_content}"
                batch_summary = summarize_text(batch_prompt)
                api_call_count += 1
                for batch_file in small_files_batch:
                    file_summaries[batch_file] = f"Part of batch summary: {batch_summary[:200]}..."
        
                small_files_batch = []
                batch_content = ""
                batch_tokens = 0
            

            print(f"[{i}/{len(file_paths)}] Processing large file {relative_path}...")
            summary = summarize_file(path)
            file_summaries[relative_path] = summary
            api_call_count += 1
        
        else:
            file_section = f"=== {relative_path} ===\n{content[:2000]}\n\n" 
            section_tokens = estimate_tokens(file_section)

            if batch_tokens + section_tokens > BATCH_TOKEN_LIMIT and small_files_batch:
                print(f"Processing batch of {len(small_files_batch)} small files...")
                batch_prompt = f"Summarize these code files:\n\n{batch_content}"
                batch_summary = summarize_text(batch_prompt)
                api_call_count += 1
                for batch_file in small_files_batch:
                    file_summaries[batch_file] = f"Part of batch summary: {batch_summary[:200]}..."
                
                small_files_batch = [relative_path]
                batch_content = file_section
                batch_tokens = section_tokens
            else:
                small_files_batch.append(relative_path)
                batch_content += file_section
                batch_tokens += section_tokens
        
        print(f"[{i}/{len(file_paths)}] Processed {relative_path} (API calls so far: {api_call_count})")
    
    if small_files_batch:
        print(f"Processing final batch of {len(small_files_batch)} small files...")
        batch_prompt = f"Summarize these code files:\n\n{batch_content}"
        batch_summary = summarize_text(batch_prompt)
        api_call_count += 1
        for batch_file in small_files_batch:
            file_summaries[batch_file] = f"Part of batch summary: {batch_summary[:200]}..."
    
    print(f"\nFile summarization complete! Total API calls: {api_call_count} (reduced from {len(file_paths)})")
    print("\nGenerating README content...")
    new_content = generate_readme_content(file_summaries)
    api_call_count += 1
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
    
    print(f"README generation complete! Total API calls used: {api_call_count}")
    
    estimated_time = api_call_count * RATE_LIMITS.get(provider, RATE_LIMITS["default"])["delay"]
    print(f"Estimated processing time with rate limits: ~{estimated_time//60}m {estimated_time%60}s")

if __name__ == "__main__":
    main()