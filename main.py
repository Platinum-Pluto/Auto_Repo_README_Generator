import os
from llm import available_client, api_call, summarizer_message, readme_gen_message

client, provider = available_client()

EXCLUDE_DIRS = {'.git', '__pycache__', 'node_modules', 'venv', 'env', '.idea', 'build', 'dist'}
EXCLUDE_FILES = {'README.md', 'LICENSE', '.gitignore'}
EXTENSIONS = (".py", ".js", ".ts", ".java", ".cpp", ".h", ".md", ".html", ".css", ".go", ".rs") 

MAX_TOKENS = 100000  
CHUNK_SIZE = 5000  

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
    if estimate_tokens(content) <= MAX_TOKENS // 2:  
        return summarize_text(f"Summarize the following code/file: {relative_path}\n{content}")

    chunks = [content[i:i + CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
    chunk_summaries = []
    for i, chunk in enumerate(chunks, 1):
        summary = summarize_text(f"Summarize chunk {i}/{len(chunks)} of file {relative_path}:\n{chunk}")
        chunk_summaries.append(summary)

    combined_chunks = "\n".join(chunk_summaries)
    return summarize_text(f"Combine these chunk summaries into a cohesive file summary for {relative_path}:\n{combined_chunks}")

def summarize_text(prompt):
    """Call LLM to summarize text, with robust error handling."""
    try:
        response = api_call(client, provider, prompt, summarizer_message)
        return response
    except Exception as e:
        error_message = str(e).lower()
        if "context" in error_message and "length" in error_message:
            print("Context too long; skipping or handling.")
            return "Summary skipped due to excessive length."
        elif "rate limit" in error_message or "quota" in error_message:
            print("Rate limit or quota exceeded.")
            return "Summary skipped due to rate limit."
        elif "timeout" in error_message:
            print("Request timed out.")
            return "Summary skipped due to timeout."
        else:
            print(f"Unexpected API error: {e}")
            return "Error generating summary."


def generate_readme_content(file_summaries):
    """Generate README using summaries of files."""
    prompt = "Generate a comprehensive README.md in Markdown format for this repository. Include sections like Project Title, Description, Installation, Usage, Contributing, and License. Base it on the following file summaries:\n\n"
    
    for path, summary in file_summaries.items():
        prompt += f"### File: {path}\n{summary}\n\n"
    
    if estimate_tokens(prompt) > MAX_TOKENS:
        print("Overall prompt too large; summarizing summaries.")
        meta_summaries = "\n".join([f"{path}: {summarize_text(summary)}" for path, summary in file_summaries.items()])
        prompt = prompt.replace("\n\n" + "\n\n".join([f"### File: {path}\n{summary}\n\n" for path, summary in file_summaries.items()]), meta_summaries)
    
    try:
        response = api_call(client, provider, prompt, readme_gen_message)
        return response
    except Exception as e:
        print(f"README generation error: {e}")
        return "# README.md\n\nError generating documentation due to repository size."

def main():
    repo_dir = os.getcwd()
    readme_path = os.path.join(repo_dir, "README.md")
    file_paths = collect_file_paths(repo_dir)
    

    file_summaries = {}
    for path in file_paths:
        relative_path = os.path.relpath(path, repo_dir)
        summary = summarize_file(path)
        file_summaries[relative_path] = summary
        print(f"Summarized {relative_path}")
    

    new_content = generate_readme_content(file_summaries)
    

    if not os.path.exists(readme_path):
        print("Creating README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    else:
        with open(readme_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()
        if existing_content != new_content:
            print("Updating README.md")
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        else:
            print("README.md is up-to-date")

if __name__ == "__main__":
    main()