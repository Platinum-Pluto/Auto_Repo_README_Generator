# LLM-Powered README Generator

This repository houses a sophisticated Python script designed to **automatically generate comprehensive `README.md` files** for any project directory using advanced Large Language Models (LLMs). By intelligently analyzing code and documentation files, summarizing their content, and orchestrating LLM interactions, this tool streamlines the often tedious process of creating and maintaining project documentation.

It aims to provide developers with an efficient way to quickly document codebases, ensuring projects have a well-structured and informative `README.md` without manual effort.

## Features

*   **Automated README Generation:** Analyzes project files to automatically construct a detailed and structured `README.md` file.
*   **Intelligent File Collection:** Scans target directories, intelligently filtering for relevant source code and documentation files based on predefined extensions, while excluding common non-source directories and files.
*   **Multi-LLM Integration:** Supports integration with various LLM providers (e.g., OpenAI, Anthropic, Mistral, Google), dynamically selecting and utilizing an available client for summarization and generation.
*   **Efficient Token Management:**
    *   **Token Estimation:** Estimates token counts for content to optimize API calls.
    *   **Content Chunking:** For large files, the content is divided into smaller, manageable chunks, which are individually summarized, and then these chunk summaries are combined.
    *   **Batch Processing:** Groups multiple small files together to summarize them in a single LLM call, significantly improving efficiency and reducing API overhead.
*   **Robust API Interaction:** Implements comprehensive error handling mechanisms, including **rate limiting and exponential backoff retry logic**, to ensure reliable communication with LLM APIs, gracefully handling common issues like rate limits, network errors, and timeouts.
*   **Structured README Output:** Generates a `README.md` with standard sections such as Project Title, Description, Features, Installation, Usage, and File Structure.
*   **Fallback Mechanism:** Includes a robust fallback README generation in case the primary LLM generation process encounters critical failures, ensuring a basic level of documentation is always provided.

## Installation

To get this README generator up and running, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/llm-powered-readme-generator.git
    cd llm-powered-readme-generator
    ```
    *(Note: Replace `your_username/llm-powered-readme-generator.git` with the actual repository URL)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    This project requires Python and specific LLM client libraries. A `requirements.txt` file (not provided in summaries, but common) would typically list these. You'll need libraries for whichever LLMs you plan to use (e.g., `openai`, `anthropic`, `mistralai`, `google-generativeai`) and potentially `tiktoken` for token estimation.
    ```bash
    pip install openai anthropic mistralai google-generativeai # Install relevant LLM clients
    # You might also need:
    # pip install tiktoken
    ```

4.  **Configure LLM API Keys:**
    Set your LLM API keys as environment variables. This is the recommended secure practice. Examples include:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key_here"
    export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
    export MISTRAL_API_KEY="your_mistral_api_key_here"
    export GOOGLE_API_KEY="your_google_api_key_here"
    ```
    The system will dynamically choose an available LLM client based on the configured keys.

## Usage

To generate a `README.md` for your project, navigate to the project directory and run the `main.py` script.

1.  **Navigate to your project directory:**
    ```bash
    cd /path/to/your/project
    ```

2.  **Run the script:**
    You can specify a target directory to analyze and an output path for the `README.md`.
    ```bash
    python /path/to/llm-powered-readme-generator/main.py --target-directory . --output README.md
    ```
    *   `--target-directory .`: Specifies the current directory as the project to analyze.
    *   `--output README.md`: Sets the output filename for the generated README.

    If `--target-directory` is omitted, the script might default to the current working directory where `main.py` is executed from. The generated `README.md` will appear in the specified output location.

## File Structure

The core components of this repository are:

*   `main.py`: This is the primary orchestration script. It handles the entire workflow from collecting and filtering files, estimating tokens, managing file chunking and batching, coordinating LLM calls, implementing retry logic, and finally assembling and writing the generated `README.md`.
*   `llm.py`: This module is responsible for managing interactions with various Large Language Models. It abstracts away the complexity of integrating with different providers (OpenAI, Mistral, Anthropic, Google, etc.) and dynamically selects an available LLM client based on configuration.
*   `HF_Space_Ollama/app.py`: This file likely represents a specific deployment or example for integrating with an LLM, possibly Ollama, within a Hugging Face Space environment. Its purpose, as summarized, is to manage interaction with multiple LLMs from various providers and dynamically select an available client, suggesting it might be an alternative or complementary LLM interaction layer tailored for specific deployment scenarios.

## Contributing

We welcome contributions to improve the LLM-Powered README Generator! If you'd like to contribute, please follow these guidelines:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `bugfix/issue-description`.
3.  **Make your changes.**
4.  **Write clear, concise commit messages.**
5.  **Submit a pull request** with a detailed description of your changes.

Please ensure your code adheres to good practices, and feel free to open an issue to discuss any significant changes or new features before starting work.

## License

This project is licensed under the [MIT License](LICENSE).

*(Note: A `LICENSE` file would typically be present in the root of the repository detailing the full license terms. For this README generation, the specific license was not provided in the summaries, so "MIT License" is a common open-source placeholder.)*