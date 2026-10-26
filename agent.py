import os
import re

import requests
from huggingface_hub import hf_hub_download
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    WikipediaSearchTool,
    LiteLLMModel,
    tool,
)

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
MODEL_ID = os.getenv("MODEL_ID", "anthropic/claude-sonnet-5-5")


@tool
def download_task_file(task_id: str, file_name: str) -> str:
    """Downloads the file attached to a task and returns its local path.

    Args:
        task_id: The task_id of the current question.
        file_name: The attached file's name, as given in the question.
    """
    r = requests.get(f"{DEFAULT_API_URL}/files/{task_id}", timeout=30)
    if r.ok:
        path = os.path.join("/tmp", file_name)
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    # The scoring API often has no files; fall back to the gated GAIA dataset
    # (accept its terms on the Hub and set HF_TOKEN).
    return hf_hub_download(
        repo_id="gaia-benchmark/GAIA",
        repo_type="dataset",
        filename=f"2023/validation/{file_name}",
        token=os.getenv("HF_TOKEN"),
    )


INSTRUCTIONS = """You are answering a GAIA benchmark question. Research carefully, verify facts, then
call final_answer with ONLY the answer:
- a number: digits only, no units, no thousands separators (unless the question asks)
- a string: as few words as possible, no articles, no abbreviations
- a list: comma-separated, following the rules above for each element
Never add explanations or a 'FINAL ANSWER' prefix."""


# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
class BasicAgent:
    def __init__(self):
        # LiteLLM reads the key for the provider in MODEL_ID from the environment
        # (ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, ...).
        model = LiteLLMModel(model_id=MODEL_ID, max_tokens=4096)
        self.agent = CodeAgent(
            tools=[
                DuckDuckGoSearchTool(),
                VisitWebpageTool(),
                WikipediaSearchTool(),
                download_task_file,
            ],
            model=model,
            max_steps=12,
            # Markdown code fences: with the default <code> tags, Claude sometimes
            # answers in its native tool-call XML and the step fails to parse.
            code_block_tags="markdown",
            additional_authorized_imports=[
                "pandas", "openpyxl", "numpy", "json", "re", "math",
                "datetime", "requests", "bs4", "collections", "itertools",
            ],
        )
        print(f"BasicAgent initialized with {MODEL_ID}.")

    def __call__(self, question: str, task_id: str = "", file_name: str = "") -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        prompt = f"{INSTRUCTIONS}\n\nQuestion: {question}"
        if file_name:
            prompt += (
                f"\n\nThis question has an attached file ({file_name}). "
                f"Get it with download_task_file(task_id='{task_id}', file_name='{file_name}')."
            )
        try:
            answer = str(self.agent.run(prompt)).strip()
        except Exception as e:
            print(f"Agent error on {task_id}: {e}")
            return ""
        answer = re.sub(r"^final answer:\s*", "", answer, flags=re.I)
        print(f"Agent answer: {answer}")
        return answer
