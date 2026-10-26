"""Run the agent on a few questions locally, without submitting.

Usage: python try_local.py [N]   (default: 3 questions)
"""
import sys

import requests

from agent import DEFAULT_API_URL, BasicAgent

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
questions = requests.get(f"{DEFAULT_API_URL}/questions", timeout=15).json()[:n]
agent = BasicAgent()
for q in questions:
    answer = agent(q["question"], q["task_id"], q.get("file_name", ""))
    print(f"\n=== {q['task_id']}\nQ: {q['question'][:200]}\nA: {answer}\n")
