"""Answer all questions locally, then submit them to the scoring API.

Hosting a Gradio Space needs HF PRO, so this does what app.py does without one.
Answers are cached in answers.json, so a rerun only asks the unanswered
questions; delete an entry (or the file) to retry it.

Usage:
  python submit_local.py --code-url <public link to this code>          # run + submit
  python submit_local.py --code-url <link> --dry-run                    # run, don't submit
"""
import argparse
import json
import os

import requests

from agent import DEFAULT_API_URL, BasicAgent

CACHE = "answers.json"

parser = argparse.ArgumentParser()
parser.add_argument("--username", default="aplemenos")
parser.add_argument("--code-url", required=True, help="public link to your agent code")
parser.add_argument("--dry-run", action="store_true", help="answer but don't submit")
args = parser.parse_args()

answers: dict[str, str] = {}
if os.path.exists(CACHE):
    with open(CACHE) as f:
        answers = json.load(f)

questions = requests.get(f"{DEFAULT_API_URL}/questions", timeout=15).json()
todo = [q for q in questions if not answers.get(q["task_id"])]
print(f"{len(questions)} questions, {len(todo)} to answer")

agent = BasicAgent() if todo else None
for i, q in enumerate(todo, 1):
    print(f"\n[{i}/{len(todo)}] {q['question'][:120]}")
    answers[q["task_id"]] = agent(q["question"], q["task_id"], q.get("file_name", ""))  # type: ignore[misc]
    with open(CACHE, "w") as f:
        json.dump(answers, f, indent=2)

print("\nAnswers:")
for q in questions:
    print(f"- {q['question'][:70]!r} -> {answers.get(q['task_id'], '')!r}")

if args.dry_run:
    raise SystemExit("\nDry run: not submitted.")

payload = {
    "username": args.username,
    "agent_code": args.code_url,
    "answers": [
        {"task_id": q["task_id"], "submitted_answer": answers.get(q["task_id"], "")}
        for q in questions
    ],
}
r = requests.post(f"{DEFAULT_API_URL}/submit", json=payload, timeout=60)
print(f"\n{r.status_code}: {r.text}")
