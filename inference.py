"""
MedRecordAudit — Baseline Inference Agent
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT
- The script must emit exactly three line types to stdout:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import re
import httpx
from openai import OpenAI

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")  # No default — must be set by user
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK = "MedRecordAudit"
MAX_STEPS = 30

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)


def log_start(task_name: str):
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def call_env(endpoint: str, method: str = "GET", body: dict = None) -> dict:
    """Make an HTTP call to the environment server."""
    url = f"{ENV_URL}{endpoint}"
    with httpx.Client(timeout=60.0) as http:
        if method == "POST":
            response = http.post(url, json=body or {})
        else:
            response = http.get(url)
        response.raise_for_status()
        return response.json()


def ask_llm(prompt: str, max_tokens: int = 1024) -> str:
    """Call the LLM via OpenAI client."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical record auditor AI. You review patient medical records "
                        "to find missed diagnoses, dangerous drug interactions, contradictions "
                        "between doctors, declining lab trends, and monitoring failures. "
                        "Be precise and cite specific record IDs as evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {str(e)}"


def run_episode(difficulty: str = "easy", case_id: str = None):
    """Run a single episode of the environment."""
    step_count = 0
    rewards = []

    # --- RESET ---
    reset_body = {"difficulty": difficulty}
    if case_id:
        reset_body["case_id"] = case_id

    state = call_env("/reset", method="POST", body=reset_body)
    task_name = state.get("case_id", f"{difficulty}_unknown")

    log_start(task_name)

    patient = state.get("patient", {})
    record_index = state.get("record_index", [])
    budget = state.get("budget_remaining", 10)

    # Get task-specific instructions
    task = state.get("task", {})
    task_title = task.get("title", "Medical Record Audit")
    task_instruction = task.get("instruction", "Audit this patient's records for safety issues.")
    focus_areas = task.get("focus_areas", [])
    expected_findings = task.get("expected_findings", 0)

    # --- STEP 1: Ask LLM which records to prioritize ---
    patient_summary = json.dumps(patient, indent=2)
    records_summary = "\n".join(
        f"  ID {r['id']}: [{r['date']}] {r['type']} — {r.get('summary', '')}"
        for r in record_index[:50]
    )

    focus_text = f"\nFOCUS AREAS: {', '.join(focus_areas)}" if focus_areas else ""
    expected_text = f"\nYou are expected to find {expected_findings} issue(s)." if expected_findings else ""

    priority_prompt = f"""TASK: {task_title}

INSTRUCTIONS: {task_instruction}
{focus_text}{expected_text}

PATIENT:
{patient_summary}

AVAILABLE RECORDS ({len(record_index)} total, showing first 50):
{records_summary}

You have a budget of {budget} steps. You need to:
- Read the most important records (prescriptions and lab results first)
- Flag any issues you find
- Submit a report

Which record IDs should I read first? List the top {min(budget - 3, 10)} most important record IDs as a JSON array.
Focus on: prescriptions (drug interactions), lab results (trends), and visit notes from different specialists (contradictions).

Respond with ONLY a JSON array of record IDs, e.g. [3, 7, 12, 18]"""

    llm_response = ask_llm(priority_prompt)

    # Parse record IDs from LLM response
    try:
        match = re.search(r'\[[\d,\s]+\]', llm_response)
        if match:
            priority_ids = json.loads(match.group())
        else:
            prescriptions = [r["id"] for r in record_index if r["type"] == "prescription"]
            labs = [r["id"] for r in record_index if r["type"] == "lab_result"]
            priority_ids = (prescriptions + labs)[:min(budget - 3, 10)]
    except (json.JSONDecodeError, ValueError):
        prescriptions = [r["id"] for r in record_index if r["type"] == "prescription"]
        labs = [r["id"] for r in record_index if r["type"] == "lab_result"]
        priority_ids = (prescriptions + labs)[:min(budget - 3, 10)]

    # --- STEP 2: Read priority records ---
    read_contents = []
    for record_id in priority_ids:
        if budget <= 3:
            break

        result = call_env("/step", method="POST", body={
            "action": "read_record",
            "record_id": record_id,
        })

        step_count += 1
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        rewards.append(reward)

        log_step(step_count, f"read_record({record_id})", reward, done)

        if done:
            break

        record_data = result.get("info", {}).get("record", {})
        read_contents.append(record_data)
        budget = result.get("state", {}).get("budget_remaining", 0)

    # --- STEP 3: Ask LLM to identify issues ---
    records_text = "\n\n".join(
        f"RECORD #{r.get('id', '?')} ({r.get('date', '?')}, {r.get('type', '?')}):\n{json.dumps(r, indent=2)}"
        for r in read_contents[:8]
    )

    analysis_prompt = f"""TASK: {task_title}
INSTRUCTIONS: {task_instruction}
{focus_text}{expected_text}

PATIENT:
{patient_summary}

RECORDS YOU REVIEWED:
{records_text}

Find ALL issues in these records. Look for:
1. Drug interactions (two drugs that shouldn't be given together)
2. Drug contraindications (drug given despite a condition that forbids it)
3. Allergy violations (drug given despite documented allergy)
4. Declining trends (lab values getting worse over time without action)
5. Missed monitoring (required tests not performed)
6. Contradictions (two doctors giving conflicting instructions)
7. Missed diagnoses (symptoms suggesting a condition that was never investigated)

For each issue found, respond in this exact JSON format:
[
  {{
    "type": "drug_interaction|drug_contraindication|allergy_violation|declining_trend|missed_monitoring|contradiction|missed_diagnosis",
    "description": "Clear description of the issue",
    "evidence": [list of record IDs that support this finding]
  }}
]

If no issues found, respond with an empty array: []
Respond with ONLY the JSON array."""

    llm_analysis = ask_llm(analysis_prompt, max_tokens=2048)

    # Parse issues from LLM
    issues = []
    try:
        match = re.search(r'\[.*\]', llm_analysis, re.DOTALL)
        if match:
            issues = json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        issues = []

    # --- STEP 4: Flag issues ---
    for issue in issues:
        if budget <= 1:
            break

        result = call_env("/step", method="POST", body={
            "action": "flag_issue",
            "type": issue.get("type", "drug_interaction"),
            "description": issue.get("description", "Issue found"),
            "evidence": issue.get("evidence", []),
        })

        step_count += 1
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        error = result.get("info", {}).get("error", None)
        rewards.append(reward)

        action_str = f"flag_issue('{issue.get('type', 'unknown')}')"
        log_step(step_count, action_str, reward, done, error)

        if done:
            break
        budget = result.get("state", {}).get("budget_remaining", 0)

    # --- STEP 5: Submit report ---
    # If the env auto-terminated mid-flow (budget exhausted, etc), the
    # last `result` already has the rubric score in `reward`, but its
    # `info` won't have `final_score`. We submit anyway in case the env
    # is still active, then fall back to the last seen reward / 0.01 floor.
    last_reward = result.get("reward", 0.01) if rewards else 0.01

    submit_result = call_env("/step", method="POST", body={"action": "submit_report"})
    step_count += 1
    submit_info = submit_result.get("info", {})

    if "final_score" in submit_info:
        final_score = submit_info["final_score"]
    elif "error" in submit_info:
        # "Episode already ended" path — env auto-terminated earlier.
        # The rubric was computed; use the last meaningful reward we saw.
        final_score = last_reward
    else:
        final_score = submit_result.get("reward", last_reward)

    final_score = max(0.01, min(0.99, final_score))
    rewards.append(final_score)

    log_step(step_count, "submit_report()", final_score, True)

    # --- END ---
    success = final_score >= 0.1
    log_end(success, step_count, final_score, rewards)

    return final_score


ALL_TASKS = [
    ("easy", "easy_001"),
    ("medium", "medium_001"),
    ("hard", "hard_001"),
]


def main():
    """
    Main entry point.

    Usage:
        py inference.py                  # run ALL 9 tasks (what judges run)
        py inference.py easy_001         # run a single task
        py inference.py easy             # run all easy tasks
        py inference.py medium           # run all medium tasks
        py inference.py hard             # run all hard tasks
    """
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "all":
        tasks_to_run = ALL_TASKS
    elif target in ("easy", "medium", "hard"):
        tasks_to_run = [(d, c) for d, c in ALL_TASKS if d == target]
    else:
        difficulty = target.split("_")[0]
        tasks_to_run = [(difficulty, target)]

    scores = {}
    for difficulty, case_id in tasks_to_run:
        print(f"\n# Running {case_id}...", file=sys.stderr)
        try:
            score = run_episode(difficulty=difficulty, case_id=case_id)
            scores[case_id] = score
        except Exception as e:
            print(f"# Error running {case_id}: {e}", file=sys.stderr)
            scores[case_id] = 0.0

    # Summary to stderr
    print(f"\n# RESULTS", file=sys.stderr)
    for case_id, score in scores.items():
        print(f"#   {case_id}: {score:.2f}", file=sys.stderr)
    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"#   Average: {avg:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
