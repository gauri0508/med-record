"""
MedRecordAudit — Naive LLM Baseline (Phase 0 Path C, Plan B)

Runs a deliberately-unsophisticated LLM agent against the env. No prompt
engineering, no multi-step reasoning, no record-reading strategy. Just:

  1. Reset the env
  2. Show the LLM the patient + record summaries (only)
  3. Ask "find issues, list them as JSON"
  4. Flag whatever it says
  5. Submit

This represents what a developer would do if they just wired an LLM
into the env without thinking. It's the realistic "untrained, naively
used" baseline — a fair lower bound that any sensible training
pipeline should beat.

Reads env vars (same as inference.py — uses Groq):
    HF_TOKEN       — Groq API key
    API_BASE_URL   — Groq API URL (https://api.groq.com/openai/v1)
    MODEL_NAME     — model identifier (default llama-3.1-8b-instant)
    ENV_URL        — deployed HF Space URL

Usage:
    python3 experiments/run_naive_llm_baseline.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI


ALL_TASKS = [
    ("easy", "easy_001"), ("easy", "easy_002"), ("easy", "easy_003"),
    ("medium", "medium_001"), ("medium", "medium_002"), ("medium", "medium_003"),
    ("hard", "hard_001"), ("hard", "hard_002"), ("hard", "hard_003"),
]


def call_env(env_url: str, endpoint: str, body: dict = None) -> dict:
    url = f"{env_url.rstrip('/')}{endpoint}"
    with httpx.Client(timeout=120.0) as http:
        r = http.post(url, json=body or {})
        r.raise_for_status()
        return r.json()


def naive_episode(client: OpenAI, model: str, env_url: str,
                  difficulty: str, case_id: str) -> dict:
    """
    Naive agent: one LLM call, blind flag, submit.

    Deliberately does NOT:
      - Read any records (so evidence_validity rubric will tank)
      - Cross-reference any drug/condition databases
      - Iterate or refine

    Just looks at the record-index summaries (already returned by reset)
    and tries to identify issues from those alone.
    """
    started = time.time()
    state = call_env(env_url, "/reset", {"difficulty": difficulty, "case_id": case_id})

    patient = state["patient"]
    record_index = state["record_index"]
    instruction = state.get("task", {}).get("instruction", "Audit this patient's records.")

    # Minimal prompt — no system message, no clever scaffolding, no examples.
    # This is what someone would write in 2 minutes without thinking.
    prompt = f"""Patient: {json.dumps(patient)}

Records:
{chr(10).join(f"  ID {r['id']}: {r.get('summary','')}" for r in record_index)}

{instruction}

Reply with a JSON list of issues you spot. Each issue must be:
{{"type": "drug_interaction|drug_contraindication|allergy_violation|declining_trend|missed_monitoring|contradiction|missed_diagnosis", "description": "short", "evidence": [record_ids]}}

Just the JSON, no other text."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,  # higher than inference.py's 0.2 — more naive
        )
        llm_text = response.choices[0].message.content.strip()
    except Exception as e:
        # On API failure, submit empty — gives 0.01 floor
        result = call_env(env_url, "/step", {"action": "submit_report"})
        return {
            "case_id": case_id, "difficulty": difficulty,
            "score": result["info"].get("final_score", 0.01),
            "n_steps": 1, "elapsed_seconds": round(time.time() - started, 1),
            "error": f"LLM call failed: {e}",
        }

    # Parse the JSON list — naive parsing, accepts whatever
    issues = []
    try:
        match = re.search(r"\[.*\]", llm_text, re.DOTALL)
        if match:
            issues = json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        issues = []

    # Flag each issue WITHOUT reading any records first (naive!)
    flagged = 0
    rejected = 0
    for issue in issues[:10]:  # cap at 10 to leave budget
        try:
            r = call_env(env_url, "/step", {
                "action": "flag_issue",
                "type": issue.get("type", "drug_interaction"),
                "description": str(issue.get("description", "issue"))[:480],
                "evidence": issue.get("evidence", []) or [],
            })
            if "error" in r["info"]:
                rejected += 1
            else:
                flagged += 1
        except Exception:
            rejected += 1
        if r.get("done"):
            break

    # Submit
    final = call_env(env_url, "/step", {"action": "submit_report"})
    info = final["info"]

    return {
        "case_id": case_id,
        "difficulty": difficulty,
        "score": info.get("final_score", 0.01),
        "findings_submitted": flagged,
        "findings_rejected": rejected,
        "correct_findings": info.get("correct_findings", 0),
        "false_positives": info.get("false_positives", 0),
        "rubric_breakdown": info.get("rubric_breakdown", {}),
        "n_steps": flagged + 1,  # flag attempts + submit
        "elapsed_seconds": round(time.time() - started, 1),
        "raw_llm_issues_count": len(issues),
    }


def main():
    required = ["HF_TOKEN", "API_BASE_URL", "MODEL_NAME", "ENV_URL"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"ERROR: missing env vars: {missing}", file=sys.stderr)
        print("Set them with `export VAR=value` and re-run.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["HF_TOKEN"],
    )
    model = os.environ["MODEL_NAME"]
    env_url = os.environ["ENV_URL"]

    out_path = Path("experiments/baselines/untrained_naive_llm.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"# Naive LLM baseline runner", file=sys.stderr)
    print(f"# model:   {model}", file=sys.stderr)
    print(f"# api:     {os.environ['API_BASE_URL']}", file=sys.stderr)
    print(f"# env_url: {env_url}", file=sys.stderr)
    print(f"# strategy: 1 LLM call, no record reads, blind flag, submit", file=sys.stderr)
    print(file=sys.stderr)

    started_at = time.time()
    results = {}
    for difficulty, case_id in ALL_TASKS:
        print(f"  Running {case_id} ...", file=sys.stderr, end=" ", flush=True)
        try:
            result = naive_episode(client, model, env_url, difficulty, case_id)
            results[case_id] = result
            print(f"score={result['score']:.4f}  flagged={result.get('findings_submitted',0)}  "
                  f"correct={result.get('correct_findings',0)}  ({result['elapsed_seconds']:.1f}s)",
                  file=sys.stderr)
        except Exception as e:
            results[case_id] = {"case_id": case_id, "difficulty": difficulty,
                                "score": 0.0, "error": str(e)}
            print(f"ERROR ({e})", file=sys.stderr)

    valid = [r["score"] for r in results.values() if not r.get("error")]
    avg = sum(valid) / max(1, len(valid))
    elapsed = time.time() - started_at

    summary = {
        "agent": "untrained_naive_llm",
        "model": model,
        "api_base_url": os.environ["API_BASE_URL"],
        "env_url": env_url,
        "elapsed_seconds": round(elapsed, 1),
        "average_score": round(avg, 4),
        "strategy": "1 LLM call, no reads, blind flag, submit",
        "per_case": results,
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(file=sys.stderr)
    print(f"# Results saved to: {out_path}", file=sys.stderr)
    print(f"# Average score:    {avg:.4f}", file=sys.stderr)
    print(f"# Elapsed:          {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
