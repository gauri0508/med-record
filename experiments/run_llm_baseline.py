"""
MedRecordAudit — Untrained LLM Baseline Runner (Phase 0)

Runs inference.py's run_episode() function for all 9 cases and saves
structured JSON output to experiments/baselines/untrained_llm.json
matching the random.json format so the comparison table can pivot
cleanly across all baselines.

Reads these env vars (must be exported before running):
    HF_TOKEN       — your LLM API key (Groq, OpenAI, etc.)
    API_BASE_URL   — LLM API base URL
    MODEL_NAME     — model identifier
    ENV_URL        — deployed HF Space URL

Usage:
    python3 experiments/run_llm_baseline.py
"""

import json
import os
import sys
import time
import traceback
from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path

# Make project root importable so `inference` resolves
sys.path.insert(0, str(Path(__file__).parent.parent))

import inference  # noqa: E402

ALL_TASKS = [
    ("easy", "easy_001"), ("easy", "easy_002"), ("easy", "easy_003"),
    ("medium", "medium_001"), ("medium", "medium_002"), ("medium", "medium_003"),
    ("hard", "hard_001"), ("hard", "hard_002"), ("hard", "hard_003"),
]


def get_clean_episode_data(difficulty: str, case_id: str) -> dict:
    """
    Run one episode via inference.run_episode() and capture both the
    final score AND the rich submit_report info from a follow-up state read.

    Why: inference.run_episode() returns only the float score, but we want
    the rubric_breakdown / correct_findings / etc. for the comparison table.
    We get those by reading the env state immediately after the episode
    ends — submit_report's last call mutated env state with self._last_rubric_scores
    on the server, but we can't access internal state over HTTP. Instead,
    we re-parse the LAST submit response that inference.py made.

    Easiest approach: re-run a thin loop of our own that mirrors inference.py
    but captures submit_report info directly. Below is that approach.
    """
    # We just call the same env endpoints that inference.py does, but here
    # we capture the submit response directly so we get rubric_breakdown.

    # Step 1: capture inference.py's stdout while it runs the episode
    buf = StringIO()
    error = None
    score = 0.0
    started = time.time()

    try:
        with redirect_stdout(buf):
            score = inference.run_episode(difficulty=difficulty, case_id=case_id)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc(file=sys.stderr)

    elapsed = time.time() - started
    log = buf.getvalue()

    # Parse [STEP] and [END] lines for richer info
    n_steps = 0
    rewards = []
    for line in log.splitlines():
        if line.startswith("[STEP]"):
            n_steps += 1
            # parse reward=X.XX
            for token in line.split():
                if token.startswith("reward="):
                    try:
                        rewards.append(float(token.split("=", 1)[1]))
                    except ValueError:
                        pass

    return {
        "case_id": case_id,
        "difficulty": difficulty,
        "score": round(score, 4),
        "n_steps": n_steps,
        "rewards_per_step": rewards,
        "elapsed_seconds": round(elapsed, 1),
        "error": error,
    }


def main():
    # Sanity check env vars
    required = ["HF_TOKEN", "API_BASE_URL", "MODEL_NAME", "ENV_URL"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"ERROR: missing env vars: {missing}", file=sys.stderr)
        print("Set them with `export VAR=value` and re-run.", file=sys.stderr)
        sys.exit(1)

    out_path = Path("experiments/baselines/untrained_llm.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"# Untrained LLM baseline runner", file=sys.stderr)
    print(f"# model:   {os.environ['MODEL_NAME']}", file=sys.stderr)
    print(f"# api:     {os.environ['API_BASE_URL']}", file=sys.stderr)
    print(f"# env_url: {os.environ['ENV_URL']}", file=sys.stderr)
    print(f"# tasks:   {len(ALL_TASKS)} cases", file=sys.stderr)
    print(file=sys.stderr)

    started_at = time.time()
    results = {}

    for difficulty, case_id in ALL_TASKS:
        print(f"  Running {case_id} ...", file=sys.stderr, end=" ", flush=True)
        result = get_clean_episode_data(difficulty, case_id)
        results[case_id] = result
        if result.get("error"):
            print(f"ERROR ({result['error']})", file=sys.stderr)
        else:
            print(f"score={result['score']:.4f}  steps={result['n_steps']}  ({result['elapsed_seconds']:.1f}s)",
                  file=sys.stderr)

    avg = sum(r["score"] for r in results.values() if not r.get("error")) / max(1, sum(1 for r in results.values() if not r.get("error")))
    elapsed = time.time() - started_at

    summary = {
        "agent": "untrained_llm",
        "model": os.environ["MODEL_NAME"],
        "api_base_url": os.environ["API_BASE_URL"],
        "env_url": os.environ["ENV_URL"],
        "elapsed_seconds": round(elapsed, 1),
        "average_score": round(avg, 4),
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
