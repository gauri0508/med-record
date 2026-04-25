"""
Wrapper around inference.run_episode that does 3 trials per case and averages.

Replaces experiments/baselines/untrained_llm.json with a 3-trial-avg version
so the trained-model comparison is fair (trained eval will also be multi-trial).

Usage:
    python3 experiments/run_llm_baseline_3trials.py
"""

import json
import os
import sys
import time
import traceback
from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import inference  # noqa: E402

ALL_TASKS = [
    ("easy", "easy_001"),
    ("medium", "medium_001"),
    ("hard", "hard_001"),
]
NUM_TRIALS = 3


def run_episode_capture(difficulty: str, case_id: str) -> tuple[float, int]:
    """Run inference.run_episode and capture only the score and step count."""
    buf = StringIO()
    score = 0.0
    started = time.time()
    try:
        with redirect_stdout(buf):
            score = inference.run_episode(difficulty=difficulty, case_id=case_id)
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    log = buf.getvalue()
    n_steps = sum(1 for ln in log.splitlines() if ln.startswith("[STEP]"))
    elapsed = time.time() - started
    return score, n_steps, round(elapsed, 1)


def main():
    required = ["HF_TOKEN", "API_BASE_URL", "MODEL_NAME", "ENV_URL"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"ERROR: missing env vars: {missing}", file=sys.stderr)
        sys.exit(1)

    out_path = Path("experiments/baselines/untrained_llm.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"# Smart LLM baseline — {NUM_TRIALS} trials per case")
    print(f"# model:   {os.environ['MODEL_NAME']}")
    print(f"# api:     {os.environ['API_BASE_URL']}")
    print(f"# env_url: {os.environ['ENV_URL']}")
    print()

    started_at = time.time()
    results = {}
    for difficulty, case_id in ALL_TASKS:
        scores = []
        n_steps_list = []
        elapsed_list = []
        for trial in range(NUM_TRIALS):
            score, n_steps, elapsed = run_episode_capture(difficulty, case_id)
            scores.append(score)
            n_steps_list.append(n_steps)
            elapsed_list.append(elapsed)
            print(
                f"  {case_id:12s}  trial {trial+1}/{NUM_TRIALS}  "
                f"score={score:.4f}  steps={n_steps}  ({elapsed:.1f}s)",
                flush=True,
            )

        avg = sum(scores) / NUM_TRIALS
        results[case_id] = {
            "difficulty": difficulty,
            "avg_score": round(avg, 4),
            "best_score": round(max(scores), 4),
            "trials": [round(s, 4) for s in scores],
            "n_trials": NUM_TRIALS,
        }

    avg_overall = sum(r["avg_score"] for r in results.values()) / len(results)
    elapsed = time.time() - started_at

    summary = {
        "agent": "untrained_llm",
        "model": os.environ["MODEL_NAME"],
        "api_base_url": os.environ["API_BASE_URL"],
        "env_url": os.environ["ENV_URL"],
        "n_trials_per_case": NUM_TRIALS,
        "elapsed_seconds": round(elapsed, 1),
        "average_score": round(avg_overall, 4),
        "per_case": results,
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"# Results saved to: {out_path}")
    print(f"# Smart LLM 3-trial average: {avg_overall:.4f}")
    print(f"# Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
