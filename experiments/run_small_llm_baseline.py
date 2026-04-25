"""
MedRecordAudit — Small / Dumb LLM Baseline (Phase 0 Path C)

Runs the same inference.run_episode() flow but against a *smaller* base model
than the Groq Llama-3.1-8B used for the strong baseline. Establishes a
realistic floor for what a small LLM can do without training, so the
post-training improvement story is dramatic instead of marginal.

Recommended model: meta-llama/Llama-3.2-3B-Instruct via HuggingFace
Inference Providers (OpenAI-compatible router). Requires:
    HF_TOKEN_HF       — your HuggingFace API token (note the suffix —
                        we use a separate var so it doesn't collide
                        with the Groq key in HF_TOKEN)

Falls back gracefully if the env var isn't set.

Usage:
    export HF_TOKEN_HF="hf_..."
    python3 experiments/run_small_llm_baseline.py
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

# Small model config — uses HF Inference Providers router
SMALL_MODEL_CONFIG = {
    "API_BASE_URL": "https://router.huggingface.co/v1",
    # Llama-3.2-3B-Instruct via HF Inference Providers.
    # The :hf-inference suffix uses HuggingFace's own serverless inference
    # (free for non-Pro users, sometimes with cold-start latency).
    "MODEL_NAME": "meta-llama/Llama-3.2-3B-Instruct:hf-inference",
    # We use HF_TOKEN_HF env var so it doesn't collide with HF_TOKEN
    # (which the user set to their Groq key earlier in this session).
    "TOKEN_VAR": "HF_TOKEN_HF",
    "ENV_URL": "https://gauri0508-med-record-audit.hf.space",
}


def configure_inference_for_small_model():
    """Override inference.py's module-level config to use the small model."""
    token = os.environ.get(SMALL_MODEL_CONFIG["TOKEN_VAR"])
    if not token:
        raise RuntimeError(
            f"Set {SMALL_MODEL_CONFIG['TOKEN_VAR']} to your HuggingFace API token first.\n"
            f"  export {SMALL_MODEL_CONFIG['TOKEN_VAR']}=\"hf_...\""
        )

    # Override the module-level OpenAI client in inference.py
    from openai import OpenAI
    inference.client = OpenAI(
        base_url=SMALL_MODEL_CONFIG["API_BASE_URL"],
        api_key=token,
    )
    inference.MODEL_NAME = SMALL_MODEL_CONFIG["MODEL_NAME"]
    inference.API_BASE_URL = SMALL_MODEL_CONFIG["API_BASE_URL"]
    inference.HF_TOKEN = token
    inference.ENV_URL = SMALL_MODEL_CONFIG["ENV_URL"]


def get_episode_data(difficulty: str, case_id: str) -> dict:
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
    n_steps = sum(1 for ln in log.splitlines() if ln.startswith("[STEP]"))
    return {
        "case_id": case_id,
        "difficulty": difficulty,
        "score": round(score, 4),
        "n_steps": n_steps,
        "elapsed_seconds": round(elapsed, 1),
        "error": error,
    }


def main():
    try:
        configure_inference_for_small_model()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    out_path = Path("experiments/baselines/untrained_small_llm.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"# Small/dumb LLM baseline runner", file=sys.stderr)
    print(f"# model:   {inference.MODEL_NAME}", file=sys.stderr)
    print(f"# api:     {inference.API_BASE_URL}", file=sys.stderr)
    print(f"# env_url: {inference.ENV_URL}", file=sys.stderr)
    print(file=sys.stderr)

    started_at = time.time()
    results = {}
    for difficulty, case_id in ALL_TASKS:
        print(f"  Running {case_id} ...", file=sys.stderr, end=" ", flush=True)
        result = get_episode_data(difficulty, case_id)
        results[case_id] = result
        if result.get("error"):
            print(f"ERROR ({result['error']})", file=sys.stderr)
        else:
            print(f"score={result['score']:.4f}  steps={result['n_steps']}  ({result['elapsed_seconds']:.1f}s)",
                  file=sys.stderr)

    valid_scores = [r["score"] for r in results.values() if not r.get("error")]
    avg = sum(valid_scores) / max(1, len(valid_scores))
    elapsed = time.time() - started_at

    summary = {
        "agent": "untrained_small_llm",
        "model": inference.MODEL_NAME,
        "api_base_url": inference.API_BASE_URL,
        "env_url": inference.ENV_URL,
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
