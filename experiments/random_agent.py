"""
MedRecordAudit — Random Agent Baseline (Phase 0)

Picks random valid actions until budget runs out or episode ends.
Establishes the lower bound for the before/after improvement table.

Output format matches the structure used by inference.py results so the
final comparison table can pivot all baselines + trained scores together.

Usage:
    py experiments/random_agent.py [--env-url URL] [--seed SEED] [--out PATH]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import httpx


ALL_TASKS = [
    ("easy", "easy_001"), ("easy", "easy_002"), ("easy", "easy_003"),
    ("medium", "medium_001"), ("medium", "medium_002"), ("medium", "medium_003"),
    ("hard", "hard_001"), ("hard", "hard_002"), ("hard", "hard_003"),
]

# Random vocabulary for cross_reference queries — mix of common drugs/conditions
# so a fraction of queries hit relevant ground-truth terms by chance
QUERY_VOCAB = [
    "warfarin", "metformin", "aspirin", "lisinopril", "atorvastatin",
    "amoxicillin", "penicillin", "ibuprofen", "albuterol", "insulin",
    "diabetes", "hypertension", "asthma", "cardiac", "infection",
    "bleeding", "allergy", "creatinine", "glucose", "potassium",
]

ISSUE_TYPES = [
    "drug_interaction", "drug_contraindication", "allergy_violation",
    "declining_trend", "missed_monitoring", "contradiction", "missed_diagnosis",
]

DESC_TEMPLATES = [
    "Possible drug interaction between two medications",
    "Patient may have an undiagnosed condition based on labs",
    "Lab values appear to be trending in a concerning direction",
    "Possible allergy concern with prescribed medication",
    "Monitoring may not have been performed as scheduled",
    "Conflicting information between provider visit notes",
]


def call_env(env_url: str, endpoint: str, body: dict = None) -> dict:
    """POST to /reset or /step on the deployed env."""
    url = f"{env_url.rstrip('/')}{endpoint}"
    with httpx.Client(timeout=60.0) as http:
        if body is not None:
            r = http.post(url, json=body)
        else:
            r = http.get(url)
        r.raise_for_status()
        return r.json()


def random_action(rng: random.Random, num_records: int) -> dict:
    """Pick a random valid non-terminal action.

    Distribution roughly matches what a confused agent would do:
        56% read_record
        22% cross_reference
        22% flag_issue
    Submit is NOT picked randomly — the runner submits deliberately when
    budget drops to 2 or below, so the env always returns the clean
    submit_report info (with rubric_breakdown, findings_submitted, etc.).
    """
    roll = rng.random()

    if roll < 0.56:
        return {"action": "read_record", "record_id": rng.randint(1, num_records)}

    if roll < 0.78:
        return {"action": "cross_reference", "query": rng.choice(QUERY_VOCAB)}

    # Random flag — likely wrong type, occasionally right by chance
    n_evidence = rng.randint(1, 3)
    evidence = sorted(rng.sample(range(1, num_records + 1), min(n_evidence, num_records)))
    return {
        "action": "flag_issue",
        "type": rng.choice(ISSUE_TYPES),
        "description": rng.choice(DESC_TEMPLATES),
        "evidence": evidence,
    }


def run_episode(env_url: str, difficulty: str, case_id: str, rng: random.Random) -> dict:
    """Run one episode with random actions; return the result dict."""
    state = call_env(env_url, "/reset", {"difficulty": difficulty, "case_id": case_id})
    num_records = state["records_available"]
    budget_total = state["budget_remaining"]

    steps = 0
    rewards = []
    final_score = 0.01
    info_final = {}
    budget_remaining = budget_total

    while True:
        # When budget drops to 2 or below, deliberately submit so we get
        # the full submit_report info (rubric_breakdown, findings_submitted,
        # correct_findings, etc.). If we let budget hit 0 the env force-ends
        # but only returns {"message": ...} with no breakdown.
        if budget_remaining <= 2:
            action = {"action": "submit_report"}
        else:
            action = random_action(rng, num_records)

        result = call_env(env_url, "/step", action)
        steps += 1
        rewards.append(result.get("reward", 0.0))

        if result.get("done"):
            info_final = result.get("info", {})
            final_score = info_final.get("final_score", rewards[-1])
            break

        budget_remaining = result.get("state", {}).get("budget_remaining", 0)

        # Safety cap: should never trigger but prevents runaway loops
        if steps > budget_total + 5:
            sub = call_env(env_url, "/step", {"action": "submit_report"})
            steps += 1
            rewards.append(sub.get("reward", 0.0))
            info_final = sub.get("info", {})
            final_score = info_final.get("final_score", 0.01)
            break

    rubric = info_final.get("rubric_breakdown", {})
    return {
        "case_id": case_id,
        "difficulty": difficulty,
        "score": final_score,
        "steps": steps,
        "findings_submitted": info_final.get("findings_submitted", 0),
        "correct_findings": info_final.get("correct_findings", 0),
        "false_positives": info_final.get("false_positives", 0),
        "rubric_breakdown": rubric,
    }


def main():
    parser = argparse.ArgumentParser(description="Random-agent baseline runner")
    parser.add_argument(
        "--env-url",
        default="https://gauri0508-med-record-audit.hf.space",
        help="Deployed environment URL (default: HF Space)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--out",
        default="experiments/baselines/random.json",
        help="Output JSON path",
    )
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repeats per case (averaged) — random has variance")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"# Random agent baseline", file=sys.stderr)
    print(f"# env_url:  {args.env_url}", file=sys.stderr)
    print(f"# seed:     {args.seed}", file=sys.stderr)
    print(f"# repeats:  {args.repeats} per case (avg) ", file=sys.stderr)
    print(f"# tasks:    {len(ALL_TASKS)} cases", file=sys.stderr)
    print(file=sys.stderr)

    started_at = time.time()
    per_case_results = {}

    for difficulty, case_id in ALL_TASKS:
        case_runs = []
        for trial in range(args.repeats):
            try:
                result = run_episode(args.env_url, difficulty, case_id, rng)
                case_runs.append(result)
                print(
                    f"  {case_id}  trial {trial+1}/{args.repeats}  "
                    f"score={result['score']:.4f}  "
                    f"findings={result['findings_submitted']}  "
                    f"correct={result['correct_findings']}",
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"  {case_id}  trial {trial+1}/{args.repeats}  ERROR: {e}", file=sys.stderr)
                case_runs.append({"case_id": case_id, "difficulty": difficulty,
                                  "score": 0.0, "error": str(e)})

        scores = [r["score"] for r in case_runs if "error" not in r]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        per_case_results[case_id] = {
            "difficulty": difficulty,
            "avg_score": round(avg_score, 4),
            "trials": case_runs,
            "n_trials": len(case_runs),
        }

    avg_overall = sum(r["avg_score"] for r in per_case_results.values()) / len(per_case_results)
    elapsed = time.time() - started_at

    summary = {
        "agent": "random",
        "env_url": args.env_url,
        "seed": args.seed,
        "repeats_per_case": args.repeats,
        "elapsed_seconds": round(elapsed, 1),
        "average_score": round(avg_overall, 4),
        "per_case": per_case_results,
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(file=sys.stderr)
    print(f"# Results saved to: {out_path}", file=sys.stderr)
    print(f"# Average score:    {avg_overall:.4f}", file=sys.stderr)
    print(f"# Elapsed:          {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
