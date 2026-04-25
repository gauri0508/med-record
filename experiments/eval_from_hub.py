"""
Evaluate the trained Qwen-3B model from HuggingFace Hub against all 9 cases.

Loads `gauri0508/med-record-audit-qwen2.5-3b-grpo` from Hub and runs N rollouts
per case using the same single-turn evaluation protocol the trainer used
(`score_completion`). Saves to `experiments/trained.json`.

This script is designed to run on Colab/Kaggle with a GPU. It will NOT work
on the user's CPU MacBook (Qwen-3B is 6 GB; needs CUDA).

Usage on Colab/Kaggle:
    1. Open a fresh notebook
    2. Paste this entire file as a cell, edit MODEL_ID/ENV_URL/HF_TOKEN as needed
    3. Run

If you want to run from inside the existing Kaggle notebook AFTER training,
just paste the `evaluate_all_cases()` body as a new cell at the end of that
notebook — the model+tokenizer are already loaded.
"""

import json
import os
import re

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "gauri0508/med-record-audit-qwen2.5-3b-grpo"
ENV_URL = "https://gauri0508-med-record-audit.hf.space"
NUM_TRIALS = 10

ALL_TASKS = [
    ("easy", "easy_001"),
    ("medium", "medium_001"),
    ("hard", "hard_001"),
]

SYSTEM_PROMPT = """You are a medical record auditor. You will review patient records to find:
- drug_interaction: Two drugs with dangerous interactions
- drug_contraindication: Drug given despite a condition that forbids it
- allergy_violation: Prescribed drug violates documented allergy
- missed_diagnosis: Clinical evidence present but diagnosis not made
- declining_trend: Lab values getting worse without action
- missed_monitoring: Required tests not performed
- contradiction: Conflicting info between providers

Available actions:
- {"action": "read_record", "record_id": <int>}
- {"action": "cross_reference", "query": "<drug or condition>"}
- {"action": "flag_issue", "type": "<one of types above>", "description": "<short clinical>", "evidence": [<record_ids>]}
- {"action": "submit_report"}

Strategy: read relevant records first, then flag issues with evidence, then submit."""


# ---------- env client ----------
class EnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, difficulty: str, case_id: str = None) -> dict:
        payload = {"difficulty": difficulty}
        if case_id:
            payload["case_id"] = case_id
        r = requests.post(f"{self.base_url}/reset", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def step(self, action: dict) -> dict:
        r = requests.post(f"{self.base_url}/step", json=action, timeout=60)
        r.raise_for_status()
        return r.json()


def build_user_prompt(state: dict) -> str:
    patient = json.dumps(state["patient"], indent=2)
    records = json.dumps(state["record_index"][:25], indent=2)
    instruction = state["task"].get("instruction", "Audit this patient's records.")
    expected = state["task"].get("expected_findings", "?")
    budget = state["budget_remaining"]
    return (
        f"Patient:\n{patient}\n\n"
        f"Record index ({len(state['record_index'])} total, showing first 25):\n{records}\n\n"
        f"Task: {instruction}\nBudget: {budget} steps. Expected findings: {expected}\n\n"
        "Output ONLY a JSON array of actions ending with submit_report."
    )


def parse_actions(completion: str) -> list:
    if not completion:
        return []
    match = re.search(r"\[.*\]", completion, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group())
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def score_completion(env_client, case_id: str, difficulty: str, completion: str) -> dict:
    actions = parse_actions(completion)
    try:
        env_client.reset(difficulty, case_id)
    except Exception:
        return {"final_score": 0.01, "rubric_breakdown": {}}

    info_final = {}
    submitted = False
    for action in actions[:30]:
        if not isinstance(action, dict) or "action" not in action:
            continue
        try:
            result = env_client.step(action)
        except Exception:
            continue
        if result.get("done"):
            info_final = result.get("info", {})
            submitted = True
            break

    if not submitted:
        try:
            result = env_client.step({"action": "submit_report"})
            info_final = result.get("info", {})
        except Exception:
            pass

    return {
        "final_score": info_final.get("final_score", 0.01),
        "rubric_breakdown": info_final.get("rubric_breakdown", {}),
    }


def evaluate_case(model, tokenizer, env_client, difficulty: str, case_id: str, num_trials: int) -> dict:
    state = env_client.reset(difficulty, case_id)
    user_prompt = build_user_prompt(state)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    scores = []
    rubrics = []
    for t in range(num_trials):
        out = model.generate(
            input_ids,
            max_new_tokens=768,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        s = score_completion(env_client, case_id, difficulty, completion)
        scores.append(s["final_score"])
        rubrics.append(s["rubric_breakdown"])

    avg = sum(scores) / num_trials
    best_idx = scores.index(max(scores))
    return {
        "difficulty": difficulty,
        "all_scores": [round(s, 4) for s in scores],
        "avg_score": round(avg, 4),
        "best_score": round(max(scores), 4),
        "best_rubric": rubrics[best_idx],
        "n_trials": num_trials,
    }


def main():
    print(f"Loading model from {MODEL_ID}...")
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
    )
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    env_client = EnvClient(ENV_URL)
    print(f"Connected to env: {ENV_URL}")
    print()

    results = {}
    for difficulty, case_id in ALL_TASKS:
        print(f"Evaluating {case_id} ({NUM_TRIALS} trials)...", flush=True)
        result = evaluate_case(model, tokenizer, env_client, difficulty, case_id, NUM_TRIALS)
        results[case_id] = result
        print(
            f"  avg={result['avg_score']:.4f}  best={result['best_score']:.4f}  "
            f"scores={result['all_scores']}"
        )

    avg_overall = sum(r["avg_score"] for r in results.values()) / len(results)
    peak_overall = sum(r["best_score"] for r in results.values()) / len(results)

    summary = {
        "agent": "trained",
        "model": MODEL_ID,
        "n_trials_per_case": NUM_TRIALS,
        "average_score": round(avg_overall, 4),
        "peak_average_score": round(peak_overall, 4),
        "per_case": results,
    }

    out_path = "trained.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"=== Summary ===")
    print(f"Trained model AVG  (over {NUM_TRIALS} trials/case): {avg_overall:.4f}")
    print(f"Trained model PEAK (best of {NUM_TRIALS}/case):     {peak_overall:.4f}")
    print(f"Compare baselines: random=0.167, naive_llm=0.327, smart_llm=0.707")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
