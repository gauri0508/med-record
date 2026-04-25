"""Builder for a small Kaggle notebook that runs the 10-trial eval on the
trained model loaded from HuggingFace Hub. Run once to (re)generate:

    training/eval_kaggle.ipynb

Then upload that .ipynb to Kaggle, set HF_TOKEN secret + GPU T4 x2,
click Save Version → Save & Run All. Output: trained.json available
from the Kaggle Output panel.
"""

import json
from pathlib import Path


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.lstrip("\n").splitlines(keepends=True)}


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.lstrip("\n").splitlines(keepends=True)}


CELLS = [
    md_cell("""
# MedRecordAudit — Trained Model Evaluation

Loads the trained Qwen2.5-3B from `gauri0508/med-record-audit-qwen2.5-3b-grpo`
and runs **10 trials per case** on `easy_001`, `medium_001`, `hard_001` against
the deployed env. Saves the results to `/kaggle/working/trained.json`.

**Setup:**
1. Settings → Accelerator → GPU T4 x2
2. Add-ons → Secrets → add `HF_TOKEN` (your hf_... token), toggle attach ON
3. Save Version → Save & Run All
"""),

    md_cell("## Cell 1 — Install dependencies + load HF_TOKEN\n\nUses Unsloth to load (the model on Hub uses Unsloth's hybrid quantization format that plain `transformers` can't reload directly)."),
    code_cell("""
!pip install -q unsloth "trl>=0.11.0" "transformers>=4.45" requests

import os
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
print(f"HF_TOKEN present: {bool(os.environ.get('HF_TOKEN'))}")
"""),

    md_cell("## Cell 2 — Load trained model via Unsloth"),
    code_cell("""
from unsloth import FastLanguageModel

MODEL_ID = "gauri0508/med-record-audit-qwen2.5-3b-grpo"
ENV_URL = "https://gauri0508-med-record-audit.hf.space"

print(f"Loading {MODEL_ID} via Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=4096,
    dtype=None,             # auto-detect from saved checkpoint
    load_in_4bit=True,      # the saved checkpoint is bnb-4bit
    token=os.environ["HF_TOKEN"],
)
FastLanguageModel.for_inference(model)
print(f"Loaded: {sum(p.numel() for p in model.parameters()):,} params on {model.device}")
"""),

    md_cell("## Cell 3 — Define helpers (env client, prompt, scorer)"),
    code_cell("""
import json, re, requests

SYSTEM_PROMPT = \"\"\"You are a medical record auditor. You will review patient records to find:
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

Strategy: read relevant records first, then flag issues with evidence, then submit.\"\"\"


class EnvClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")
    def reset(self, difficulty, case_id=None):
        body = {"difficulty": difficulty}
        if case_id: body["case_id"] = case_id
        r = requests.post(f"{self.base_url}/reset", json=body, timeout=60)
        r.raise_for_status()
        return r.json()
    def step(self, action):
        r = requests.post(f"{self.base_url}/step", json=action, timeout=60)
        r.raise_for_status()
        return r.json()


env_client = EnvClient(ENV_URL)


def build_user_prompt(state):
    return (
        f"Patient:\\n{json.dumps(state['patient'], indent=2)}\\n\\n"
        f"Record index ({len(state['record_index'])} total, showing first 25):\\n"
        f"{json.dumps(state['record_index'][:25], indent=2)}\\n\\n"
        f"Task: {state['task'].get('instruction', 'Audit.')}\\n"
        f"Budget: {state['budget_remaining']} steps. "
        f"Expected findings: {state['task'].get('expected_findings', '?')}\\n\\n"
        "Output ONLY a JSON array of actions ending with submit_report."
    )


def parse_actions(completion):
    if not completion: return []
    m = re.search(r"\\[.*\\]", completion, re.DOTALL)
    if not m: return []
    try:
        p = json.loads(m.group())
        return p if isinstance(p, list) else []
    except Exception:
        return []


def score_completion(case_id, difficulty, completion):
    actions = parse_actions(completion)
    try: env_client.reset(difficulty, case_id)
    except: return {"final_score": 0.01, "rubric_breakdown": {}}
    info_final = {}; submitted = False
    for action in actions[:30]:
        if not isinstance(action, dict) or "action" not in action: continue
        try: result = env_client.step(action)
        except: continue
        if result.get("done"):
            info_final = result.get("info", {}); submitted = True; break
    if not submitted:
        try:
            r = env_client.step({"action": "submit_report"})
            info_final = r.get("info", {})
        except: pass
    return {"final_score": info_final.get("final_score", 0.01),
            "rubric_breakdown": info_final.get("rubric_breakdown", {})}


print("Helpers ready.")
"""),

    md_cell("## Cell 4 — Run 10-trial evaluation on 3 cases"),
    code_cell("""
ALL_TASKS = [
    ("easy", "easy_001"),
    ("medium", "medium_001"),
    ("hard", "hard_001"),
]
NUM_TRIALS = 10

results = {}
for difficulty, case_id in ALL_TASKS:
    print(f"Evaluating {case_id} ({NUM_TRIALS} trials)...", flush=True)
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
    for t in range(NUM_TRIALS):
        out = model.generate(
            input_ids, max_new_tokens=512, temperature=0.7, do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        completion = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        s = score_completion(case_id, difficulty, completion)
        scores.append(s["final_score"])
        rubrics.append(s["rubric_breakdown"])

    avg = sum(scores) / NUM_TRIALS
    best_idx = scores.index(max(scores))
    results[case_id] = {
        "difficulty": difficulty,
        "all_scores": [round(s, 4) for s in scores],
        "avg_score": round(avg, 4),
        "best_score": round(max(scores), 4),
        "best_rubric": rubrics[best_idx],
        "n_trials": NUM_TRIALS,
    }
    print(f"  avg={avg:.4f}  best={max(scores):.4f}  scores={[f'{s:.2f}' for s in scores]}")

avg_overall = sum(r["avg_score"] for r in results.values()) / len(results)
peak_overall = sum(r["best_score"] for r in results.values()) / len(results)

trained_data = {
    "agent": "trained",
    "model": MODEL_ID,
    "n_trials_per_case": NUM_TRIALS,
    "average_score": round(avg_overall, 4),
    "peak_average_score": round(peak_overall, 4),
    "per_case": results,
}

with open("/kaggle/working/trained.json", "w") as f:
    json.dump(trained_data, f, indent=2)

print()
print(f"=== Final ===")
print(f"AVG  (10 trials/case): {avg_overall:.4f}")
print(f"PEAK (best of 10/case): {peak_overall:.4f}")
print(f"Saved to /kaggle/working/trained.json")
print(f"Compare baselines: random=0.241, naive_llm=0.072, smart_llm=0.542")
"""),
]


NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main():
    out = Path("training/eval_kaggle.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NOTEBOOK, indent=1))
    print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
