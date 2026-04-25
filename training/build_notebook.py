"""
Builder script for training/train_grpo.ipynb.

Why a builder script: Jupyter .ipynb files are JSON with very strict
nesting and escaping rules. Hand-editing them is error-prone. This
script defines each cell as plain Python source then assembles a valid
.ipynb file. Run once to (re)generate the notebook:

    python3 training/build_notebook.py

The notebook follows MASTER_PLAN.md Phase 7 (lines 1110-1368), with
only the user-approved deviations:
  - No W&B (skipped wandb install / init / log / finish)
  - Hub model id: gauri0508/med-record-audit-qwen2.5-3b-grpo
  - Env URL: https://gauri0508-med-record-audit.hf.space
  - Cell 7 patched to actually train via TRL's GRPOTrainer (master plan
    Cell 7 was a stub that called run_rollout in a loop without ever
    invoking gradient updates).
"""

import json
from pathlib import Path


def code_cell(source: str) -> dict:
    """Build a Jupyter code cell from a multi-line string."""
    # Split into lines preserving newlines (Jupyter wants list of strings)
    lines = source.lstrip("\n").splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def md_cell(source: str) -> dict:
    """Build a Jupyter markdown cell."""
    lines = source.lstrip("\n").splitlines(keepends=True)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


CELLS = [
    md_cell("""
# MedRecordAudit — GRPO Training

Trains `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` via TRL's `GRPOTrainer`
against the deployed MedRecordAudit env on HuggingFace Spaces. The
5-component rubric breakdown returned by the env's `submit_report`
endpoint is the reward signal, with curriculum progression
from easy → medium → hard cases.

**Run on Google Colab T4 (free).** GPU is required — Unsloth needs CUDA.

Output: a LoRA-adapted Qwen2.5-3B model pushed to
`gauri0508/med-record-audit-qwen2.5-3b-grpo` and a
`trainer_state.json` containing per-step rewards for each of the 5
rubric components, used to generate training curve plots.
"""),

    md_cell("## Cell 1 — Setup\n\nInstall dependencies. Metrics are written to stdout and `trainer_state.json` — no external logger needed."),
    code_cell("""
# Install dependencies — metrics go to stdout and trainer_state.json
!pip install -q "unsloth[colab-new]" "trl>=0.11.0" "transformers>=4.45" datasets matplotlib pandas requests

import os
import re
import json
import requests
from dataclasses import asdict
from pathlib import Path
"""),

    md_cell("## Cell 2 — Configuration\n\nAll hyperparameters in one dict. Adjust `max_steps`, `num_generations`, and `max_completion_length` based on your GPU's time budget."),
    code_cell("""
CONFIG = {
    # Base model — Qwen2.5-3B-Instruct in 4-bit fits a free Colab T4
    "model_name": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",

    # LoRA — only ~30M trainable params (≈1% of the 3B base)
    "lora_r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],

    # GRPO hyperparameters
    "learning_rate": 5e-6,
    "num_generations": 4,    # G in GRPO — number of rollouts per training step
    "max_steps": 250,        # roughly 2–3 hours on T4
    "beta": 0.04,            # KL coefficient (penalizes drift from base model)

    # Episode caps
    "max_episode_steps": 12,
    "max_seq_length": 4096,
    "max_completion_length": 768,

    # Endpoints
    "env_url": "https://gauri0508-med-record-audit.hf.space",
    "hub_model_id": "gauri0508/med-record-audit-qwen2.5-3b-grpo",

    # Output dir for checkpoints + trainer_state.json
    "output_dir": "med-record-audit-qwen2.5-3b-grpo",
}

print(json.dumps(CONFIG, indent=2))
"""),

    md_cell("## Cell 3 — Load model with Unsloth\n\nQwen2.5-3B in 4-bit quantization, with LoRA r=16 adapters attached for parameter-efficient fine-tuning."),
    code_cell("""
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CONFIG["model_name"],
    max_seq_length=CONFIG["max_seq_length"],
    dtype=None,           # auto-detect
    load_in_4bit=True,    # 4-bit quantization for T4
)

model = FastLanguageModel.get_peft_model(
    model,
    r=CONFIG["lora_r"],
    target_modules=CONFIG["target_modules"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print(f"Loaded {CONFIG['model_name']} with LoRA r={CONFIG['lora_r']}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
"""),

    md_cell("## Cell 4 — Environment client\n\nThin HTTP wrapper around the deployed env. Each `step()` call returns `{state, reward, done, info}`; the `info` dict on `submit_report` contains the 5 rubric components used for training rewards."),
    code_cell("""
class MedRecordClient:
    \"\"\"Thin HTTP wrapper around the deployed HF Space.\"\"\"

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def reset(self, difficulty: str, case_id: str = None) -> dict:
        payload = {"difficulty": difficulty}
        if case_id:
            payload["case_id"] = case_id
        r = requests.post(f"{self.base_url}/reset", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, action: dict) -> dict:
        r = requests.post(f"{self.base_url}/step", json=action, timeout=30)
        r.raise_for_status()
        return r.json()


env_client = MedRecordClient(CONFIG["env_url"])

# Verify connection (patient dict has 'id'/'age'/'gender' — no 'name')
state = env_client.reset("easy", "easy_001")
patient = state["patient"]
print(f"Connected! Patient {patient.get('id')} ({patient.get('age')}yo), Budget: {state['budget_remaining']}")
"""),

    md_cell("""## Cell 5 — Rollout + scoring helpers

Two helpers:

1. `run_rollout(...)` — multi-turn rollout used for evaluation and debugging. The model generates one action at a time; the env executes it and the conversation continues until `submit_report` or budget exhaustion.

2. `score_completion(...)` — single-turn scorer used by the reward functions during training. The model emits a JSON action list in one completion, which is replayed on a fresh env episode to compute the rubric breakdown.

Both functions call the same env. The single-turn variant is what `GRPOTrainer` needs (it expects one completion per prompt with no multi-turn loops inside the reward function)."""),
    code_cell("""
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


def build_user_prompt(state: dict) -> str:
    \"\"\"Format a single user prompt from the env's reset state.\"\"\"
    patient = json.dumps(state["patient"], indent=2)
    records = json.dumps(state["record_index"][:25], indent=2)  # cap for prompt length
    instruction = state["task"].get("instruction", "Audit this patient's records.")
    expected = state["task"].get("expected_findings", "?")
    budget = state["budget_remaining"]
    return (
        f"Patient:\\n{patient}\\n\\n"
        f"Record index ({len(state['record_index'])} total, showing first 25):\\n{records}\\n\\n"
        f"Task: {instruction}\\nBudget: {budget} steps. Expected findings: {expected}\\n\\n"
        "Output ONLY a JSON array of actions ending with submit_report. "
        "Example: [{\\\"action\\\": \\\"read_record\\\", \\\"record_id\\\": 1}, "
        "{\\\"action\\\": \\\"flag_issue\\\", \\\"type\\\": \\\"allergy_violation\\\", "
        "\\\"description\\\": \\\"...\\\", \\\"evidence\\\": [1, 2]}, "
        "{\\\"action\\\": \\\"submit_report\\\"}]"
    )


# ---------------------------------------------------------------------------
# Multi-turn rollout — used for evaluation and debugging
# ---------------------------------------------------------------------------
def run_rollout(model, tokenizer, env_client, difficulty: str, case_id: str,
                max_steps: int = None) -> dict:
    \"\"\"Multi-turn rollout. Used for evaluation runs, NOT GRPO training itself.\"\"\"
    if max_steps is None:
        max_steps = CONFIG["max_episode_steps"]

    state = env_client.reset(difficulty, case_id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(state)},
    ]

    step_rewards = []
    all_actions = []
    final_info = {}

    for step in range(max_steps):
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
        with model.disable_adapter() if False else _noop():  # always with adapters
            output = model.generate(
                input_ids, max_new_tokens=256, temperature=0.7, do_sample=True
            )
        response_text = tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        )

        try:
            action = json.loads(response_text.strip())
            if isinstance(action, list):
                action = action[0] if action else {"action": "submit_report"}
        except json.JSONDecodeError:
            action = {"action": "submit_report"}

        try:
            result = env_client.step(action)
        except Exception:
            break

        step_rewards.append(result.get("reward", 0.01))
        all_actions.append(action)
        messages.append({"role": "assistant", "content": response_text})

        if result.get("done"):
            final_info = result.get("info", {})
            break

        messages.append({
            "role": "user",
            "content": (
                f"Result: {json.dumps(result['info'])[:500]}\\n"
                f"Budget remaining: {result['state']['budget_remaining']}"
            ),
        })

    if not final_info:
        # Force submit if we ran out of steps without done
        try:
            result = env_client.step({"action": "submit_report"})
            final_info = result.get("info", {})
        except Exception:
            pass

    return {
        "final_score": final_info.get("final_score", 0.01),
        "rubric_breakdown": final_info.get("rubric_breakdown", {}),
        "step_rewards": step_rewards,
        "num_steps": len(all_actions),
    }


from contextlib import contextmanager
@contextmanager
def _noop():
    yield


# ---------------------------------------------------------------------------
# Single-turn scorer — used by reward functions during GRPO training
# ---------------------------------------------------------------------------
def parse_actions_from_completion(completion: str) -> list:
    \"\"\"Extract a JSON action list from the model's completion text.\"\"\"
    if not completion:
        return []
    # Try to find the outermost JSON array
    match = re.search(r"\\[.*\\]", completion, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group())
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def score_completion(case_id: str, difficulty: str, completion: str) -> dict:
    \"\"\"Replay the model's action list on a fresh env episode; return rubric.

    Returns: {"final_score": float, "rubric_breakdown": dict}
    \"\"\"
    actions = parse_actions_from_completion(completion)

    try:
        env_client.reset(difficulty, case_id)
    except Exception:
        return {"final_score": 0.01, "rubric_breakdown": {}}

    info_final = {}
    submitted = False
    for action in actions[:CONFIG["max_episode_steps"] + 5]:
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

    # Force submit if model never did
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


print("Helpers ready: run_rollout, parse_actions_from_completion, score_completion")
"""),

    md_cell("""## Cell 6 — Reward functions (with caching)

Five reward functions are defined, one per rubric component returned by the env. Each returns a list of floats (one per completion in the batch). To avoid running the env five times per batch, results are cached per batch using a small LRU.

`reward_weights` in `GRPOConfig` is set so only `reward_fn_total` drives gradients — the other four are computed for free (cache hit) and surface as separate columns in `trainer_state.json`, so per-component training curves can be plotted later."""),
    code_cell("""
from collections import OrderedDict
import json as _json


# Tiny LRU: completion-batch tuple -> list of rubric_breakdown dicts
_score_cache: \"OrderedDict[tuple, list]\" = OrderedDict()
_CACHE_MAX = 8


def _completion_to_text(comp) -> str:
    \"\"\"Normalize a TRL completion (str or list-of-messages) to plain text.\"\"\"
    if isinstance(comp, list):
        return "".join(m.get("content", "") for m in comp if isinstance(m, dict))
    return str(comp or "")


def _completion_to_key(comp) -> str:
    \"\"\"Hashable string representation of a completion for the LRU cache.\"\"\"
    if isinstance(comp, list):
        return _json.dumps(comp, sort_keys=True, default=str)
    return str(comp or "")


def _score_batch(case_ids, difficulties, completions) -> list:
    \"\"\"Run env episodes for all completions in a batch; cache by tuple key.\"\"\"
    # Build a hashable cache key — stringify completions (lists aren't hashable).
    key = tuple(
        (cid, diff, _completion_to_key(c))
        for cid, diff, c in zip(case_ids, difficulties, completions)
    )
    if key in _score_cache:
        _score_cache.move_to_end(key)
        return _score_cache[key]

    results = []
    for cid, diff, comp in zip(case_ids, difficulties, completions):
        comp_text = _completion_to_text(comp)
        results.append(score_completion(cid, diff, comp_text))

    _score_cache[key] = results
    if len(_score_cache) > _CACHE_MAX:
        _score_cache.popitem(last=False)
    return results


# ---------------------------------------------------------------------------
# Reward functions — one per rubric component.
# Signatures match TRL GRPOTrainer's expected reward_func contract.
# ---------------------------------------------------------------------------
def reward_fn_total(completions, case_id=None, difficulty=None, **kwargs):
    \"\"\"Overall rubric score from the env. Primary GRPO training signal.\"\"\"
    if case_id is None or difficulty is None:
        return [0.0] * len(completions)
    results = _score_batch(case_id, difficulty, completions)
    return [r["final_score"] for r in results]


def reward_fn_finding_accuracy(completions, case_id=None, difficulty=None, **kwargs):
    if case_id is None or difficulty is None:
        return [0.0] * len(completions)
    results = _score_batch(case_id, difficulty, completions)
    return [r["rubric_breakdown"].get("finding_accuracy", 0.0) for r in results]


def reward_fn_evidence_validity(completions, case_id=None, difficulty=None, **kwargs):
    if case_id is None or difficulty is None:
        return [0.0] * len(completions)
    results = _score_batch(case_id, difficulty, completions)
    return [r["rubric_breakdown"].get("evidence_validity", 0.0) for r in results]


def reward_fn_completeness(completions, case_id=None, difficulty=None, **kwargs):
    if case_id is None or difficulty is None:
        return [0.0] * len(completions)
    results = _score_batch(case_id, difficulty, completions)
    return [r["rubric_breakdown"].get("completeness", 0.0) for r in results]


def reward_fn_anti_hacking(completions, case_id=None, difficulty=None, **kwargs):
    if case_id is None or difficulty is None:
        return [0.0] * len(completions)
    results = _score_batch(case_id, difficulty, completions)
    return [r["rubric_breakdown"].get("anti_hacking", 0.0) for r in results]


# The list of reward functions passed to GRPOTrainer.
REWARD_FUNCTIONS = [
    reward_fn_total,
    reward_fn_finding_accuracy,
    reward_fn_evidence_validity,
    reward_fn_completeness,
    reward_fn_anti_hacking,
]

# Only `total` drives the gradient. Others are logged as separate columns.
REWARD_WEIGHTS = [1.0, 0.0, 0.0, 0.0, 0.0]

print(f"{len(REWARD_FUNCTIONS)} reward functions registered. Weights: {REWARD_WEIGHTS}")
"""),

    md_cell("""## Cell 7 — Build dataset, configure GRPO, train

The training loop. Curriculum is baked into the dataset ordering: easy cases for the first 50 steps, easy + medium for steps 50–150, then a mix of all three difficulties. Each dataset row carries the formatted prompt plus the case metadata (`case_id`, `difficulty`) which flows through to the reward functions as kwargs.

`GRPOTrainer.train()` handles everything: generation, reward computation, advantage normalization, and the policy update."""),
    code_cell("""
import random
import torch
from datasets import Dataset

# Pre-cache patient + records for all 9 cases so dataset build doesn't reset env 250x
print("Caching all 9 cases...")
case_cache = {}
for difficulty in ["easy", "medium", "hard"]:
    for n in [1, 2, 3]:
        cid = f"{difficulty}_{n:03d}"
        s = env_client.reset(difficulty, cid)
        case_cache[cid] = {
            "user_prompt": build_user_prompt(s),
            "difficulty": difficulty,
        }
        print(f"  cached {cid}")


def build_curriculum_dataset(max_steps: int) -> Dataset:
    \"\"\"Static curriculum: easy first, then mix in medium, then hard.

    Stage 1 (steps 0-50):   only easy
    Stage 2 (steps 50-150): easy + medium (50/50)
    Stage 3 (steps 150+):   easy 30% / medium 30% / hard 40%
    \"\"\"
    rng = random.Random(42)
    easy_ids = ["easy_001", "easy_002", "easy_003"]
    medium_ids = ["medium_001", "medium_002", "medium_003"]
    hard_ids = ["hard_001", "hard_002", "hard_003"]

    rows = []
    for step in range(max_steps):
        if step < 50:
            cid = rng.choice(easy_ids)
        elif step < 150:
            cid = rng.choice(easy_ids if rng.random() < 0.5 else medium_ids)
        else:
            r = rng.random()
            if r < 0.3:
                cid = rng.choice(easy_ids)
            elif r < 0.6:
                cid = rng.choice(medium_ids)
            else:
                cid = rng.choice(hard_ids)

        c = case_cache[cid]
        # GRPOTrainer expects the prompt as a list-of-messages (chat format)
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": c["user_prompt"]},
        ]
        rows.append({
            "prompt": prompt,
            "case_id": cid,
            "difficulty": c["difficulty"],
        })
    return Dataset.from_list(rows)


train_dataset = build_curriculum_dataset(CONFIG["max_steps"])
print(f"Dataset built: {len(train_dataset)} rows")
print(f"Stage breakdown: easy-only first 50, easy+medium 50-150, all-mix after")


# ---------------------------------------------------------------------------
# Configure GRPOTrainer
# ---------------------------------------------------------------------------
from trl import GRPOConfig, GRPOTrainer

# T4 GPUs don't support bfloat16 (only fp16). Detect at runtime so the
# notebook works on T4 (Colab free), A100, H100, etc.
_use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print(f"GPU bf16 support: {_use_bf16}  (using {'bf16' if _use_bf16 else 'fp16'})")

grpo_config = GRPOConfig(
    output_dir=CONFIG["output_dir"],
    learning_rate=CONFIG["learning_rate"],
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=CONFIG["num_generations"],
    max_steps=CONFIG["max_steps"],
    max_completion_length=CONFIG["max_completion_length"],
    beta=CONFIG["beta"],
    save_steps=50,
    logging_steps=1,
    bf16=_use_bf16,
    fp16=not _use_bf16,
    optim="adamw_8bit",
    report_to=[],          # disable external logging — use trainer_state.json
    remove_unused_columns=False,  # keep case_id, difficulty for reward fns
    reward_weights=REWARD_WEIGHTS,
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    reward_funcs=REWARD_FUNCTIONS,
    train_dataset=train_dataset,
)

print("Starting training...")
trainer.train()
print("Training complete.")
"""),

    md_cell("""## Cell 7b — Load HF_TOKEN from Colab Secrets

Required for Cell 8 to push the trained model to HuggingFace Hub. Without this, the trained model only exists inside the Colab runtime and is lost when the runtime disconnects.

**Setup before running this cell:**
1. Open the **Secrets** panel in Colab's left sidebar (key 🔑 icon)
2. Add a new secret named `HF_TOKEN` with your HuggingFace API token (must have *Write* permission to your model repo)
3. Toggle **Notebook access** ON for the secret"""),
    code_cell("""
# Load HF_TOKEN — try Colab Secrets first, fall back to os.environ
import os

try:
    from google.colab import userdata  # type: ignore
    token = userdata.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = token
    print("HF_TOKEN loaded from Colab Secrets.")
except Exception as e:
    if os.environ.get("HF_TOKEN"):
        print("HF_TOKEN already set via os.environ.")
    else:
        print(f"WARNING: Could not load HF_TOKEN from Colab Secrets ({e}).")
        print("Cell 8 will save locally only and skip the hub push.")
        print("Add HF_TOKEN as a Colab Secret, or set os.environ['HF_TOKEN'] manually.")

print(f"HF_TOKEN present: {bool(os.environ.get('HF_TOKEN'))}")
"""),

    md_cell("""## Cell 8 — Save and push merged model

Uses Unsloth's `save_pretrained_merged` to merge the LoRA adapters back into the base model and save as 16-bit. Naively upcasting from 4-bit to 16-bit and merging produces broken weights; `save_pretrained_merged` handles this correctly."""),
    code_cell("""
# Save locally
model.save_pretrained_merged(
    CONFIG["output_dir"],
    tokenizer,
    save_method="merged_16bit",
)
print(f"Saved locally to: {CONFIG['output_dir']}")

# Push to HF Hub (HF_TOKEN should have been loaded by previous cell)
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("WARNING: HF_TOKEN not set. Model saved locally but NOT pushed to hub.")
    print("To push later: set os.environ['HF_TOKEN'] and call:")
    print(f"  model.push_to_hub_merged('{CONFIG['hub_model_id']}', tokenizer, save_method='merged_16bit', token='hf_...')")
else:
    model.push_to_hub_merged(
        CONFIG["hub_model_id"],
        tokenizer,
        save_method="merged_16bit",
        token=hf_token,
    )
    print(f"Pushed to: https://huggingface.co/{CONFIG['hub_model_id']}")

# Trainer state JSON contains per-step metrics for plotting later
ts_path = Path(CONFIG["output_dir"]) / "trainer_state.json"
if ts_path.exists():
    print(f"Trainer state with per-step metrics: {ts_path}")
"""),
]


# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------
NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
        "colab": {
            "provenance": [],
            "gpuType": "T4",
        },
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main():
    out = Path("training/train_grpo.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(NOTEBOOK, indent=1))
    print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
