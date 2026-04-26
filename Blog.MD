# Teaching a Small Model to Catch Medical Errors

*A hackathon story about building MedRecordAudit — an OpenEnv RL environment for medical record auditing, training a 3B model on it, and learning where the wins were real and where they weren't.*

---

## The problem

Every year, around **one in ten** patients are harmed by their own healthcare. The WHO links over **three million deaths globally** to unsafe care. In the US alone, **diagnostic errors** leave nearly **eight hundred thousand** people dead or permanently disabled — every year.

Most of these aren't rare or exotic events. They happen because a critical piece of information — a drug interaction, a declining lab trend, a contradiction between two specialists — is buried in a fragmented patient chart that no clinician can fully cross-reference under time pressure.

This is exactly the kind of long-horizon, multi-hop reasoning that LLMs are nominally good at. But there's almost no standard environment that asks them to *actually do it* — most medical-AI benchmarks are multiple-choice, short-context, or ask the model to generate a single answer rather than navigate evidence across hundreds of records.

So when I joined the Meta PyTorch × Scaler OpenEnv hackathon, I built one.

---

## What MedRecordAudit looks like

MedRecordAudit is a fully OpenEnv-compliant RL environment. The agent is given a patient — labs, prescriptions, visit notes, sometimes years of history — and a finite budget of steps. It has four actions:

```python
read_record(record_id)                       # pull the contents of a specific record
cross_reference(query)                       # look up drug interactions, contraindications
flag_issue(type, description, evidence)      # raise a finding with supporting record IDs
submit_report()                              # end the episode and trigger scoring
```

It must navigate a record index, decide which records actually matter, and flag specific issues — drug interactions, missed diagnoses, allergy violations, declining lab trends, monitoring gaps, contradictions between providers — with evidence record IDs that back them up.

There are **three difficulty tiers**:

| Tier | Records | Hidden issues | Story |
|---|---:|---:|---|
| Easy | ~20 | 1 | One clear interaction, one short visit |
| Medium | ~80 | 3 | Multi-specialist disagreement, mid-length history |
| Hard | ~150 | 5–6 | Years of records, layered errors, declining trends |

The full env contains **750 records and 29 hidden ground-truth issues** across 95 passing tests. The hard cases need genuine planning — read everything and you'll exhaust the step budget before submitting.

---

## The reward function (and why it has five parts)

My first version had a single monolithic reward function. It was *trivial to game*. Submit a high-volume report with every type of issue, copy-paste descriptions, and you could score above-random on cases you'd never read.

So I rewrote it as **five composable rubrics**, each capped at a fraction of the total:

| Rubric | Max | What it measures |
|---|---:|---|
| Finding accuracy | 0.40 | Did the agent find the actual ground-truth issues? |
| Evidence validity | 0.20 | Are the cited record IDs the ones that actually support the finding? |
| Completeness | 0.20 | Did it find *all* the issues, not just one? |
| Efficiency | 0.10 | Did it solve the case within budget? |
| Anti-hacking | 0.10 | Penalties for duplicate flags, length-stuffing, unread evidence |

Plus six explicit anti-hacking guards in the env itself: duplicate-flag rejection, description length caps, evidence-ID validation, deep-copy state (so the agent can't mutate the world), unread-evidence warnings, and a hard cap on flag count per episode.

Why this matters: the rubrics force the agent to *do the work*, not pattern-match to the shape of a "good audit report." If the policy doesn't read the right records, it can't pass evidence validity. If it spams findings, the anti-hacking rubric subtracts more than the finding rubric adds. The reward landscape is shaped to reward the actual behavior I care about.

---

## Training: 150 GRPO steps on a single Kaggle T4

I used **Qwen-2.5-3B-Instruct** as the base, **TRL's `GRPOTrainer`** as the optimizer, and **Unsloth** for fast 4-bit LoRA fine-tuning. Curriculum: 40 steps of pure easy, 50 of easy+medium mixed, 60 of all three difficulties.

A few decisions worth noting:

- **`num_generations=2`** — the GRPO minimum, but anything higher OOMs a T4 at 4096-token context.
- **`max_completion_length=512`** — enough room for the agent's full JSON action plan per episode.
- **Push to Hub every 25 steps** — Kaggle kernels die unpredictably, and checkpointing externally meant I could pick up where I left off.

Total run: **150 steps, ~6.5 hours on a single T4**. Final training-time reward stats: **peak 0.82**, last-step 0.76, mean 0.46 across the run. Sixty-five of 150 steps reached ≥0.5; fifty reached ≥0.7.

---

## The honest results

I evaluated the trained model with **10 trials per case** against four baselines: a random agent, a naive LLM (one-shot prompt), a smart LLM (multi-step prompted Llama-3.1-8B on Groq), and the trained 3B Qwen.

| Case | Random | Naive LLM | Smart LLM (8B) | **Trained (3B)** |
|---|---:|---:|---:|---:|
| easy_001 | 0.22 | 0.01 | 0.79 | **0.78** |
| medium_001 | 0.23 | 0.20 | **0.35** | 0.20 |
| hard_001 | 0.27 | 0.01 | **0.48** | 0.09 |
| **Overall avg** | 0.24 | 0.07 | **0.54** | 0.35 |
| Trained peak (best-of-10) | — | — | — | 0.64 |

This isn't the headline I wanted to write. So let me be honest about what's real and what isn't.

**What's real:**

- **The trained 3B model ties Smart-LLM-8B on easy cases** — 0.78 vs 0.79, at under half the parameters. On its training distribution, RL works exactly as advertised: a small model matches a much larger zero-shot one.
- **It crushes Random by 1.5×** and **Naive LLM by 5×** in average score. RL teaches *something* genuine beyond zero-shot prompting.
- **Peak performance is much higher than average** — best-of-10 averages **0.64**, including 0.69 on medium and 0.40 on hard. The policy *can* solve harder cases; it just doesn't sample reliably yet.

**What's not real:**

- The trained model **doesn't generalize to medium and hard** in average performance. It overfits to the easy distribution. On hard cases, it averages 0.09 — *worse than random*.
- This isn't surprising for 150 GRPO steps on a 3-case curriculum and a single T4 GPU. It is, however, the honest truth of what a one-day hackathon train run actually buys you.

I could have cherry-picked the easy result and called it a win. The community is better served if I don't.

---

## Three things I learned

**1. The environment matters more than the model.** The most reusable artifact from this project isn't the trained checkpoint — it's the environment. Other people can train for longer, on bigger GPUs, with broader curricula, and the eval surface stays the same. That's the OpenEnv promise: agents come and go, environments accumulate value.

**2. Anti-hacking has to be designed in, not bolted on.** The first time I trained, the model learned to spam `flag_issue` with empty descriptions because the early reward function rewarded volume. Five rubrics and six guards later, it can't. If you're building an RL environment, assume your agent will find the cheapest path to reward, and shape the reward landscape accordingly.

**3. RL on small models with limited compute does *something* — but generalization needs steps you don't have in a hackathon.** The peak scores prove the policy is in there. Pulling it out reliably across difficulty tiers is a different — and much longer — game. A 3B model with 150 GRPO steps can match an 8B model on its training distribution. Generalizing across distributions is a 1000-step problem, not a 150-step one.

---

## Why this matters beyond a hackathon

The number that opened this post — eight hundred thousand Americans dead or disabled per year from diagnostic errors — is the headline. But the deeper number is the one I had to confront in the results: **0.09 on hard cases**. That's not a model failure; it's a measurement. It says, *here is exactly how much my one-day RL run can reason across 150 records and five interconnected errors.* Without an environment that surfaces that gap, you can't even begin to close it.

Building benchmarks that resist gaming, that score the *behavior* and not the format, that make small models actually try — that's the work. Training a model on it for longer than 150 steps is the easy part.

---

## Try it

Everything is open:

- **Live env (HuggingFace Space):** https://gauri0508-med-record-audit.hf.space
- **Model card:** https://huggingface.co/gauri0508/med-record-audit-qwen2.5-3b-grpo
- **Code:** https://github.com/gauri0508/med-record
- **90-second demo video:** https://youtu.be/stKO4MCaSh0

If you train this for longer, on a different base model, or with a richer curriculum — tag me. I'd genuinely like to see how far this env can go.

— Gauri
