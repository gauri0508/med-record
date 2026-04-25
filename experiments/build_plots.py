"""
MedRecordAudit — Training plots generator.

Reads training/trainer_state.json (produced by GRPOTrainer) and emits 4 PNGs
for the README:

    assets/plots/total_reward_curve.png      Per-step reward + rolling avg
    assets/plots/reward_components.png       5 rubric components on one axes
    assets/plots/loss_and_kl.png             Training loss + KL divergence
    assets/plots/baseline_vs_trained.png     Bar chart: random / naive / smart / trained

The first 3 plots come from trainer_state.json. The 4th plot reads:
    experiments/baselines/random.json
    experiments/baselines/untrained_naive_llm.json
    experiments/baselines/untrained_llm.json
    experiments/trained.json   (optional; produces 3-bar chart if missing)

Usage:
    python3 experiments/build_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np


matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 150
matplotlib.rcParams['savefig.bbox'] = 'tight'

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "assets" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def load_trainer_state() -> list:
    """Returns the per-step log_history list from trainer_state.json."""
    path = ROOT / "training" / "trainer_state.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run training first")
    return json.loads(path.read_text())["log_history"]


def rolling_mean(values: list[float], window: int) -> list[float]:
    return [
        sum(values[max(0, i - window + 1) : i + 1]) / min(window, i + 1)
        for i in range(len(values))
    ]


# ---------------------------------------------------------------------------
# Plot 1: Total reward over training steps
# ---------------------------------------------------------------------------
def plot_total_reward(steps: list[dict]) -> None:
    x = [s["step"] for s in steps]
    rewards = [s["reward"] for s in steps]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, rewards, alpha=0.35, color="steelblue", linewidth=1, label="Per-step reward")
    ax.plot(x, rolling_mean(rewards, 10), color="steelblue", linewidth=2, label="10-step moving avg")
    ax.plot(x, rolling_mean(rewards, 25), color="darkblue", linewidth=2.5, label="25-step moving avg")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Reward (0–1)")
    ax.set_title("MedRecordAudit — GRPO Training Reward")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")

    out = PLOTS / "total_reward_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Plot 2: Per-rubric reward components
# ---------------------------------------------------------------------------
def plot_reward_components(steps: list[dict]) -> None:
    x = [s["step"] for s in steps]
    components = {
        "finding_accuracy": ("rewards/reward_fn_finding_accuracy/mean", "tab:blue"),
        "evidence_validity": ("rewards/reward_fn_evidence_validity/mean", "tab:green"),
        "completeness": ("rewards/reward_fn_completeness/mean", "tab:orange"),
        "anti_hacking": ("rewards/reward_fn_anti_hacking/mean", "tab:red"),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, (key, color) in components.items():
        vals = [s.get(key, 0.0) for s in steps]
        ax.plot(x, rolling_mean(vals, 10), color=color, linewidth=2, label=name)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Component reward (10-step moving avg)")
    ax.set_title("MedRecordAudit — Rubric Components During Training")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)

    out = PLOTS / "reward_components.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Plot 3: Training loss + KL divergence
# ---------------------------------------------------------------------------
def plot_loss_and_kl(steps: list[dict]) -> None:
    x = [s["step"] for s in steps]
    losses = [s["loss"] for s in steps]
    kls = [s.get("kl", 0.0) for s in steps]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_loss = "tab:red"
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Training loss (negative = policy improving)", color=color_loss)
    ax1.plot(x, losses, alpha=0.4, color=color_loss, linewidth=1, label="Loss")
    ax1.plot(x, rolling_mean(losses, 10), color=color_loss, linewidth=2, label="Loss (10-step avg)")
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    color_kl = "tab:blue"
    ax2.set_ylabel("KL divergence from base model", color=color_kl)
    ax2.plot(x, kls, color=color_kl, linewidth=2, label="KL")
    ax2.tick_params(axis="y", labelcolor=color_kl)
    ax2.set_ylim(bottom=0)

    plt.title("MedRecordAudit — Training Loss & KL Divergence")
    out = PLOTS / "loss_and_kl.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Plot 4: Baseline vs trained per-case bars
# ---------------------------------------------------------------------------
def plot_baselines_vs_trained() -> None:
    cases = ["easy_001", "medium_001", "hard_001"]

    def load_or_none(path: str) -> dict | None:
        p = ROOT / path
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def per_case_score(data: dict | None, case_id: str) -> float | None:
        if data is None:
            return None
        c = data.get("per_case", {}).get(case_id)
        if c is None:
            return None
        # support both avg_score and score keys
        return c.get("avg_score", c.get("score"))

    sources = [
        ("Random",    "experiments/baselines/random.json",                "lightgray"),
        ("Naive LLM", "experiments/baselines/untrained_naive_llm.json",  "skyblue"),
        ("Smart LLM", "experiments/baselines/untrained_llm.json",        "steelblue"),
        ("Trained",   "experiments/trained.json",                         "tab:green"),
    ]
    available = []
    for label, path, color in sources:
        data = load_or_none(path)
        if data is not None:
            available.append((label, data, color))

    if not available:
        print("  Skipping baseline_vs_trained — no baseline files found")
        return

    n_groups = len(cases)
    n_bars = len(available)
    bar_w = 0.8 / n_bars
    indices = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, (label, data, color) in enumerate(available):
        scores = [per_case_score(data, c) or 0.0 for c in cases]
        offset = (i - (n_bars - 1) / 2) * bar_w
        ax.bar(indices + offset, scores, bar_w, label=label, color=color, edgecolor="black", linewidth=0.5)

    ax.set_xticks(indices)
    ax.set_xticklabels(cases, rotation=30, ha="right")
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0, 1)
    ax.set_title("MedRecordAudit — Per-case Performance: Baselines vs Trained")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    # Vertical separators between difficulty tiers
    for x in [0.5, 1.5]:
        ax.axvline(x, color="black", alpha=0.2, linewidth=0.8)

    out = PLOTS / "baseline_vs_trained.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    print(f"Output directory: {PLOTS}")
    print()
    steps = load_trainer_state()
    print(f"Loaded {len(steps)} training steps from trainer_state.json")
    plot_total_reward(steps)
    plot_reward_components(steps)
    plot_loss_and_kl(steps)
    plot_baselines_vs_trained()
    print()
    print("All plots written.")


if __name__ == "__main__":
    main()
