"""
MedRecordAudit — Baseline + Trained Comparison Builder

Reads any subset of:
    experiments/baselines/random.json
    experiments/baselines/untrained_llm.json
    experiments/trained.json   (after Phase 8 GRPO training)

Produces a unified comparison:
    experiments/comparison.json   — machine-readable
    experiments/comparison.md     — markdown table for README

Usage:
    python3 experiments/build_comparison.py
"""

import json
from pathlib import Path

CASES = [
    ("easy", "easy_001"),
    ("medium", "medium_001"),
    ("hard", "hard_001"),
]

BASELINE_FILES = {
    "Random": "experiments/baselines/random.json",
    "Naive LLM": "experiments/baselines/untrained_naive_llm.json",
    "Smart LLM": "experiments/baselines/untrained_llm.json",
    "Trained": "experiments/trained.json",
}


def load_or_none(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def extract_score(data: dict, case_id: str) -> float:
    """Extract per-case score from any of the supported file formats."""
    if data is None:
        return None

    per_case = data.get("per_case", {})
    case = per_case.get(case_id)
    if case is None:
        return None

    # Random uses avg_score across trials; LLM and trained use score
    if "avg_score" in case:
        return case["avg_score"]
    if "score" in case:
        return case["score"]
    return None


def main():
    loaded = {label: load_or_none(path) for label, path in BASELINE_FILES.items()}

    # Build comparison rows
    rows = []
    for difficulty, case_id in CASES:
        row = {"case": case_id, "difficulty": difficulty}
        for label, data in loaded.items():
            row[label] = extract_score(data, case_id)
        rows.append(row)

    # Compute per-agent averages
    averages = {"case": "**Average**", "difficulty": ""}
    for label, data in loaded.items():
        if data is None:
            averages[label] = None
        else:
            averages[label] = data.get("average_score")

    # ---------- write JSON ----------
    Path("experiments").mkdir(exist_ok=True)
    out_json = Path("experiments/comparison.json")
    out_json.write_text(json.dumps({
        "agents_loaded": [k for k, v in loaded.items() if v is not None],
        "agents_missing": [k for k, v in loaded.items() if v is None],
        "rows": rows,
        "averages": averages,
    }, indent=2))

    # ---------- write Markdown ----------
    out_md = Path("experiments/comparison.md")
    cols = ["case", "difficulty"] + list(BASELINE_FILES.keys())

    def fmt_cell(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    lines = []
    lines.append("# MedRecordAudit — Baseline & Trained Comparison\n")
    lines.append("Per-case scores in [0, 1]. Higher is better.\n")
    lines.append("| " + " | ".join(c.capitalize() for c in cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(fmt_cell(row.get(c)) for c in cols) + " |")
    lines.append("| " + " | ".join(
        f"**{fmt_cell(averages.get(c))}**" if averages.get(c) is not None else "—"
        for c in cols
    ) + " |")

    # Notes
    lines.append("")
    for label, data in loaded.items():
        if data is None:
            lines.append(f"- ⚠️ `{label}` baseline not yet captured")

    out_md.write_text("\n".join(lines))

    # ---------- print to stdout ----------
    print(out_md.read_text())
    print()
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
