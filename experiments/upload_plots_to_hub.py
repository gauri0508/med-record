"""
Upload the 4 README plots to the HF Model repo so README image embeds render
on GitHub, on the HF Space, and on the model card.

Usage:
    export HF_TOKEN_HF="hf_..."  # your HuggingFace token (write permission)
    python3 experiments/upload_plots_to_hub.py
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "gauri0508/med-record-audit-qwen2.5-3b-grpo"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "assets" / "plots"
PLOTS = [
    "baseline_vs_trained.png",
    "total_reward_curve.png",
    "reward_components.png",
    "loss_and_kl.png",
]


def main():
    token = os.environ.get("HF_TOKEN_HF") or os.environ.get("HF_HUB_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN_HF (or HF_HUB_TOKEN) to your HuggingFace token", file=sys.stderr)
        print("       (HF_TOKEN is reserved for the Groq API key earlier in this session)", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)

    for name in PLOTS:
        local = PLOTS_DIR / name
        if not local.exists():
            print(f"  SKIP {name} (not found at {local})")
            continue
        size_kb = local.stat().st_size // 1024
        print(f"  uploading {name} ({size_kb} KB)...", flush=True)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=name,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Add training plot: {name}",
        )
        print(f"  ✓ https://huggingface.co/{REPO_ID}/resolve/main/{name}")


if __name__ == "__main__":
    main()
