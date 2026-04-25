"""
CurriculumSampler — manages reward history and selects difficulty for next episode.

Used by the training script (Phase 7 GRPO Colab) to drive progressive
difficulty escalation. The environment also supports curriculum mode natively
via difficulty="curriculum" parameter in reset() — both paths use the same
threshold constants for consistency.

Schedule:
    easy → medium when avg reward over last N episodes >= 0.35
    medium → hard when avg reward over last N episodes >= 0.55

A small fraction of episodes always sampled at "easy" (default 20%) to
prevent catastrophic forgetting once the curriculum advances.
"""

import random
from typing import List, Tuple


class CurriculumSampler:
    """
    Tracks reward history and returns (difficulty, case_id) pairs for episodes.

    Args:
        threshold_easy: avg reward to advance from easy → medium (default 0.35).
        threshold_medium: avg reward to advance from medium → hard (default 0.55).
        history_window: how many recent rewards to average (default 10).
        easy_mix_rate: fraction of non-easy episodes that are forced to easy
            for catastrophic-forgetting prevention (default 0.20).
        cases_per_difficulty: number of distinct cases at each difficulty
            (default 3 — easy_001/002/003, etc).
        rng_seed: optional integer to make sample_case_id() deterministic.
    """

    def __init__(
        self,
        threshold_easy: float = 0.35,
        threshold_medium: float = 0.55,
        history_window: int = 10,
        easy_mix_rate: float = 0.20,
        cases_per_difficulty: int = 1,
        rng_seed: int = None,
    ):
        self.threshold_easy = threshold_easy
        self.threshold_medium = threshold_medium
        self.history_window = history_window
        self.easy_mix_rate = easy_mix_rate
        self.cases_per_difficulty = cases_per_difficulty
        self.reward_history: List[float] = []
        self.episode_count = 0
        # Local Random instance — does not affect the global random.random
        # state, so tests using global random aren't perturbed.
        self._rng = random.Random(rng_seed) if rng_seed is not None else random

    def record_reward(self, reward: float) -> None:
        """Append the reward of the just-finished episode to history."""
        self.reward_history.append(float(reward))
        self.episode_count += 1

    def current_difficulty(self) -> str:
        """Return the current curriculum stage based on recent reward window."""
        recent = self.reward_history[-self.history_window:]
        if not recent:
            return "easy"
        avg = sum(recent) / len(recent)
        if avg >= self.threshold_medium:
            return "hard"
        if avg >= self.threshold_easy:
            return "medium"
        return "easy"

    def sample_case_id(self) -> Tuple[str, str]:
        """
        Returns (difficulty, case_id) for the next episode.

        At higher curriculum stages (medium/hard), a fraction (`easy_mix_rate`)
        of episodes are forced to "easy" to prevent forgetting earlier skills.
        """
        difficulty = self.current_difficulty()
        if difficulty != "easy" and self._rng.random() < self.easy_mix_rate:
            difficulty = "easy"
        case_num = self._rng.randint(1, self.cases_per_difficulty)
        case_id = f"{difficulty}_{case_num:03d}"
        return difficulty, case_id

    def summary(self) -> dict:
        """Diagnostic snapshot for training logs / W&B."""
        recent = self.reward_history[-self.history_window:]
        recent_avg = sum(recent) / len(recent) if recent else 0.0
        total_avg = sum(self.reward_history) / len(self.reward_history) if self.reward_history else 0.0
        return {
            "episodes": self.episode_count,
            "current_difficulty": self.current_difficulty(),
            "recent_avg": round(recent_avg, 4),
            "total_avg": round(total_avg, 4),
            "history_window": self.history_window,
        }
