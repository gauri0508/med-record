"""
Tests for Phase 4 curriculum mode.

Two surfaces under test:
  1. CurriculumSampler (training/curriculum.py) — manages reward history and
     samples (difficulty, case_id) for the next episode.
  2. MedRecordAuditEnv.reset(difficulty="curriculum", ...) — env-side curriculum
     resolver: picks easy/medium/hard from a passed-in reward history.

Run with: py -m pytest tests/test_curriculum.py -v
"""

import pytest
from env.environment import MedRecordAuditEnv
from training.curriculum import CurriculumSampler


# ============================================================
# CurriculumSampler unit tests
# ============================================================

def test_sampler_starts_at_easy():
    s = CurriculumSampler()
    assert s.current_difficulty() == "easy"
    assert s.episode_count == 0


def test_sampler_low_reward_stays_easy():
    s = CurriculumSampler()
    for _ in range(10):
        s.record_reward(0.10)
    assert s.current_difficulty() == "easy"


def test_sampler_medium_threshold():
    s = CurriculumSampler()
    for _ in range(10):
        s.record_reward(0.40)
    assert s.current_difficulty() == "medium"


def test_sampler_hard_threshold():
    s = CurriculumSampler()
    for _ in range(10):
        s.record_reward(0.60)
    assert s.current_difficulty() == "hard"


def test_sampler_uses_only_recent_window():
    """A long history of high rewards followed by a slump should drop level."""
    s = CurriculumSampler(history_window=10)
    # 50 high rewards
    for _ in range(50):
        s.record_reward(0.80)
    assert s.current_difficulty() == "hard"
    # Then 10 low rewards — recent avg drops below 0.35
    for _ in range(10):
        s.record_reward(0.10)
    assert s.current_difficulty() == "easy"


def test_sampler_record_reward_increments_count():
    s = CurriculumSampler()
    s.record_reward(0.5)
    s.record_reward(0.3)
    assert s.episode_count == 2
    assert len(s.reward_history) == 2


def test_sampler_summary_format():
    s = CurriculumSampler()
    s.record_reward(0.4)
    s.record_reward(0.5)
    summary = s.summary()
    assert summary["episodes"] == 2
    assert summary["current_difficulty"] in ("easy", "medium", "hard")
    assert isinstance(summary["recent_avg"], float)
    assert isinstance(summary["total_avg"], float)


def test_sampler_summary_handles_empty_history():
    s = CurriculumSampler()
    summary = s.summary()
    assert summary["episodes"] == 0
    assert summary["current_difficulty"] == "easy"
    assert summary["recent_avg"] == 0.0


def test_sampler_sample_case_id_format():
    s = CurriculumSampler(rng_seed=42)
    diff, cid = s.sample_case_id()
    assert diff in ("easy", "medium", "hard")
    assert cid.startswith(f"{diff}_")
    assert len(cid.split("_")[1]) == 3  # e.g., 001, 002, 003


def test_sampler_easy_mix_at_high_stage():
    """At hard stage with easy_mix_rate=1.0, every sample is forced to easy."""
    s = CurriculumSampler(easy_mix_rate=1.0, rng_seed=42)
    for _ in range(20):
        s.record_reward(0.80)
    assert s.current_difficulty() == "hard"
    # With easy_mix_rate=1.0, every sample must come back as easy
    for _ in range(10):
        diff, cid = s.sample_case_id()
        assert diff == "easy"
        assert cid.startswith("easy_")


def test_sampler_no_easy_mix_when_at_easy():
    """At easy stage, easy_mix_rate is irrelevant — always easy anyway."""
    s = CurriculumSampler(easy_mix_rate=1.0, rng_seed=42)
    for _ in range(10):
        diff, _ = s.sample_case_id()
        assert diff == "easy"


def test_sampler_thresholds_configurable():
    """Custom thresholds shift the boundaries."""
    s = CurriculumSampler(threshold_easy=0.20, threshold_medium=0.30)
    for _ in range(10):
        s.record_reward(0.25)
    assert s.current_difficulty() == "medium"  # would be easy with defaults


def test_sampler_rng_seed_reproducible():
    """Same seed → same sample sequence."""
    s1 = CurriculumSampler(rng_seed=123)
    s2 = CurriculumSampler(rng_seed=123)
    seq1 = [s1.sample_case_id() for _ in range(10)]
    seq2 = [s2.sample_case_id() for _ in range(10)]
    assert seq1 == seq2


# ============================================================
# Environment curriculum mode (env.reset(difficulty="curriculum"))
# ============================================================

def test_env_curriculum_no_history_returns_easy():
    """Cold start with no history defaults to easy."""
    env = MedRecordAuditEnv()
    state = env.reset(difficulty="curriculum")
    assert state["difficulty"] == "easy"
    assert state["budget_remaining"] == 15  # easy budget


def test_env_curriculum_low_history_returns_easy():
    """Spec AC: history of [0.1]*5 returns easy."""
    env = MedRecordAuditEnv()
    state = env.reset(difficulty="curriculum", curriculum_reward_history=[0.1] * 5)
    assert state["difficulty"] == "easy"
    assert state["budget_remaining"] == 15


def test_env_curriculum_high_history_returns_hard():
    """Spec AC: history of [0.6]*10 returns hard."""
    env = MedRecordAuditEnv()
    state = env.reset(difficulty="curriculum", curriculum_reward_history=[0.6] * 10)
    assert state["difficulty"] == "hard"
    assert state["budget_remaining"] == 30  # hard budget


def test_env_curriculum_medium_threshold():
    """Average between 0.35 and 0.55 returns medium."""
    env = MedRecordAuditEnv()
    state = env.reset(difficulty="curriculum", curriculum_reward_history=[0.45] * 10)
    assert state["difficulty"] == "medium"
    assert state["budget_remaining"] == 25  # medium budget


def test_env_curriculum_uses_window():
    """Long history of high rewards followed by recent slump drops level."""
    env = MedRecordAuditEnv()
    history = [0.9] * 50 + [0.1] * 10
    state = env.reset(difficulty="curriculum", curriculum_reward_history=history)
    assert state["difficulty"] == "easy"


def test_env_curriculum_resolved_difficulty_visible_to_agent():
    """Agent observation never shows 'curriculum' — only the resolved level."""
    env = MedRecordAuditEnv()
    state = env.reset(difficulty="curriculum", curriculum_reward_history=[0.5] * 10)
    assert state["difficulty"] != "curriculum"
    assert state["difficulty"] in ("easy", "medium", "hard")


def test_env_curriculum_picks_random_case():
    """Curriculum mode ignores case_id from caller and picks at random."""
    env = MedRecordAuditEnv()
    # Pin to easy via low history; case_id arg should be ignored
    state = env.reset(
        difficulty="curriculum",
        case_id="medium_001",
        curriculum_reward_history=[0.1] * 5,
    )
    # Resolved to easy regardless of caller-supplied medium_001
    assert state["difficulty"] == "easy"
    assert state["case_id"].startswith("easy_")


def test_env_curriculum_threshold_boundaries():
    """Boundary checks just above each threshold (avoids float-precision edges)."""
    env = MedRecordAuditEnv()
    # Just above easy → medium boundary (0.35)
    state = env.reset(difficulty="curriculum", curriculum_reward_history=[0.36] * 10)
    assert state["difficulty"] == "medium"
    # Just above medium → hard boundary (0.55)
    state = env.reset(difficulty="curriculum", curriculum_reward_history=[0.56] * 10)
    assert state["difficulty"] == "hard"
    # Just below easy → medium boundary
    state = env.reset(difficulty="curriculum", curriculum_reward_history=[0.34] * 10)
    assert state["difficulty"] == "easy"
    # Just below medium → hard boundary
    state = env.reset(difficulty="curriculum", curriculum_reward_history=[0.54] * 10)
    assert state["difficulty"] == "medium"


def test_env_curriculum_full_episode_works():
    """Curriculum mode is fully compatible with a complete episode flow."""
    env = MedRecordAuditEnv()
    state = env.reset(difficulty="curriculum", curriculum_reward_history=[0.0] * 5)
    assert state["difficulty"] == "easy"
    # Run a partial episode
    env.step({"action": "read_record", "record_id": 1})
    env.step({
        "action": "flag_issue",
        "type": "allergy_violation",
        "description": "test",
        "evidence": [1],
    })
    result = env.step({"action": "submit_report"})
    assert result["done"] is True
    assert "rubric_breakdown" in result["info"]


def test_env_invalid_difficulty_still_rejected():
    """Phase 4 must not break the existing invalid-difficulty validation."""
    env = MedRecordAuditEnv()
    with pytest.raises(ValueError):
        env.reset(difficulty="impossible")
