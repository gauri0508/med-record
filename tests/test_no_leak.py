"""
Tests for ground-truth privacy and per-step reward signal correctness (Phase 3).

The agent learns from rewards but must never see ground-truth identifiers.
Per-step rewards encode "this was a useful action" as a numeric signal only —
the agent cannot read which records are evidence or which type of issue exists.

Run with: py -m pytest tests/test_no_leak.py -v
"""

import pytest
from env.environment import MedRecordAuditEnv


ALL_CASES = [
    ("easy", "easy_001"), ("easy", "easy_002"), ("easy", "easy_003"),
    ("medium", "medium_001"), ("medium", "medium_002"), ("medium", "medium_003"),
    ("hard", "hard_001"), ("hard", "hard_002"), ("hard", "hard_003"),
]

# Strict forbidden keys — must never appear in pre-submit observations.
# Includes the bare "ground_truth" prefix; the agent should not even see counts.
FORBIDDEN_KEYS_STRICT = [
    "ground_truth",
    "ground_truth_issues",
    "evidence_records",
    "what_should_have_happened",
    "clinical_reference",
]

# Submit-time forbidden keys — at submit, the agent legitimately sees
# `ground_truth_count` (just an integer of how many issues existed) so it
# knows recall. The full data (issues, descriptions, evidence_records) must
# still be hidden.
FORBIDDEN_KEYS_AT_SUBMIT = [
    "ground_truth_issues",
    "evidence_records",
    "what_should_have_happened",
    "clinical_reference",
]


# ============================================================
# GROUND TRUTH PRIVACY: state observation
# ============================================================

@pytest.mark.parametrize("difficulty,case_id", ALL_CASES)
def test_ground_truth_not_in_initial_state(difficulty, case_id):
    """reset() returns state — strict ground-truth privacy required."""
    env = MedRecordAuditEnv()
    state = env.reset(difficulty=difficulty, case_id=case_id)
    state_str = str(state)
    for key in FORBIDDEN_KEYS_STRICT:
        assert key not in state_str, f"Forbidden key '{key}' leaked in initial state for {case_id}"


@pytest.mark.parametrize("difficulty,case_id", ALL_CASES)
def test_ground_truth_not_in_step_info(difficulty, case_id):
    """step() info dict — strict ground-truth privacy required during episode."""
    env = MedRecordAuditEnv()
    env.reset(difficulty=difficulty, case_id=case_id)
    for action in [
        {"action": "read_record", "record_id": 1},
        {"action": "read_record", "record_id": 2},
        {"action": "cross_reference", "query": "warfarin"},
        {"action": "cross_reference", "query": "diabetes"},
    ]:
        result = env.step(action)
        info_str = str(result["info"])
        for key in FORBIDDEN_KEYS_STRICT:
            assert key not in info_str, \
                f"Forbidden key '{key}' leaked in step info for {case_id} (action={action['action']})"


@pytest.mark.parametrize("difficulty,case_id", ALL_CASES)
def test_ground_truth_data_not_in_submit_info(difficulty, case_id):
    """
    submit_report info — must not contain raw GT data (issues list, descriptions,
    evidence_records, etc). Note: a benign integer `ground_truth_count` IS exposed
    so the agent learns recall — this is intentional scoring feedback, not a leak.
    """
    env = MedRecordAuditEnv()
    env.reset(difficulty=difficulty, case_id=case_id)
    result = env.step({"action": "submit_report"})
    info_str = str(result["info"])
    for key in FORBIDDEN_KEYS_AT_SUBMIT:
        assert key not in info_str, \
            f"Forbidden key '{key}' leaked in submit info for {case_id}"


@pytest.mark.parametrize("difficulty,case_id", ALL_CASES)
def test_submit_info_only_exposes_count_not_data(difficulty, case_id):
    """At submit, the only allowed GT-derived field is the integer ground_truth_count."""
    env = MedRecordAuditEnv()
    env.reset(difficulty=difficulty, case_id=case_id)
    result = env.step({"action": "submit_report"})
    info = result["info"]
    if "ground_truth_count" in info:
        assert isinstance(info["ground_truth_count"], int)
        assert info["ground_truth_count"] >= 0


# ============================================================
# PER-STEP REWARD: signal correctness
# ============================================================

def test_step_reward_for_gt_record_read():
    """Reading a known ground-truth evidence record returns step_reward > 0."""
    env = MedRecordAuditEnv()
    env.reset("easy", "easy_001")
    # easy_001 ground truth evidence is records [1, 2, 17, 18]
    result = env.step({"action": "read_record", "record_id": 1})
    assert "step_reward" in result["info"], "GT record read should attach step_reward to info"
    assert result["info"]["step_reward"] > 0, f"step_reward should be > 0, got {result['info']['step_reward']}"


def test_step_reward_zero_for_non_gt_record():
    """Reading a non-evidence record returns step_reward == 0.0 in info."""
    env = MedRecordAuditEnv()
    env.reset("easy", "easy_001")
    # Record 5 is not in easy_001's GT evidence (which is [1, 2, 17, 18])
    result = env.step({"action": "read_record", "record_id": 5})
    assert "step_reward" in result["info"], "step_reward key should always be present"
    assert result["info"]["step_reward"] == 0.0


def test_step_reward_first_read_only_no_farming():
    """Re-reading the same GT record yields step_reward == 0.0 (anti-farming)."""
    env = MedRecordAuditEnv()
    env.reset("easy", "easy_001")
    # First read — gets the bonus
    first = env.step({"action": "read_record", "record_id": 1})
    assert first["info"]["step_reward"] > 0
    # Second read of same record — zero (anti-farming)
    second = env.step({"action": "read_record", "record_id": 1})
    assert second["info"]["step_reward"] == 0.0, \
        "Re-read of same GT record must yield step_reward == 0.0 (anti-farming)"


def test_step_reward_for_cross_reference_match():
    """A cross_reference query that matches GT description gets a small step_reward."""
    env = MedRecordAuditEnv()
    env.reset("easy", "easy_001")
    # easy_001 GT mentions amoxicillin / penicillin allergy
    result = env.step({"action": "cross_reference", "query": "amoxicillin"})
    # Either matches (step_reward > 0) or doesn't appear (no key) — both valid;
    # this case's GT description should contain "amoxicillin" so we expect a hit
    assert "step_reward" in result["info"], \
        "Query matching GT description should attach step_reward"
    assert result["info"]["step_reward"] > 0


def test_step_reward_zero_for_irrelevant_query():
    """A cross_reference query unrelated to GT yields step_reward == 0.0."""
    env = MedRecordAuditEnv()
    env.reset("easy", "easy_001")
    result = env.step({"action": "cross_reference", "query": "asparagus"})
    assert result["info"]["step_reward"] == 0.0


def test_step_reward_for_successful_flag():
    """A successful flag_issue gets a tiny step_reward attempt bonus."""
    env = MedRecordAuditEnv()
    env.reset("easy", "easy_001")
    env.step({"action": "read_record", "record_id": 1})
    result = env.step({
        "action": "flag_issue",
        "type": "drug_interaction",
        "description": "test description",
        "evidence": [1],
    })
    assert "step_reward" in result["info"]
    assert result["info"]["step_reward"] > 0


def test_step_reward_short_query_no_match():
    """1-2 char queries are ignored — step_reward == 0.0."""
    env = MedRecordAuditEnv()
    env.reset("easy", "easy_001")
    result = env.step({"action": "cross_reference", "query": "a"})
    assert result["info"]["step_reward"] == 0.0


# ============================================================
# SPEC-EXACT: test_step_reward_signal_only (MASTER_PLAN.md line 800)
# ============================================================

@pytest.mark.parametrize("difficulty,case_id", ALL_CASES)
def test_step_reward_signal_only(difficulty, case_id):
    """Step reward may be positive (useful read) but info must not say WHY.

    Spec match for MASTER_PLAN.md Phase 3, lines 800-810:
      - if step_reward in info, it must be a float
      - 'evidence_records' must NOT appear in info string
    """
    env = MedRecordAuditEnv()
    env.reset(difficulty=difficulty, case_id=case_id)
    result = env.step({"action": "read_record", "record_id": 1})
    if "step_reward" in result["info"]:
        assert isinstance(result["info"]["step_reward"], float)
    info_str = str(result["info"])
    assert "evidence_records" not in info_str


# ============================================================
# CUMULATIVE TRAJECTORY REWARD: tracks across the episode
# ============================================================

def test_cumulative_step_reward_accumulates():
    """_cumulative_step_reward sums up across the episode."""
    env = MedRecordAuditEnv()
    env.reset("easy", "easy_001")
    assert env._cumulative_step_reward == 0.0

    # Read a GT record (1 is in evidence)
    env.step({"action": "read_record", "record_id": 1})
    after_first = env._cumulative_step_reward
    assert after_first > 0

    # Read another GT record (2 is in evidence) — total goes up
    env.step({"action": "read_record", "record_id": 2})
    after_second = env._cumulative_step_reward
    assert after_second > after_first


def test_cumulative_step_reward_resets_on_reset():
    """A new reset() zeroes the cumulative trajectory reward."""
    env = MedRecordAuditEnv()
    env.reset("easy", "easy_001")
    env.step({"action": "read_record", "record_id": 1})
    assert env._cumulative_step_reward > 0

    env.reset("easy", "easy_002")
    assert env._cumulative_step_reward == 0.0


# ============================================================
# REWARD VALUE: validator-safe range
# ============================================================

def test_step_reward_field_in_validator_range():
    """The reward field of step() output is always in (0, 1) — validator requirement."""
    env = MedRecordAuditEnv()
    env.reset("easy", "easy_001")
    for action in [
        {"action": "read_record", "record_id": 1},     # GT — positive
        {"action": "read_record", "record_id": 5},     # non-GT — neutral
        {"action": "cross_reference", "query": "warfarin"},
        {"action": "cross_reference", "query": "asparagus"},
    ]:
        result = env.step(action)
        assert 0.0 < result["reward"] < 1.0, \
            f"reward={result['reward']} out of (0,1) for {action}"
