"""
Tests for anti-reward-hacking guards in MedRecordAuditEnv (Phase 2).

Each guard rejects or warns about a specific exploit pattern:
  - Duplicate flag rejection
  - Description length cap (500 chars)
  - Evidence ID existence check
  - Unread evidence warning (soft signal — flag still accepted)
  - State deep-copy (agent cannot mutate internal env state via observation)
  - Hard budget cap (no extension path)

Run with: py -m pytest tests/test_antihack.py -v
"""

import pytest
from env.environment import MedRecordAuditEnv


@pytest.fixture
def env():
    e = MedRecordAuditEnv()
    e.reset("easy", "easy_001")
    return e


# ============================================================
# GUARD: Duplicate flag rejection
# ============================================================

def test_duplicate_flag_rejected(env):
    """Same (type, description prefix) cannot be flagged twice."""
    env.step({"action": "read_record", "record_id": 1})
    first = env.step({
        "action": "flag_issue",
        "type": "drug_interaction",
        "description": "warfarin and aspirin interaction causing bleeding risk",
        "evidence": [1],
    })
    assert "error" not in first["info"], "First flag should succeed"

    duplicate = env.step({
        "action": "flag_issue",
        "type": "drug_interaction",
        "description": "warfarin and aspirin interaction causing bleeding risk",
        "evidence": [1],
    })
    assert "error" in duplicate["info"]
    assert "duplicate" in duplicate["info"]["error"].lower()
    # Only one finding should be stored
    assert len(env.findings) == 1


def test_duplicate_flag_different_type_allowed(env):
    """Same description but different type IS allowed (legitimate cross-type concern)."""
    env.step({"action": "read_record", "record_id": 1})
    env.step({
        "action": "flag_issue",
        "type": "drug_interaction",
        "description": "warfarin causing bleeding risk in elderly patient",
        "evidence": [1],
    })
    second = env.step({
        "action": "flag_issue",
        "type": "missed_monitoring",
        "description": "warfarin causing bleeding risk in elderly patient",
        "evidence": [1],
    })
    assert "error" not in second["info"]
    assert len(env.findings) == 2


# ============================================================
# GUARD: Description length cap (500 chars)
# ============================================================

def test_description_too_long_rejected(env):
    """Descriptions over 500 chars are rejected to prevent keyword stuffing."""
    long_desc = "x" * 501
    result = env.step({
        "action": "flag_issue",
        "type": "drug_interaction",
        "description": long_desc,
        "evidence": [1],
    })
    assert "error" in result["info"]
    assert "too long" in result["info"]["error"].lower()
    assert len(env.findings) == 0


def test_description_exactly_500_accepted(env):
    """Boundary check: exactly 500 chars is accepted."""
    desc = "a" * 500
    result = env.step({
        "action": "flag_issue",
        "type": "drug_interaction",
        "description": desc,
        "evidence": [1],
    })
    assert "error" not in result["info"]


# ============================================================
# GUARD: Evidence ID existence
# ============================================================

def test_invalid_evidence_ids_rejected(env):
    """Evidence IDs that don't exist in the case are rejected."""
    result = env.step({
        "action": "flag_issue",
        "type": "allergy_violation",
        "description": "penicillin allergy violated by amoxicillin prescription",
        "evidence": [9999, 8888],
    })
    assert "error" in result["info"]
    assert "do not exist" in result["info"]["error"].lower()
    assert len(env.findings) == 0


def test_partially_invalid_evidence_rejected(env):
    """Even one invalid ID rejects the whole flag (no partial acceptance)."""
    result = env.step({
        "action": "flag_issue",
        "type": "allergy_violation",
        "description": "valid evidence mixed with invalid",
        "evidence": [1, 2, 9999],
    })
    assert "error" in result["info"]
    assert "9999" in result["info"]["error"]


# ============================================================
# GUARD: Unread evidence warning (soft)
# ============================================================

def test_unread_evidence_warns(env):
    """Citing >50% unread records issues a warning but allows the flag."""
    # Don't read any records — cite IDs that exist but were not opened
    result = env.step({
        "action": "flag_issue",
        "type": "allergy_violation",
        "description": "penicillin allergy violated by amoxicillin prescription",
        "evidence": [1, 2, 3, 4],
    })
    # Soft guard: flag accepted, warning attached
    assert "error" not in result["info"]
    assert "warning" in result["info"]
    assert "not read" in result["info"]["warning"].lower()


def test_majority_read_no_warning(env):
    """If majority of cited records were read, no warning."""
    env.step({"action": "read_record", "record_id": 1})
    env.step({"action": "read_record", "record_id": 2})
    env.step({"action": "read_record", "record_id": 3})
    # 3 of 4 read - 75% read, below the >50% unread threshold
    result = env.step({
        "action": "flag_issue",
        "type": "allergy_violation",
        "description": "penicillin allergy violated by amoxicillin prescription",
        "evidence": [1, 2, 3, 4],
    })
    assert "error" not in result["info"]
    assert "warning" not in result["info"]


# ============================================================
# GUARD: State deep copy
# ============================================================

def test_state_findings_is_deep_copy(env):
    """Mutating findings list in returned state does not affect env."""
    env.step({"action": "read_record", "record_id": 1})
    env.step({
        "action": "flag_issue",
        "type": "drug_interaction",
        "description": "test finding",
        "evidence": [1],
    })

    state = env.state()
    state["findings"].append({"type": "fake", "description": "injected", "evidence": []})
    state["findings"][0]["description"] = "MUTATED"

    fresh_state = env.state()
    assert len(fresh_state["findings"]) == 1, "External mutation should not affect env"
    assert fresh_state["findings"][0]["description"] == "test finding"


def test_state_records_reviewed_is_copy(env):
    """Mutating records_reviewed list in returned state does not affect env."""
    env.step({"action": "read_record", "record_id": 1})
    state = env.state()
    state["records_reviewed"].append(999)
    fresh_state = env.state()
    assert 999 not in fresh_state["records_reviewed"]


def test_state_record_index_is_deep_copy(env):
    """Mutating record_index in state does not affect env's record summaries."""
    state = env.state()
    if state["record_index"]:
        state["record_index"][0]["summary"] = "TAMPERED"
    fresh_state = env.state()
    if fresh_state["record_index"]:
        assert fresh_state["record_index"][0]["summary"] != "TAMPERED"


# ============================================================
# GUARD: Hard budget cap (cannot be extended)
# ============================================================

def test_budget_cannot_be_extended_via_action(env):
    """No action extends the budget. Confirm budget only decrements."""
    initial_budget = env.budget
    # Try every action type — none should increase budget
    env.step({"action": "read_record", "record_id": 1})
    env.step({"action": "cross_reference", "query": "warfarin"})
    env.step({"action": "flag_issue", "type": "drug_interaction", "description": "test", "evidence": [1]})
    assert env.budget < initial_budget


def test_budget_exhaustion_terminates_episode(env):
    """When budget hits 0, the episode auto-terminates on next action."""
    # Burn all 15 steps reading records
    for i in range(1, 16):
        env.step({"action": "read_record", "record_id": min(i, 20)})
    # One more action triggers auto-end
    result = env.step({"action": "read_record", "record_id": 1})
    assert result["done"] is True
