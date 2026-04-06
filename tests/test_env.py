"""
Tests for MedRecordAudit environment.
Run with: py -m pytest tests/ -v
"""

import pytest
from env.environment import MedRecordAuditEnv


# ============================================================
# SETUP
# ============================================================

ALL_CASES = [
    ("easy", "easy_001"),
    ("easy", "easy_002"),
    ("easy", "easy_003"),
    ("medium", "medium_001"),
    ("medium", "medium_002"),
    ("medium", "medium_003"),
    ("hard", "hard_001"),
    ("hard", "hard_002"),
    ("hard", "hard_003"),
]


@pytest.fixture
def env():
    return MedRecordAuditEnv()


# ============================================================
# TEST: Environment loads all 9 cases
# ============================================================

@pytest.mark.parametrize("difficulty,case_id", ALL_CASES)
def test_reset_loads_case(env, difficulty, case_id):
    state = env.reset(difficulty=difficulty, case_id=case_id)
    assert state["case_id"] == case_id
    assert state["difficulty"] == difficulty
    assert "patient" in state
    assert "record_index" in state
    assert "task" in state
    assert len(state["record_index"]) > 0
    assert state["budget_remaining"] > 0
    assert state["steps_taken"] == 0
    assert state["done"] == False


# ============================================================
# TEST: Task instructions load for each case
# ============================================================

@pytest.mark.parametrize("difficulty,case_id", ALL_CASES)
def test_task_loaded(env, difficulty, case_id):
    state = env.reset(difficulty=difficulty, case_id=case_id)
    task = state["task"]
    assert task["task_id"] == case_id
    assert "title" in task
    assert "instruction" in task
    assert len(task["instruction"]) > 20
    assert "focus_areas" in task
    assert "expected_findings" in task
    assert task["expected_findings"] > 0


# ============================================================
# TEST: Correct budgets per difficulty
# ============================================================

def test_easy_budget(env):
    state = env.reset("easy", "easy_001")
    assert state["budget_remaining"] == 15

def test_medium_budget(env):
    state = env.reset("medium", "medium_001")
    assert state["budget_remaining"] == 25

def test_hard_budget(env):
    state = env.reset("hard", "hard_001")
    assert state["budget_remaining"] == 30


# ============================================================
# TEST: Correct record counts per difficulty
# ============================================================

def test_easy_record_count(env):
    state = env.reset("easy", "easy_001")
    assert state["records_available"] == 20

def test_medium_record_count(env):
    state = env.reset("medium", "medium_001")
    assert state["records_available"] == 80

def test_hard_record_count(env):
    state = env.reset("hard", "hard_001")
    assert state["records_available"] == 150


# ============================================================
# TEST: read_record action
# ============================================================

def test_read_record(env):
    env.reset("easy", "easy_001")
    result = env.step({"action": "read_record", "record_id": 1})
    assert result["done"] == False
    assert "record" in result["info"]
    assert result["info"]["record"]["id"] == 1
    assert result["state"]["budget_remaining"] == 14
    assert 1 in result["state"]["records_reviewed"]


def test_read_record_invalid_id(env):
    env.reset("easy", "easy_001")
    result = env.step({"action": "read_record", "record_id": 999})
    assert "error" in result["info"]


# ============================================================
# TEST: cross_reference action
# ============================================================

def test_cross_reference_drug(env):
    env.reset("easy", "easy_001")
    result = env.step({"action": "cross_reference", "query": "warfarin"})
    assert result["done"] == False
    assert "results" in result["info"]
    assert len(result["info"]["results"]["drugs"]) > 0


def test_cross_reference_disease(env):
    env.reset("easy", "easy_001")
    result = env.step({"action": "cross_reference", "query": "diabetes"})
    assert result["done"] == False


def test_cross_reference_empty(env):
    env.reset("easy", "easy_001")
    result = env.step({"action": "cross_reference", "query": "xyznonexistent"})
    assert result["done"] == False


# ============================================================
# TEST: flag_issue action
# ============================================================

def test_flag_issue(env):
    env.reset("easy", "easy_001")
    result = env.step({
        "action": "flag_issue",
        "type": "allergy_violation",
        "description": "test finding",
        "evidence": [1, 2],
    })
    assert result["done"] == False
    assert result["info"]["finding_number"] == 1
    assert len(result["state"]["findings"]) == 1


def test_flag_issue_invalid_type(env):
    env.reset("easy", "easy_001")
    result = env.step({
        "action": "flag_issue",
        "type": "invalid_type",
        "description": "test",
        "evidence": [],
    })
    assert "error" in result["info"]


def test_flag_issue_missing_description(env):
    env.reset("easy", "easy_001")
    result = env.step({
        "action": "flag_issue",
        "type": "drug_interaction",
        "description": "",
        "evidence": [],
    })
    assert "error" in result["info"]


# ============================================================
# TEST: submit_report action
# ============================================================

def test_submit_report_no_findings(env):
    env.reset("easy", "easy_001")
    result = env.step({"action": "submit_report"})
    assert result["done"] == True
    assert result["info"]["final_score"] == 0.0
    assert result["info"]["findings_submitted"] == 0


def test_submit_report_with_correct_finding(env):
    env.reset("easy", "easy_001")
    env.step({
        "action": "flag_issue",
        "type": "allergy_violation",
        "description": "Amoxicillin prescribed despite documented penicillin allergy with anaphylaxis",
        "evidence": [1, 2, 17, 18],
    })
    result = env.step({"action": "submit_report"})
    assert result["done"] == True
    assert result["info"]["final_score"] > 0.0
    assert result["info"]["findings_submitted"] == 1


def test_submit_report_with_wrong_finding(env):
    env.reset("easy", "easy_001")
    env.step({
        "action": "flag_issue",
        "type": "drug_interaction",
        "description": "completely wrong finding about nothing",
        "evidence": [5, 6],
    })
    result = env.step({"action": "submit_report"})
    assert result["done"] == True
    # Wrong finding should score lower than correct finding
    assert result["info"]["final_score"] < 0.3


# ============================================================
# TEST: Score is always 0.0 - 1.0
# ============================================================

@pytest.mark.parametrize("difficulty,case_id", ALL_CASES)
def test_score_in_range(env, difficulty, case_id):
    env.reset(difficulty=difficulty, case_id=case_id)
    result = env.step({"action": "submit_report"})
    score = result["info"]["final_score"]
    assert 0.0 <= score <= 1.0


# ============================================================
# TEST: Budget decrements correctly
# ============================================================

def test_budget_decrements(env):
    state = env.reset("easy", "easy_001")
    initial_budget = state["budget_remaining"]

    env.step({"action": "read_record", "record_id": 1})
    env.step({"action": "cross_reference", "query": "warfarin"})
    env.step({"action": "flag_issue", "type": "drug_interaction", "description": "test", "evidence": [1]})

    current_state = env.state()
    assert current_state["budget_remaining"] == initial_budget - 3
    assert current_state["steps_taken"] == 3


# ============================================================
# TEST: Episode ends when budget exhausted
# ============================================================

def test_budget_exhaustion(env):
    env.reset("easy", "easy_001")
    # Burn all 15 steps reading records
    for i in range(1, 16):
        result = env.step({"action": "read_record", "record_id": min(i, 20)})
    # Budget is now 0 — next action triggers auto-end
    result = env.step({"action": "read_record", "record_id": 1})
    assert result["done"] == True


# ============================================================
# TEST: Cannot act after episode ends
# ============================================================

def test_no_action_after_done(env):
    env.reset("easy", "easy_001")
    env.step({"action": "submit_report"})
    result = env.step({"action": "read_record", "record_id": 1})
    assert result["done"] == True
    assert "error" in result["info"]


# ============================================================
# TEST: Invalid action
# ============================================================

def test_invalid_action(env):
    env.reset("easy", "easy_001")
    result = env.step({"action": "fly_to_moon"})
    assert "error" in result["info"]
    assert result["state"]["budget_remaining"] == 15  # no budget consumed


# ============================================================
# TEST: Invalid difficulty
# ============================================================

def test_invalid_difficulty(env):
    with pytest.raises(ValueError):
        env.reset("impossible", "easy_001")


# ============================================================
# TEST: Ground truth never exposed in state
# ============================================================

@pytest.mark.parametrize("difficulty,case_id", ALL_CASES)
def test_ground_truth_hidden(env, difficulty, case_id):
    state = env.reset(difficulty=difficulty, case_id=case_id)
    state_str = str(state)
    assert "ground_truth" not in state_str
    assert "what_should_have_happened" not in state_str
    assert "clinical_reference" not in state_str
