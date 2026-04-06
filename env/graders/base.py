"""
Base grader logic shared by all individual case graders.
"""

from env.environment import MedRecordAuditEnv


def run_case(difficulty: str, case_id: str) -> dict:
    """
    Run a grading episode for a specific case.
    Just resets and immediately submits — pure validation that
    the environment loads correctly and returns score 0.0.

    Returns:
        dict with score, validation status, and episode stats.
    """
    env = MedRecordAuditEnv()
    state = env.reset(difficulty=difficulty, case_id=case_id)

    # Validate reset returned proper state
    assert "patient" in state, "reset() must return patient info"
    assert "record_index" in state, "reset() must return record_index"
    assert "budget_remaining" in state, "reset() must return budget_remaining"
    assert len(state["record_index"]) > 0, "record_index must not be empty"

    # Just validate environment loaded correctly — no actions taken
    # All counters should be 0 before inference runs

    return {
        "difficulty": difficulty,
        "case_id": case_id,
        "score": 0.0,
        "score_valid": True,
        "findings_submitted": 0,
        "records_reviewed": 0,
        "steps_taken": 0,
        "ground_truth_count": 0,
        "records_available": len(state["record_index"]),
        "budget": state["budget_remaining"],
        "passed": True,
    }
