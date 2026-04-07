"""
Base grader logic shared by all individual case graders.
"""

from env.environment import MedRecordAuditEnv


def run_case(difficulty: str, case_id: str) -> dict:
    """
    Run a grading episode for a specific case.
    Resets and immediately submits — validates the environment
    loads correctly and returns a score strictly in (0, 1).

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

    # Submit immediately — score will be 0.01 (clamped minimum)
    result = env.step({"action": "submit_report"})
    score = result["info"].get("final_score", 0.01)

    return {
        "difficulty": difficulty,
        "case_id": case_id,
        "score": score,
        "score_valid": 0.0 < score < 1.0,
        "findings_submitted": 0,
        "records_reviewed": 0,
        "steps_taken": 1,
        "ground_truth_count": result["info"].get("ground_truth_count", 0),
        "records_available": len(state["record_index"]),
        "budget": state["budget_remaining"],
        "passed": 0.0 < score < 1.0,
    }
