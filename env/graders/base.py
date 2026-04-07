"""
Base grader logic shared by all individual case graders.
"""

from env.environment import MedRecordAuditEnv


def run_case(difficulty: str, case_id: str) -> dict:
    """
    Run a grading episode for a specific case.
    Resets, reads evidence records, flags ground truth issues,
    and submits a report to produce a valid score in (0, 1).

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

    # Read first evidence record from each ground truth issue
    read_ids = set()
    for issue in env.ground_truth:
        for rid in issue.get("evidence_records", [])[:1]:
            if rid in env.records and rid not in read_ids:
                env.step({"action": "read_record", "record_id": rid})
                read_ids.add(rid)
                if env.done:
                    break
        if env.done:
            break

    # Flag each ground truth issue
    if not env.done:
        for issue in env.ground_truth:
            if env.budget <= 1:
                break
            env.step({
                "action": "flag_issue",
                "type": issue["type"],
                "description": issue["description"],
                "evidence": issue.get("evidence_records", []),
            })

    # Submit report
    if not env.done:
        result = env.step({"action": "submit_report"})
        score = result["info"].get("final_score", 0.0)
    else:
        score = env.reward

    # Validate score is in (0, 1)
    score_valid = 0.0 < score < 1.0

    return {
        "difficulty": difficulty,
        "case_id": case_id,
        "score": score,
        "score_valid": score_valid,
        "findings_submitted": len(env.findings),
        "records_reviewed": len(env.reviewed_records),
        "steps_taken": env.steps_taken,
        "ground_truth_count": len(env.ground_truth),
        "records_available": len(state["record_index"]),
        "budget": state["budget_remaining"],
        "passed": score_valid,
    }
