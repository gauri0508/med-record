"""
Base grader logic shared by all individual case graders.

The grader validates that the environment works correctly by playing
through a basic episode: read a few records, flag a plausible issue
based on record content, and submit. No access to ground truth.
"""

from env.environment import MedRecordAuditEnv


def run_case(difficulty: str, case_id: str) -> dict:
    """
    Run a grading episode for a specific case.
    Plays through like a basic agent — reads prescriptions and labs,
    flags a plausible issue based on what it finds, and submits.

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

    record_index = state["record_index"]

    # Prioritize: prescriptions > labs > visit notes
    prescriptions = [r for r in record_index if r["type"] == "prescription"]
    labs = [r for r in record_index if r["type"] == "lab_result"]
    visits = [r for r in record_index if r["type"] == "visit_note"]
    priority = prescriptions + labs + visits

    # Read a few priority records (leave budget for flagging + submit)
    max_reads = min(len(priority), env.budget - 3)
    read_contents = []
    for r in priority[:max_reads]:
        if env.budget <= 3:
            break
        result = env.step({"action": "read_record", "record_id": r["id"]})
        if result["done"]:
            break
        record = result["info"].get("record", {})
        read_contents.append(record)

    # Flag a plausible issue based on what we read
    if not env.done and env.budget > 1:
        # Build a description from what we actually saw
        drug_names = []
        for rec in read_contents:
            if rec.get("type") == "prescription":
                drug_names.append(rec.get("drug", "unknown"))

        if len(drug_names) >= 2:
            env.step({
                "action": "flag_issue",
                "type": "drug_interaction",
                "description": f"Potential interaction between {drug_names[0]} and {drug_names[1]}",
                "evidence": [read_contents[0].get("id", 1), read_contents[1].get("id", 2)],
            })
        elif drug_names:
            env.step({
                "action": "flag_issue",
                "type": "drug_contraindication",
                "description": f"Review needed for {drug_names[0]} given patient conditions",
                "evidence": [read_contents[0].get("id", 1)],
            })
        else:
            env.step({
                "action": "flag_issue",
                "type": "declining_trend",
                "description": "Lab values show concerning trend requiring review",
                "evidence": [read_contents[0].get("id", 1)] if read_contents else [1],
            })

    # Submit report
    if not env.done:
        result = env.step({"action": "submit_report"})
        score = result["info"].get("final_score", 0.01)
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
