"""
Run a specific task and grade it.

Usage:
    py run_task.py easy_001
    py run_task.py medium_002
    py run_task.py hard_003
    py run_task.py all          # run all 9 tasks
"""

import sys
import json
from env.environment import MedRecordAuditEnv


ALL_TASKS = [
    "easy_001", "easy_002", "easy_003",
    "medium_001", "medium_002", "medium_003",
    "hard_001", "hard_002", "hard_003",
]


def run_task(task_id: str):
    """Run a task: load case, simulate a basic agent, grade it."""
    difficulty = task_id.split("_")[0]

    env = MedRecordAuditEnv()
    state = env.reset(difficulty=difficulty, case_id=task_id)

    task = state["task"]
    patient = state["patient"]
    record_index = state["record_index"]
    budget = state["budget_remaining"]

    print(f"\n{'='*60}")
    print(f"TASK: {task['title']}")
    print(f"CASE: {task_id} | DIFFICULTY: {difficulty}")
    print(f"{'='*60}")
    print(f"\nPATIENT: {patient['age']}{patient['gender']}")
    print(f"CONDITIONS: {', '.join(patient.get('known_conditions', []))}")
    print(f"MEDICATIONS: {', '.join(patient.get('current_medications', []))}")
    print(f"ALLERGIES: {', '.join(patient.get('allergies', [])) or 'None'}")
    print(f"\nINSTRUCTION: {task['instruction']}")
    print(f"FOCUS: {', '.join(task.get('focus_areas', []))}")
    print(f"EXPECTED FINDINGS: {task.get('expected_findings', '?')}")
    print(f"RECORDS: {len(record_index)} | BUDGET: {budget} steps")

    # --- Simple agent: read prescriptions + labs, then flag based on patterns ---
    print(f"\n--- RUNNING AGENT ---")

    # Read all prescriptions first
    prescriptions = [r for r in record_index if r["type"] == "prescription"]
    labs = [r for r in record_index if r["type"] == "lab_result"]
    allergy_records = [r for r in record_index if r["type"] == "allergy_record"]
    visit_notes = [r for r in record_index if r["type"] == "visit_note"]

    read_budget = max(1, budget - task.get("expected_findings", 1) - 1)
    records_to_read = []

    # Priority: allergy records > prescriptions > labs > visits
    for r in allergy_records:
        if len(records_to_read) < read_budget:
            records_to_read.append(r["id"])
    for r in prescriptions:
        if len(records_to_read) < read_budget:
            records_to_read.append(r["id"])
    for r in labs:
        if len(records_to_read) < read_budget:
            records_to_read.append(r["id"])
    for r in visit_notes:
        if len(records_to_read) < read_budget:
            records_to_read.append(r["id"])

    # Read records
    read_contents = []
    for rid in records_to_read:
        result = env.step({"action": "read_record", "record_id": rid})
        if result["done"]:
            break
        record = result["info"].get("record", {})
        read_contents.append(record)
        print(f"  [READ] Record #{rid}: {record.get('type', '?')} — {record.get('summary', '?')}")

    print(f"\n  Records read: {len(read_contents)}")
    print(f"  Budget remaining: {env.budget}")

    # Submit report (no findings from basic agent — score will be 0)
    result = env.step({"action": "submit_report"})
    info = result["info"]

    print(f"\n--- RESULTS ---")
    print(f"  Score: {info.get('final_score', 0.0)}")
    print(f"  Findings submitted: {info.get('findings_submitted', 0)}")
    print(f"  Records reviewed: {info.get('records_reviewed', 0)}")
    print(f"  Steps taken: {info.get('steps_taken', 0)}")
    print(f"  Ground truth issues: {info.get('ground_truth_count', 0)}")

    # Show what the agent SHOULD have found
    print(f"\n--- ANSWER KEY (ground truth) ---")
    for i, issue in enumerate(env.ground_truth, 1):
        print(f"  {i}. [{issue['severity'].upper()}] {issue['type']}")
        print(f"     {issue['description'][:120]}...")
        print(f"     Evidence records: {issue['evidence_records']}")
    print()

    return info.get("final_score", 0.0)


def main():
    if len(sys.argv) < 2:
        print("Usage: py run_task.py <task_id>")
        print("       py run_task.py easy_001")
        print("       py run_task.py all")
        print(f"\nAvailable tasks: {', '.join(ALL_TASKS)}")
        sys.exit(1)

    task_id = sys.argv[1]

    if task_id == "all":
        scores = {}
        for tid in ALL_TASKS:
            scores[tid] = run_task(tid)

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for tid, score in scores.items():
            print(f"  {tid}: {score}")
    else:
        if task_id not in ALL_TASKS:
            print(f"Unknown task: {task_id}")
            print(f"Available: {', '.join(ALL_TASKS)}")
            sys.exit(1)
        run_task(task_id)


if __name__ == "__main__":
    main()
