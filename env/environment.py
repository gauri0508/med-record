"""
MedRecordAudit — Core RL Environment

This is the "game engine" of the environment. It loads patient cases,
manages state, handles agent actions, and tracks the episode.

The agent NEVER sees ground_truth_issues — only the grader/reward module does.
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

# Base paths
DATA_DIR = Path(__file__).parent.parent / "data"
CASES_DIR = DATA_DIR / "cases"
TASKS_DIR = Path(__file__).parent.parent / "tasks"


class MedRecordAuditEnv:
    """
    RL Environment for medical record auditing.

    The agent receives a patient case with a record index (summaries only).
    It must strategically read records, cross-reference medical databases,
    flag issues, and submit a report — all within a limited step budget.
    """

    # Budget per difficulty level
    BUDGETS = {
        "easy": 15,
        "medium": 25,
        "hard": 30,
    }

    def __init__(self):
        # Load reference databases once
        self.drugs_db = self._load_json(DATA_DIR / "drugs.json")
        self.diseases_db = self._load_json(DATA_DIR / "diseases.json")
        self.lab_ranges_db = self._load_json(DATA_DIR / "lab_ranges.json")

        # Episode state (initialized on reset)
        self.task = None
        self.case = None
        self.patient = None
        self.records = None
        self.record_index = None
        self.ground_truth = None
        self.difficulty = None
        self.case_id = None
        self.budget = 0
        self.total_budget = 0
        self.steps_taken = 0
        self.reviewed_records = []
        self.findings = []
        self.done = False
        self.reward = 0.0

    def _load_json(self, path: Path) -> dict:
        """Load a JSON file and return its contents."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _list_cases(self, difficulty: str) -> list:
        """List all case files for a given difficulty."""
        case_dir = CASES_DIR / difficulty
        if not case_dir.exists():
            return []
        return sorted(case_dir.glob("case_*.json"))

    def reset(self, difficulty: str = "easy", case_id: Optional[str] = None) -> dict:
        """
        Start a new episode.

        Args:
            difficulty: "easy", "medium", or "hard"
            case_id: specific case ID (e.g., "easy_001") or None for random

        Returns:
            Initial state visible to the agent.
        """
        if difficulty not in self.BUDGETS:
            raise ValueError(f"Invalid difficulty: {difficulty}. Must be one of {list(self.BUDGETS.keys())}")

        self.difficulty = difficulty
        self.done = False
        self.reward = 0.0
        self.steps_taken = 0
        self.reviewed_records = []
        self.findings = []
        self.budget = self.BUDGETS[difficulty]
        self.total_budget = self.BUDGETS[difficulty]

        # Load case
        cases = self._list_cases(difficulty)
        if not cases:
            raise FileNotFoundError(f"No cases found for difficulty: {difficulty}")

        if case_id:
            # Find specific case
            target = None
            for c in cases:
                case_data = self._load_json(c)
                if case_data.get("case_id") == case_id:
                    target = c
                    break
            if target is None:
                raise ValueError(f"Case not found: {case_id}")
            case_path = target
        else:
            case_path = random.choice(cases)

        self.case = self._load_json(case_path)
        self.case_id = self.case["case_id"]
        self.patient = self.case["patient"]
        self.records = {r["id"]: r for r in self.case["records"]}

        # Load task instructions
        task_path = TASKS_DIR / self.case_id / "task.json"
        if task_path.exists():
            self.task = self._load_json(task_path)
        else:
            self.task = {
                "task_id": self.case_id,
                "difficulty": difficulty,
                "title": "Medical Record Audit",
                "instruction": "Audit this patient's medical records for safety issues including drug interactions, contraindications, allergy violations, declining lab trends, missed monitoring, and contradictions between providers.",
                "focus_areas": [],
                "expected_findings": 0,
            }
        self.ground_truth = self.case["ground_truth_issues"]

        # Build record index (visible to agent — summaries only, NOT full content)
        self.record_index = []
        for r in self.case["records"]:
            entry = {
                "id": r["id"],
                "date": r["date"],
                "type": r["type"],
                "summary": r.get("summary", ""),
            }
            # Add doctor/department for visit notes
            if r["type"] == "visit_note":
                entry["doctor"] = r.get("doctor", "")
                entry["department"] = r.get("department", "")
            # Add drug name for prescriptions
            elif r["type"] == "prescription":
                entry["drug"] = r.get("drug", "")
                entry["prescriber"] = r.get("prescriber", "")
            # Add ordered_by for lab results
            elif r["type"] == "lab_result":
                entry["ordered_by"] = r.get("ordered_by", "")
            self.record_index.append(entry)

        return self.state()

    def step(self, action: dict) -> dict:
        """
        Execute one action in the environment.

        Args:
            action: dict with "action" key and action-specific parameters:
                - {"action": "read_record", "record_id": int}
                - {"action": "cross_reference", "query": str}
                - {"action": "flag_issue", "type": str, "description": str, "evidence": list[int]}
                - {"action": "submit_report"}

        Returns:
            dict with keys: state, reward, done, info
        """
        if self.done:
            return {
                "state": self.state(),
                "reward": max(0.01, min(0.99, self.reward)),
                "done": True,
                "info": {"error": "Episode already ended. Call reset() to start a new one."},
            }

        if self.budget <= 0:
            # Budget exhausted — force end
            self.done = True
            self.reward = self._compute_reward()
            return {
                "state": self.state(),
                "reward": max(0.01, min(0.99, self.reward)),
                "done": True,
                "info": {"message": "Budget exhausted. Episode ended automatically."},
            }

        action_type = action.get("action", "")
        info = {}

        if action_type == "read_record":
            info = self._handle_read_record(action)
        elif action_type == "cross_reference":
            info = self._handle_cross_reference(action)
        elif action_type == "flag_issue":
            info = self._handle_flag_issue(action)
        elif action_type == "submit_report":
            info = self._handle_submit_report()
        else:
            info = {"error": f"Unknown action: {action_type}. Valid actions: read_record, cross_reference, flag_issue, submit_report"}
            # Don't consume budget for invalid actions
            return {
                "state": self.state(),
                "reward": 0.01,
                "done": False,
                "info": info,
            }

        # Clamp reward to (0.01, 0.99) — validator requires strictly between 0 and 1
        clamped_reward = max(0.01, min(0.99, self.reward))
        return {
            "state": self.state(),
            "reward": clamped_reward,
            "done": self.done,
            "info": info,
        }

    def _handle_read_record(self, action: dict) -> dict:
        """Read a specific medical record by ID."""
        record_id = action.get("record_id")
        if record_id is None:
            return {"error": "record_id is required for read_record action."}

        if record_id not in self.records:
            return {"error": f"Record ID {record_id} not found. Valid IDs: 1-{len(self.records)}"}

        # Consume budget
        self.budget -= 1
        self.steps_taken += 1

        # Track which records have been read
        if record_id not in self.reviewed_records:
            self.reviewed_records.append(record_id)

        # Return full record content
        record = self.records[record_id]
        return {"record": record}

    def _handle_cross_reference(self, action: dict) -> dict:
        """Search reference databases for relevant medical information."""
        query = action.get("query", "").lower().strip()
        if not query:
            return {"error": "query is required for cross_reference action."}

        # Consume budget
        self.budget -= 1
        self.steps_taken += 1

        results = {
            "drugs": [],
            "diseases": [],
            "lab_info": [],
        }

        # Search drugs database
        if "drugs" in self.drugs_db:
            for drug_key, drug_info in self.drugs_db["drugs"].items():
                drug_name = drug_info.get("name", "").lower()
                drug_class = drug_info.get("class", "").lower()
                indications = " ".join(drug_info.get("indication", [])).lower()

                if query in drug_name or query in drug_class or query in drug_key:
                    # Return drug info including interactions
                    results["drugs"].append({
                        "name": drug_info.get("name"),
                        "class": drug_info.get("class"),
                        "contraindications": drug_info.get("contraindications", []),
                        "warnings": drug_info.get("warnings", []),
                        "interactions": drug_info.get("interactions", {}),
                        "monitoring": drug_info.get("monitoring", []),
                    })

            # Also search interactions mentioning the query
            for drug_key, drug_info in self.drugs_db["drugs"].items():
                interactions = drug_info.get("interactions", {})
                for interact_key, interact_info in interactions.items():
                    if query in interact_key.lower() or query in interact_info.get("effect", "").lower():
                        results["drugs"].append({
                            "interaction_found": f"{drug_info['name']} + {interact_key}",
                            "severity": interact_info.get("severity"),
                            "effect": interact_info.get("effect"),
                        })

        # Search allergy cross-reactivity
        if "allergy_cross_reactivity" in self.drugs_db:
            for allergy_key, allergy_info in self.drugs_db["allergy_cross_reactivity"].items():
                if query in allergy_key.lower() or query in allergy_info.get("allergic_to", "").lower():
                    results["drugs"].append({
                        "allergy_info": allergy_info,
                    })

        # Search diseases database
        if "diseases" in self.diseases_db:
            for disease_key, disease_info in self.diseases_db["diseases"].items():
                disease_name = disease_info.get("name", "").lower()
                symptoms = " ".join(disease_info.get("symptoms", [])).lower() if isinstance(disease_info.get("symptoms"), list) else ""

                if query in disease_name or query in disease_key:
                    results["diseases"].append({
                        "name": disease_info.get("name"),
                        "icd10": disease_info.get("icd10"),
                        "diagnostic_criteria": disease_info.get("diagnostic_criteria", {}),
                        "key_labs": disease_info.get("key_labs", []),
                        "common_medications": disease_info.get("common_medications", []),
                        "monitoring_schedule": disease_info.get("monitoring_schedule", ""),
                        "common_missed_issues": disease_info.get("common_missed_issues", []),
                    })

        # Search lab ranges
        for category, labs in self.lab_ranges_db.items():
            if isinstance(labs, dict):
                for lab_key, lab_info in labs.items():
                    lab_name = lab_info.get("name", "").lower() if isinstance(lab_info, dict) else ""
                    if query in lab_key.lower() or query in lab_name:
                        results["lab_info"].append({
                            "test": lab_info.get("name", lab_key) if isinstance(lab_info, dict) else lab_key,
                            "category": category,
                            "details": lab_info,
                        })

        # Deduplicate drug results
        seen = set()
        unique_drugs = []
        for d in results["drugs"]:
            key = json.dumps(d, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                unique_drugs.append(d)
        results["drugs"] = unique_drugs

        if not results["drugs"] and not results["diseases"] and not results["lab_info"]:
            return {"message": f"No results found for query: '{query}'", "results": results}

        return {"results": results}

    def _handle_flag_issue(self, action: dict) -> dict:
        """Flag a potential issue found in the records."""
        issue_type = action.get("type", "")
        description = action.get("description", "")
        evidence = action.get("evidence", [])

        if not issue_type:
            return {"error": "type is required for flag_issue. Valid types: drug_interaction, drug_contraindication, allergy_violation, declining_trend, missed_monitoring, contradiction, missed_diagnosis"}

        if not description:
            return {"error": "description is required for flag_issue."}

        valid_types = [
            "drug_interaction",
            "drug_contraindication",
            "allergy_violation",
            "declining_trend",
            "missed_monitoring",
            "contradiction",
            "missed_diagnosis",
        ]

        if issue_type not in valid_types:
            return {"error": f"Invalid issue type: {issue_type}. Valid types: {valid_types}"}

        # Consume budget
        self.budget -= 1
        self.steps_taken += 1

        # Store the finding
        finding = {
            "type": issue_type,
            "description": description,
            "evidence": evidence,
            "step_flagged": self.steps_taken,
        }
        self.findings.append(finding)

        return {
            "message": f"Issue flagged: {issue_type}",
            "finding_number": len(self.findings),
            "budget_remaining": self.budget,
        }

    def _handle_submit_report(self) -> dict:
        """Submit the final report and end the episode."""
        self.done = True
        self.steps_taken += 1

        # Compute final reward
        self.reward = self._compute_reward()

        return {
            "message": "Report submitted. Episode ended.",
            "final_score": self.reward,
            "findings_submitted": len(self.findings),
            "records_reviewed": len(self.reviewed_records),
            "steps_taken": self.steps_taken,
            "ground_truth_count": len(self.ground_truth),
        }

    def _compute_reward(self) -> float:
        """
        Compute the reward (0.0 - 1.0) based on agent's findings vs ground truth.
        Uses programmatic matching (type + evidence overlap).
        """
        # No findings submitted = minimal score (validator requires > 0)
        if not self.findings:
            return 0.01

        if not self.ground_truth:
            return 0.01

        correct_findings = 0
        findings_score = 0.0
        matched_truths = set()

        for finding in self.findings:
            best_match = None
            best_score = 0.0

            for i, truth in enumerate(self.ground_truth):
                if i in matched_truths:
                    continue

                score = self._match_finding(finding, truth)
                if score > best_score:
                    best_score = score
                    best_match = i

            if best_match is not None and best_score > 0.5:
                matched_truths.add(best_match)
                correct_findings += 1
                severity = self.ground_truth[best_match].get("severity", "moderate")
                if severity == "critical":
                    findings_score += 0.25
                elif severity == "moderate":
                    findings_score += 0.15
                else:
                    findings_score += 0.10
            else:
                # False positive penalty
                findings_score -= 0.10

        # Cap findings score
        findings_score = max(0.0, min(findings_score, 0.7))

        # Efficiency bonus (reward for using fewer steps)
        efficiency_bonus = (self.budget / self.total_budget) * 0.15 if self.total_budget > 0 else 0.0

        # Completeness bonus
        total_issues = len(self.ground_truth)
        completeness_bonus = (correct_findings / total_issues) * 0.15 if total_issues > 0 else 0.0

        total = findings_score + efficiency_bonus + completeness_bonus
        # Clamp to (0.01, 0.99) — validator requires strictly between 0 and 1
        return round(max(0.01, min(0.99, total)), 4)

    def _match_finding(self, finding: dict, truth: dict) -> float:
        """
        Score how well an agent's finding matches a ground truth issue.
        Returns 0.0 - 1.0.
        """
        score = 0.0

        # Type match (0.4 weight)
        if finding.get("type") == truth.get("type"):
            score += 0.4

        # Evidence overlap (0.3 weight)
        finding_evidence = set(finding.get("evidence", []))
        truth_evidence = set(truth.get("evidence_records", []))
        if finding_evidence and truth_evidence:
            overlap = len(finding_evidence & truth_evidence)
            total = len(truth_evidence)
            if total > 0:
                evidence_score = min(overlap / max(total * 0.3, 1), 1.0)  # Need at least 30% overlap
                score += 0.3 * evidence_score

        # Description keyword overlap (0.3 weight)
        finding_words = set(finding.get("description", "").lower().split())
        truth_words = set(truth.get("description", "").lower().split())
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "was", "were", "been", "be", "have", "has",
                       "had", "do", "does", "did", "will", "would", "could", "should", "may",
                       "might", "shall", "can", "need", "dare", "ought", "used", "to", "of",
                       "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
                       "during", "before", "after", "above", "below", "between", "and", "but",
                       "or", "nor", "not", "so", "yet", "both", "either", "neither", "each",
                       "every", "all", "any", "few", "more", "most", "other", "some", "such",
                       "no", "only", "own", "same", "than", "too", "very", "just", "because",
                       "this", "that", "these", "those", "it", "its", "which", "who", "whom",
                       "what", "where", "when", "why", "how", "if", "then", "else", "while",
                       "patient", "record", "records", "noted", "found", "despite"}
        finding_words -= stop_words
        truth_words -= stop_words
        if finding_words and truth_words:
            keyword_overlap = len(finding_words & truth_words)
            max_possible = max(len(truth_words), 1)
            keyword_score = min(keyword_overlap / (max_possible * 0.2), 1.0)  # Need 20% keyword match
            score += 0.3 * keyword_score

        return score

    def state(self) -> dict:
        """
        Return the current state visible to the agent.
        Ground truth issues are NEVER exposed.
        """
        if self.case is None:
            return {"error": "No episode started. Call reset() first."}

        return {
            "case_id": self.case_id,
            "difficulty": self.difficulty,
            "task": self.task,
            "patient": self.patient,
            "records_available": len(self.records),
            "records_reviewed": self.reviewed_records,
            "findings": self.findings,
            "steps_taken": self.steps_taken,
            "budget_remaining": self.budget,
            "done": self.done,
            "available_actions": [
                "read_record",
                "cross_reference",
                "flag_issue",
                "submit_report",
            ],
            "record_index": self.record_index,
        }
