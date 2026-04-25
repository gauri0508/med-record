"""
MedRecordAudit — Graders Package

One representative case per difficulty: easy_001, medium_001, hard_001.
Each case is comprehensive enough to exercise the full rubric (all 5
components) at its difficulty tier.
"""

from env.graders.easy_001 import grader as easy_001_grader
from env.graders.medium_001 import grader as medium_001_grader
from env.graders.hard_001 import grader as hard_001_grader


def easy_grader() -> dict:
    """Run the easy-difficulty grader."""
    result = easy_001_grader()
    return {"easy_001": result, "all_passed": result.get("passed", False)}


def medium_grader() -> dict:
    """Run the medium-difficulty grader."""
    result = medium_001_grader()
    return {"medium_001": result, "all_passed": result.get("passed", False)}


def hard_grader() -> dict:
    """Run the hard-difficulty grader."""
    result = hard_001_grader()
    return {"hard_001": result, "all_passed": result.get("passed", False)}


def run_all() -> dict:
    """Run all 3 difficulty graders."""
    results = {
        "easy": easy_grader(),
        "medium": medium_grader(),
        "hard": hard_grader(),
    }
    results["all_passed"] = (
        results["easy"]["all_passed"]
        and results["medium"]["all_passed"]
        and results["hard"]["all_passed"]
    )
    return results
