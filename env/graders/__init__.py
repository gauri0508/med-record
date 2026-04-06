"""
MedRecordAudit — Graders Package

Individual graders for each case + difficulty-level + run-all utilities.
9 total cases: 3 easy, 3 medium, 3 hard.
"""

from env.graders.easy_001 import grader as easy_001_grader
from env.graders.easy_002 import grader as easy_002_grader
from env.graders.easy_003 import grader as easy_003_grader

from env.graders.medium_001 import grader as medium_001_grader
from env.graders.medium_002 import grader as medium_002_grader
from env.graders.medium_003 import grader as medium_003_grader

from env.graders.hard_001 import grader as hard_001_grader
from env.graders.hard_002 import grader as hard_002_grader
from env.graders.hard_003 import grader as hard_003_grader


def easy_grader() -> dict:
    """Run all 3 easy case graders."""
    results = {
        "easy_001": easy_001_grader(),
        "easy_002": easy_002_grader(),
        "easy_003": easy_003_grader(),
    }
    results["all_passed"] = all(r["passed"] for r in results.values() if isinstance(r, dict))
    return results


def medium_grader() -> dict:
    """Run all 3 medium case graders."""
    results = {
        "medium_001": medium_001_grader(),
        "medium_002": medium_002_grader(),
        "medium_003": medium_003_grader(),
    }
    results["all_passed"] = all(r["passed"] for r in results.values() if isinstance(r, dict))
    return results


def hard_grader() -> dict:
    """Run all 3 hard case graders."""
    results = {
        "hard_001": hard_001_grader(),
        "hard_002": hard_002_grader(),
        "hard_003": hard_003_grader(),
    }
    results["all_passed"] = all(r["passed"] for r in results.values() if isinstance(r, dict))
    return results


def run_all() -> dict:
    """Run all 9 case graders."""
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
