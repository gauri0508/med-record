"""
Grader: medium_001
Patient: 71F with T2DM, HTN, AFib, OA, GERD, depression
Issues: SSRI hyponatremia + tramadol+bupropion seizure risk + HbA1c trend
"""
from env.graders.base import run_case

def grader() -> dict:
    return run_case("medium", "medium_001")

if __name__ == "__main__":
    print(grader())
