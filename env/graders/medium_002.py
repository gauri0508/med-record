"""
Grader: medium_002
Patient: 69M with AFib, HFrEF, HTN, T2DM, hyperlipidemia
Issues: Amiodarone thyroid monitoring gap + liver trend + renal decline
"""
from env.graders.base import run_case

def grader() -> dict:
    return run_case("medium", "medium_002")

if __name__ == "__main__":
    print(grader())
