"""
Grader: hard_001
Patient: 74M with HFrEF, AFib, CKD, T2DM, COPD, gout, MDD (14 meds, 7 specialists)
Issues: Amiodarone-warfarin interaction, metformin below eGFR 30,
        hyperkalemia cascade, SSRI hyponatremia, steroid diabetes, PPI deficiencies
"""
from env.graders.base import run_case

def grader() -> dict:
    return run_case("hard", "hard_001")

if __name__ == "__main__":
    print(grader())
