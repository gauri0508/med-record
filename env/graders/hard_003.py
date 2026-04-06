"""
Grader: hard_003
Patient: 82M with CAD, AFib, HFpEF, T2DM, CKD, HTN, BPH, OA, insomnia, GERD (15 drugs)
Issues: Aspirin+warfarin dual therapy, digoxin toxicity cascade, glipizide hypoglycemia,
        zolpidem falls/hip fracture, ACE cough class-effect, PPI cascading deficiencies
"""
from env.graders.base import run_case

def grader() -> dict:
    return run_case("hard", "hard_003")

if __name__ == "__main__":
    print(grader())
