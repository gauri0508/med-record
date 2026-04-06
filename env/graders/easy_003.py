"""
Grader: easy_003
Patient: 58M with T2DM and progressive CKD
Issue: Metformin continued despite eGFR dropping below 30
"""
from env.graders.base import run_case

def grader() -> dict:
    return run_case("easy", "easy_003")

if __name__ == "__main__":
    print(grader())
