"""
Grader: easy_002
Patient: 72M with AFib on warfarin
Issue: Ibuprofen prescribed to warfarin patient (major bleeding risk)
"""
from env.graders.base import run_case

def grader() -> dict:
    return run_case("easy", "easy_002")

if __name__ == "__main__":
    print(grader())
