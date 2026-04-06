"""
Grader: easy_001
Patient: 45F with asthma, penicillin allergy
Issue: Amoxicillin prescribed despite documented penicillin anaphylaxis
"""
from env.graders.base import run_case

def grader() -> dict:
    return run_case("easy", "easy_001")

if __name__ == "__main__":
    print(grader())
