"""
Grader: medium_003
Patient: 38F with recurrent iron deficiency anemia, hypothyroidism, IBS
Issues: Missed celiac disease + omeprazole malabsorption + autoimmune screening gap
"""
from env.graders.base import run_case

def grader() -> dict:
    return run_case("medium", "medium_003")

if __name__ == "__main__":
    print(grader())
