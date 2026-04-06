"""
Grader: hard_002
Patient: 55F with SLE, lupus nephritis, antiphospholipid syndrome, osteoporosis, MDD
Issues: HCQ retinopathy screening missed, calcium-levothyroxine interaction,
        INR gaps during flares, nephritis trend, DEXA gap on prednisone
"""
from env.graders.base import run_case

def grader() -> dict:
    return run_case("hard", "hard_002")

if __name__ == "__main__":
    print(grader())
