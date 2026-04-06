"""Run all 9 graders when called as: python -m env.graders"""

from env.graders.base import run_case


def main():
    print("Running all 9 case graders...\n")

    for difficulty, count in [("easy", 3), ("medium", 3), ("hard", 3)]:
        print(f"--- {difficulty.upper()} ---")
        for i in range(1, count + 1):
            case_id = f"{difficulty}_{i:03d}"
            result = run_case(difficulty, case_id)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  [{status}] {case_id}: score={result['score']}, "
                  f"findings={result['findings_submitted']}, "
                  f"records_read={result['records_reviewed']}, "
                  f"steps={result['steps_taken']}, "
                  f"ground_truth={result['ground_truth_count']}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
