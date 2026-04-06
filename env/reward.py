"""
MedRecordAudit — Reward Computation

Standalone reward module that can be used by graders and the environment.
Scores agent findings against ground truth issues on a 0.0 - 1.0 scale.

Scoring has two layers:
1. Programmatic: type matching + evidence record overlap
2. Keyword-based: description similarity via keyword overlap
"""

# Reward values per severity
SEVERITY_REWARDS = {
    "critical": 0.25,
    "moderate": 0.15,
    "minor": 0.10,
}

FALSE_POSITIVE_PENALTY = -0.10
MAX_FINDINGS_SCORE = 0.70
EFFICIENCY_WEIGHT = 0.15
COMPLETENESS_WEIGHT = 0.15

# Common English stop words to exclude from keyword matching
STOP_WORDS = frozenset({
    "the", "a", "an", "is", "was", "were", "been", "be", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "can", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "and",
    "but", "or", "not", "so", "yet", "both", "either", "each", "every",
    "all", "any", "few", "more", "most", "other", "some", "no", "only",
    "own", "same", "than", "too", "very", "just", "because", "this",
    "that", "these", "those", "it", "its", "which", "who", "what",
    "where", "when", "why", "how", "if", "then", "while", "patient",
    "record", "records", "noted", "found", "despite", "also", "still",
})


def match_finding(finding: dict, truth: dict) -> float:
    """
    Score how well an agent's finding matches a ground truth issue.

    Scoring breakdown:
    - Type match: 0.4 (exact match on issue type)
    - Evidence overlap: 0.3 (overlap between cited record IDs)
    - Description keywords: 0.3 (keyword overlap in descriptions)

    Returns: 0.0 - 1.0
    """
    score = 0.0

    # --- Type match (0.4) ---
    if finding.get("type") == truth.get("type"):
        score += 0.4

    # --- Evidence overlap (0.3) ---
    finding_evidence = set(finding.get("evidence", []))
    truth_evidence = set(truth.get("evidence_records", []))
    if finding_evidence and truth_evidence:
        overlap = len(finding_evidence & truth_evidence)
        # Need at least 1 overlapping record, scale by truth size
        min_needed = max(1, int(len(truth_evidence) * 0.3))
        if overlap >= 1:
            evidence_ratio = min(overlap / min_needed, 1.0)
            score += 0.3 * evidence_ratio

    # --- Description keyword overlap (0.3) ---
    finding_words = set(finding.get("description", "").lower().split()) - STOP_WORDS
    truth_words = set(truth.get("description", "").lower().split()) - STOP_WORDS
    if finding_words and truth_words:
        keyword_overlap = len(finding_words & truth_words)
        min_keywords = max(1, int(len(truth_words) * 0.15))
        if keyword_overlap >= 1:
            keyword_ratio = min(keyword_overlap / min_keywords, 1.0)
            score += 0.3 * keyword_ratio

    return score


def compute_reward(
    findings: list,
    ground_truth: list,
    budget_remaining: int,
    total_budget: int,
) -> dict:
    """
    Compute the final reward for an episode.

    Args:
        findings: list of agent's flagged issues
        ground_truth: list of ground truth issues
        budget_remaining: steps remaining when report submitted
        total_budget: total steps available at episode start

    Returns:
        dict with total score and breakdown:
        {
            "total": 0.0-1.0,
            "findings_score": float,
            "efficiency_bonus": float,
            "completeness_bonus": float,
            "correct_findings": int,
            "false_positives": int,
            "total_ground_truth": int,
            "matches": [{"finding_idx": int, "truth_idx": int, "score": float}]
        }
    """
    # If no findings submitted, score is 0 — no bonus for doing nothing
    if not findings:
        return {
            "total": 0.0,
            "findings_score": 0.0,
            "efficiency_bonus": 0.0,
            "completeness_bonus": 0.0,
            "correct_findings": 0,
            "false_positives": 0,
            "total_ground_truth": len(ground_truth),
            "matches": [],
        }

    if not ground_truth:
        return {
            "total": 0.0,
            "findings_score": 0.0,
            "efficiency_bonus": 0.0,
            "completeness_bonus": 0.0,
            "correct_findings": 0,
            "false_positives": 0,
            "total_ground_truth": 0,
            "matches": [],
        }

    correct_findings = 0
    false_positives = 0
    findings_score = 0.0
    matched_truths = set()
    matches = []

    for f_idx, finding in enumerate(findings):
        best_match_idx = None
        best_score = 0.0

        # Find best matching ground truth for this finding
        for t_idx, truth in enumerate(ground_truth):
            if t_idx in matched_truths:
                continue

            score = match_finding(finding, truth)
            if score > best_score:
                best_score = score
                best_match_idx = t_idx

        # Threshold: need at least 0.5 match score to count
        if best_match_idx is not None and best_score >= 0.5:
            matched_truths.add(best_match_idx)
            correct_findings += 1
            severity = ground_truth[best_match_idx].get("severity", "moderate")
            reward = SEVERITY_REWARDS.get(severity, 0.10)
            findings_score += reward
            matches.append({
                "finding_idx": f_idx,
                "truth_idx": best_match_idx,
                "score": round(best_score, 3),
                "severity": severity,
                "reward": reward,
            })
        else:
            false_positives += 1
            findings_score += FALSE_POSITIVE_PENALTY

    # Cap findings score
    findings_score = max(0.0, min(findings_score, MAX_FINDINGS_SCORE))

    # Efficiency bonus
    efficiency_bonus = 0.0
    if total_budget > 0:
        efficiency_bonus = (budget_remaining / total_budget) * EFFICIENCY_WEIGHT

    # Completeness bonus
    total_issues = len(ground_truth)
    completeness_bonus = 0.0
    if total_issues > 0:
        completeness_bonus = (correct_findings / total_issues) * COMPLETENESS_WEIGHT

    total = findings_score + efficiency_bonus + completeness_bonus
    total = round(max(0.0, min(1.0, total)), 4)

    return {
        "total": total,
        "findings_score": round(findings_score, 4),
        "efficiency_bonus": round(efficiency_bonus, 4),
        "completeness_bonus": round(completeness_bonus, 4),
        "correct_findings": correct_findings,
        "false_positives": false_positives,
        "total_ground_truth": total_issues,
        "matches": matches,
    }
