"""
MedRecordAudit — Composable Rubric System

Each rubric scores one independent dimension of agent behavior.
They are intentionally decoupled — a finding can score high on
finding_accuracy but low on evidence_validity if the agent guessed
the right type without reading the relevant records.

This replaces the old monolithic _compute_reward() so that during
GRPO training each component can be logged and analyzed separately.

All rubrics return floats clamped to their MAX_SCORE.
"""

from dataclasses import dataclass, field
from typing import List


# Common English / domain stop words excluded from keyword matching
STOP_WORDS = frozenset({
    "the", "a", "an", "is", "was", "were", "been", "be", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "need", "dare", "ought", "used", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "and", "but",
    "or", "nor", "not", "so", "yet", "both", "either", "neither", "each",
    "every", "all", "any", "few", "more", "most", "other", "some", "such",
    "no", "only", "own", "same", "than", "too", "very", "just", "because",
    "this", "that", "these", "those", "it", "its", "which", "who", "whom",
    "what", "where", "when", "why", "how", "if", "then", "else", "while",
    "patient", "record", "records", "noted", "found", "despite", "also", "still",
})


@dataclass
class RubricScores:
    """Container returned by compute_rubric_scores()."""
    finding_accuracy: float = 0.0     # 0.0 - 0.40
    evidence_validity: float = 0.0    # 0.0 - 0.20
    completeness: float = 0.0         # 0.0 - 0.20
    efficiency: float = 0.0           # 0.0 - 0.10
    anti_hacking: float = 0.0         # 0.0 - 0.10
    total: float = 0.01               # clamped to (0.01, 0.99)

    correct_findings: int = 0
    false_positives: int = 0
    total_ground_truth: int = 0
    duplicate_flags: int = 0
    unread_evidence_citations: int = 0
    matches: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "finding_accuracy": round(self.finding_accuracy, 4),
            "evidence_validity": round(self.evidence_validity, 4),
            "completeness": round(self.completeness, 4),
            "efficiency": round(self.efficiency, 4),
            "anti_hacking": round(self.anti_hacking, 4),
            "total": round(self.total, 4),
            "correct_findings": self.correct_findings,
            "false_positives": self.false_positives,
            "total_ground_truth": self.total_ground_truth,
            "duplicate_flags": self.duplicate_flags,
            "unread_evidence_citations": self.unread_evidence_citations,
            "matches": self.matches,
        }


class RubricFindingAccuracy:
    """
    Max contribution: 0.40

    Scores how well the agent's finding type and clinical description
    match a ground truth issue. Decoupled from evidence — clinically right
    but cited wrong records is EvidenceValidity's problem, not this rubric's.

    Per-finding match score:
      - Type match (exact): 0.50 weight
      - Keyword overlap (description vs truth.description): 0.50 weight

    A finding counts as correct if match_score >= MATCH_THRESHOLD (0.50).
    Each correct finding adds severity-based reward; false positives subtract.
    """
    MAX_SCORE = 0.40
    SEVERITY_REWARDS = {"critical": 0.25, "moderate": 0.15, "minor": 0.10}
    FALSE_POSITIVE_PENALTY = -0.10
    # Tightened from 0.50 to 0.55 (Phase 0 finding): a type-only match
    # gives exactly 0.50 from the type weight alone, which let random-flag
    # agents accidentally clear the threshold. Now correct findings must
    # also include some keyword overlap with the truth description.
    MATCH_THRESHOLD = 0.55
    MIN_KEYWORD_RATIO = 0.15

    def score(self, findings: list, ground_truth: list):
        """
        Returns (rubric_score, matches_list, correct_count, false_positive_count).
        """
        if not findings or not ground_truth:
            return 0.0, [], 0, len(findings) if findings else 0

        matched_truths = set()
        correct = 0
        false_pos = 0
        raw_score = 0.0
        matches = []

        for f_idx, finding in enumerate(findings):
            best_idx = None
            best = 0.0

            for t_idx, truth in enumerate(ground_truth):
                if t_idx in matched_truths:
                    continue
                s = self._match_one(finding, truth)
                if s > best:
                    best = s
                    best_idx = t_idx

            if best_idx is not None and best >= self.MATCH_THRESHOLD:
                matched_truths.add(best_idx)
                correct += 1
                severity = ground_truth[best_idx].get("severity", "moderate")
                reward = self.SEVERITY_REWARDS.get(severity, 0.10)
                raw_score += reward
                matches.append({
                    "finding_idx": f_idx,
                    "truth_idx": best_idx,
                    "match_score": round(best, 3),
                    "severity": severity,
                    "reward": reward,
                })
            else:
                false_pos += 1
                raw_score += self.FALSE_POSITIVE_PENALTY

        rubric_score = max(0.0, min(raw_score, self.MAX_SCORE))
        return round(rubric_score, 4), matches, correct, false_pos

    def _match_one(self, finding: dict, truth: dict) -> float:
        score = 0.0

        if finding.get("type") == truth.get("type"):
            score += 0.50

        f_words = set(finding.get("description", "").lower().split()) - STOP_WORDS
        t_words = set(truth.get("description", "").lower().split()) - STOP_WORDS
        if f_words and t_words:
            overlap = len(f_words & t_words)
            min_kw = max(1, int(len(t_words) * self.MIN_KEYWORD_RATIO))
            if overlap >= 1:
                score += 0.50 * min(overlap / min_kw, 1.0)

        return score


class RubricEvidenceValidity:
    """
    Max contribution: 0.20

    Anti-hacking rubric. Checks whether the records the agent cited as
    evidence in flag_issue were actually opened via read_record earlier.

    An agent that flags issues without reading records can still match
    the right type by chance, but it will lose almost all of this rubric.

    Logic:
      - For all findings combined, ratio = (cited records that were read) / (total cited).
      - rubric_score = MAX_SCORE * ratio
      - If no evidence cited at all, partial score (30% of MAX) — agent
        is being lazy but didn't actively cheat.
    """
    MAX_SCORE = 0.20
    NO_EVIDENCE_FRACTION = 0.30

    def score(self, findings: list, reviewed_records: List[int]):
        """
        Returns (rubric_score, total_unread_citations).
        """
        if not findings:
            return 0.0, 0

        reviewed_set = set(reviewed_records)
        total_cited = 0
        total_valid = 0
        unread_citations = 0

        for finding in findings:
            evidence = finding.get("evidence", []) or []
            for eid in evidence:
                total_cited += 1
                if eid in reviewed_set:
                    total_valid += 1
                else:
                    unread_citations += 1

        if total_cited == 0:
            return round(self.MAX_SCORE * self.NO_EVIDENCE_FRACTION, 4), 0

        validity_ratio = total_valid / total_cited
        return round(self.MAX_SCORE * validity_ratio, 4), unread_citations


class RubricCompleteness:
    """
    Max contribution: 0.20

    Rewards finding ALL ground truth issues — easy cases have 1, medium 3,
    hard 5-6. An agent that finds 5/6 hard issues should score much higher
    here than one that finds 1/6 and submits early.

    Formula: (correct_findings / total_ground_truth) * MAX_SCORE
    """
    MAX_SCORE = 0.20

    def score(self, correct_findings: int, total_ground_truth: int) -> float:
        if total_ground_truth <= 0:
            return 0.0
        ratio = min(correct_findings / total_ground_truth, 1.0)
        return round(self.MAX_SCORE * ratio, 4)


class RubricEfficiency:
    """
    Max contribution: 0.10

    Rewards agents that submit with budget remaining. Intentionally low
    weight (0.10) — Completeness (0.20) is double, so agents are encouraged
    to investigate thoroughly rather than submit immediately.

    Formula: (budget_remaining / total_budget) * MAX_SCORE
    """
    MAX_SCORE = 0.10

    def score(self, budget_remaining: int, total_budget: int) -> float:
        if total_budget <= 0:
            return 0.0
        ratio = max(0.0, min(budget_remaining / total_budget, 1.0))
        return round(self.MAX_SCORE * ratio, 4)


class RubricAntiHacking:
    """
    Max contribution: 0.10

    Penalizes known reward-hacking patterns:
      1. Duplicate flags: same (type, description_prefix) flagged twice
      2. Description stuffing: > 500 chars (keyword bombing)

    Starts at MAX_SCORE, subtracts penalties down to 0.
    Detection of unread-evidence flagging is handled by RubricEvidenceValidity.
    """
    MAX_SCORE = 0.10
    DESCRIPTION_LENGTH_LIMIT = 500
    DUPLICATE_PENALTY = 0.05
    STUFFING_PENALTY = 0.05

    def score(self, findings: list, reviewed_records: List[int]):
        """
        Returns (rubric_score, duplicate_count).
        """
        if not findings:
            return self.MAX_SCORE, 0

        score = self.MAX_SCORE
        duplicates = 0
        seen_signatures = set()

        for finding in findings:
            desc = finding.get("description", "") or ""
            if len(desc) > self.DESCRIPTION_LENGTH_LIMIT:
                score -= self.STUFFING_PENALTY

            sig = (finding.get("type", ""), desc[:60].lower().strip())
            if sig in seen_signatures:
                score -= self.DUPLICATE_PENALTY
                duplicates += 1
            else:
                seen_signatures.add(sig)

        return round(max(0.0, score), 4), duplicates


def compute_rubric_scores(
    findings: list,
    ground_truth: list,
    reviewed_records: List[int],
    budget_remaining: int,
    total_budget: int,
) -> RubricScores:
    """
    Main entry point. Compute all 5 rubric scores and combine into total.

    This replaces the old _compute_reward() / compute_reward() functions.
    The total is clamped to (0.01, 0.99) per the OpenEnv validator requirement.

    Returns:
        RubricScores dataclass with all 5 components, total, and diagnostic counts.
    """
    rubric_accuracy = RubricFindingAccuracy()
    rubric_evidence = RubricEvidenceValidity()
    rubric_completeness = RubricCompleteness()
    rubric_efficiency = RubricEfficiency()
    rubric_antihack = RubricAntiHacking()

    if not findings:
        return RubricScores(
            total=0.01,
            total_ground_truth=len(ground_truth) if ground_truth else 0,
            efficiency=rubric_efficiency.score(budget_remaining, total_budget),
        )

    if not ground_truth:
        return RubricScores(
            total=0.01,
            false_positives=len(findings),
            efficiency=rubric_efficiency.score(budget_remaining, total_budget),
        )

    accuracy_score, matches, correct, false_pos = rubric_accuracy.score(findings, ground_truth)
    evidence_score, unread_citations = rubric_evidence.score(findings, reviewed_records)
    completeness_score = rubric_completeness.score(correct, len(ground_truth))
    efficiency_score = rubric_efficiency.score(budget_remaining, total_budget)
    antihack_score, duplicates = rubric_antihack.score(findings, reviewed_records)

    raw_total = (
        accuracy_score
        + evidence_score
        + completeness_score
        + efficiency_score
        + antihack_score
    )
    total = round(max(0.01, min(0.99, raw_total)), 4)

    return RubricScores(
        finding_accuracy=accuracy_score,
        evidence_validity=evidence_score,
        completeness=completeness_score,
        efficiency=efficiency_score,
        anti_hacking=antihack_score,
        total=total,
        correct_findings=correct,
        false_positives=false_pos,
        total_ground_truth=len(ground_truth),
        duplicate_flags=duplicates,
        unread_evidence_citations=unread_citations,
        matches=matches,
    )
