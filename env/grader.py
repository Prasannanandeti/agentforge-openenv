from typing import List, Dict, Any
from .tasks import EXPECTED_FLOWS, EMPATHY_KEYWORDS

class AgentForgeGrader:
    @staticmethod
    def evaluate(task_id: str, history: List[Dict], action_sequence: List[str], internal_state: Dict, steps: int) -> float:
        score = 0.0
        expected = EXPECTED_FLOWS.get(task_id, [])


        # -------------------------
        # 1. Goal Completion (40%)
        # -------------------------
        if internal_state.get("closed"):
            if task_id == "easy_1" and internal_state.get("info_provided"):
                score += 0.4
            elif task_id == "medium_1" and internal_state.get("refund_processed"):
                score += 0.4
            elif task_id == "hard_1" and internal_state.get("refund_denied_explained") and internal_state.get("asked_id"):
                score += 0.4

        # -------------------------
        # 2. Sequence Logic (30%)
        # -------------------------
        actual_filtered = [a for a in action_sequence if a != "invalid"]

        if actual_filtered == expected:
            score += 0.3
        elif actual_filtered[:len(expected)] == expected:
            score += 0.2
        elif any(step in actual_filtered for step in expected):
            score += 0.1

        # -------------------------
        # 3. Quality Metrics (30%)
        # -------------------------

        # Tool usage quality
        if internal_state.get("tool_errors", 0) == 0:
            score += 0.1

        # Hallucination penalty
        hallucinations = internal_state.get("hallucinations", 0)
        score -= min(0.2, hallucinations * 0.1)

        # Response quality
        flat_history = str(history).lower()

        if task_id == "hard_1":
            if any(word in flat_history for word in EMPATHY_KEYWORDS):
                score += 0.2
        else:
            score += 0.2

        # -------------------------
        # 4. Efficiency Penalty
        # -------------------------
        penalty = 0.05 * steps

        final_score = score - penalty

        return round(max(0.0, min(1.0, final_score)), 2)

