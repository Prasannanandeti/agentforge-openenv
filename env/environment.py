import uuid
from typing import Tuple, Dict, Any, List
from .models import Action, Observation, Reward
from .tasks import TASKS, mock_tool_call, EXPECTED_FLOWS, EMPATHY_KEYWORDS
from .grader import AgentForgeGrader

class AgentForgeEnv:
    def __init__(self):
        self.max_steps = 8
        self.reset()


    def reset(self, task_id: str = "easy_1") -> Observation:
        self.current_task = next(t for t in TASKS if t["id"] == task_id)
        self.steps = 0
        self.history = [{"role": "user", "content": self.current_task["initial_query"]}]
        self.action_sequence = []
        self.internal_state = {
            "closed": False,
            "refund_processed": False,
            "info_provided": False,
            "asked_id": False,
            "refund_denied_explained": False,
            "tool_errors": 0,
            "hallucinations": 0
        }
        self.done = False
        self.last_tool_output = None
        return self._get_obs()

    def _get_obs(self) -> Observation:
        return Observation(
            user_query=self.current_task["initial_query"],
            conversation_history=self.history,
            current_step=self.steps,
            task_context=self.current_task["context"],
            internal_tools_output=str(self.last_tool_output) if self.last_tool_output else None
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.done:
            return self._get_obs(), Reward(value=0.0, reason="Terminal state", is_terminal=True), True, {}

        self.steps += 1
        reward_val = -0.05  # Efficiency penalty
        reasons = []

        # -------------------------
        # Hallucination Check
        # -------------------------
        if action.tool_params and "order_id" in action.tool_params:
            correct_id = self.current_task["context"].get("order_id")
            if correct_id and action.tool_params["order_id"] != correct_id:
                reward_val -= 0.3
                self.internal_state["hallucinations"] += 1
                reasons.append("Hallucination")

        # -------------------------
        # Action Handling
        # -------------------------
        if action.action_type == "reply":
            self.history.append({"role": "assistant", "content": action.text or ""})
            self.action_sequence.append("reply")

            txt = (action.text or "").lower()

            # Hard task logic
            if self.current_task["id"] == "hard_1":
                if "processing" in txt and ("cannot" in txt or "policy" in txt):
                    self.internal_state["refund_denied_explained"] = True
                if any(w in txt for w in EMPATHY_KEYWORDS):
                    reward_val += 0.1

            if "ord-" in txt:
                self.internal_state["info_provided"] = True

            reward_val += 0.05

        elif action.action_type == "ask_info":
            self.action_sequence.append("ask_info")

            if action.field == "order_id":
                self.internal_state["asked_id"] = True
                reward_val += 0.2

            self.history.append({"role": "assistant", "content": f"Clarifying {action.field}"})

        elif action.action_type == "call_tool":
            if not action.tool_name:
                reward_val -= 0.2
                self.action_sequence.append("invalid")
                reasons.append("Missing tool name")
            else:
                res = mock_tool_call(action.tool_name, action.tool_params)
                self.last_tool_output = res

                if res["status"] == "failed":
                    reward_val -= 0.4
                    self.internal_state["tool_errors"] += 1
                    self.action_sequence.append("invalid")
                else:
                    self.action_sequence.append("call_tool")
                    reward_val += 0.2

                    if res["status"] == "success" and action.tool_name == "process_refund":
                        self.internal_state["refund_processed"] = True

        elif action.action_type == "close_ticket":
            self.action_sequence.append("close_ticket")
            self.internal_state["closed"] = True
            self.done = True

            score = AgentForgeGrader.evaluate(
                self.current_task["id"],
                self.history,
                self.action_sequence,
                self.internal_state,
                self.steps
            )

            return self._get_obs(), Reward(value=score, reason="Task complete", is_terminal=True), True, {"score": score}

        else:
            self.action_sequence.append("invalid")
            reward_val -= 0.2
            reasons.append("Invalid action")

        # -------------------------
        # Sequence Bonus
        # -------------------------
        expected = EXPECTED_FLOWS.get(self.current_task["id"], [])
        idx = len(self.action_sequence) - 1

        if idx < len(expected) and self.action_sequence[-1] == expected[idx]:
            reward_val += 0.1
            reasons.append("Correct sequence")

        # -------------------------
        # Timeout Handling
        # -------------------------
        if self.steps >= self.max_steps:
            self.done = True
            return self._get_obs(), Reward(value=-0.5, reason="Timeout", is_terminal=True), True, {"score": 0.0}

        # -------------------------
        # Normalize Reward
        # -------------------------
        reward_val = max(0.0, min(1.0, reward_val))

        return self._get_obs(), Reward(
            value=round(reward_val, 2),
            reason="; ".join(reasons) or "Step",
            is_terminal=False
        ), False, {}

    def state(self) -> Dict:
        return {
            "task_id": self.current_task["id"],
            "steps": self.steps,
            "done": self.done,
            "sequence": self.action_sequence,
            "internal_state": self.internal_state
        }

