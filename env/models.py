from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal

class Action(BaseModel):
    action_type: Literal["reply", "ask_info", "call_tool", "close_ticket"] = Field(
    ..., description="Type of action"
    )
    text: Optional[str] = Field(None, description="Response text for 'reply'")
    field: Optional[str] = Field(None, description="Field name for 'ask_info'")
    tool_name: Optional[str] = Field(None, description="Tool to call")
    tool_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for tool")


    @field_validator("tool_name")
    def validate_tool_name(cls, v, info):
        if info.data.get("action_type") == "call_tool" and not v:
            raise ValueError("tool_name is required for call_tool")
        return v


class Observation(BaseModel):
    user_query: str
    conversation_history: List[Dict[str, str]]
    current_step: int
    task_context: Dict[str, Any]
    internal_tools_output: Optional[str] = None

class Reward(BaseModel):
    value: float
    reason: str
    is_terminal: bool
