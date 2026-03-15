"""
Structured action schemas for the GUI agent.

Every VLM response is parsed into an ActionPlan before execution.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"


class Action(BaseModel):
    """A single GUI action the executor can perform."""

    type: ActionType
    x: Optional[int] = None
    y: Optional[int] = None
    text: Optional[str] = None
    amount: Optional[int] = None       # scroll amount
    seconds: Optional[float] = None    # wait duration
    description: Optional[str] = None

    def summary(self) -> str:
        match self.type:
            case ActionType.CLICK:
                return f"click({self.x}, {self.y})"
            case ActionType.DOUBLE_CLICK:
                return f"double_click({self.x}, {self.y})"
            case ActionType.TYPE:
                return f'type("{self.text}")'
            case ActionType.SCROLL:
                return f"scroll({self.amount})"
            case ActionType.WAIT:
                return f"wait({self.seconds}s)"
            case _:
                return str(self.model_dump())


class ActionPlan(BaseModel):
    """The full structured output expected from the planner."""

    thought: str = ""
    actions: list[Action] = Field(default_factory=list)
    task_complete: bool = False
