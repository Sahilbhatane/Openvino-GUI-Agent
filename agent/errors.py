"""
Structured error types for the GUI agent.

Every error that surfaces in the UI carries a category, message,
and optional recovery description so the user always knows what happened.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class ErrorCategory(str, Enum):
    MODEL = "model"
    EXECUTION = "execution"
    OS = "os"
    GROUNDING = "grounding"
    SAFETY = "safety"
    TIMEOUT = "timeout"


@dataclass
class AgentError:
    category: ErrorCategory
    message: str
    recovery: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        d = {"category": self.category.value, "message": self.message}
        if self.recovery:
            d["recovery"] = self.recovery
        if self.detail:
            d["detail"] = self.detail
        return d

    def __str__(self) -> str:
        parts = [f"[{self.category.value.upper()}] {self.message}"]
        if self.recovery:
            parts.append(f"Recovery: {self.recovery}")
        return " | ".join(parts)


@dataclass
class StepErrors:
    """Collects all errors for a single iteration."""
    errors: list[AgentError] = field(default_factory=list)

    def add(
        self,
        category: ErrorCategory,
        message: str,
        recovery: str = "",
        detail: str = "",
    ) -> AgentError:
        err = AgentError(category=category, message=message, recovery=recovery, detail=detail)
        self.errors.append(err)
        return err

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self.errors]
