"""
Safety system for the GUI agent.

Validates actions before execution, blocks dangerous operations,
and enforces per-iteration action limits.
"""

import re
from typing import Optional

from models.action_schema import Action, ActionType
from utils.logger import get_logger

log = get_logger("safety")

DANGEROUS_COMMANDS = re.compile(
    r"""
    (?:^|\s|[;&|])       # preceded by start, space, or shell chaining
    (?:
        rm\s+-rf          |
        del\s+/[sfq]      |
        format\s+[a-z]:   |
        mkfs              |
        dd\s+if=          |
        shutdown           |
        reboot             |
        halt               |
        init\s+[06]        |
        taskkill\s+/f      |
        reg\s+delete       |
        sfc\s+/scannow     |
        cipher\s+/w        |
        diskpart
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

DANGEROUS_HOTKEYS = frozenset({
    ("alt", "f4"),
    ("ctrl", "alt", "delete"),
    ("ctrl", "alt", "del"),
})

BLOCKED_EXECUTABLES = re.compile(
    r"""
    (?:
        powershell(?:\.exe)?  |
        cmd(?:\.exe)?         |
        bash                  |
        sh\s                  |
        wscript               |
        cscript               |
        mshta                 |
        regsvr32              |
        certutil              |
        bitsadmin
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


class SafetyViolation(Exception):
    """Raised when an action is blocked by the safety system."""

    def __init__(self, reason: str, action: Optional[Action] = None):
        self.reason = reason
        self.action = action
        super().__init__(reason)


class SafetyGuard:
    """Validates actions before they reach the executor."""

    def __init__(
        self,
        max_actions_per_iteration: int = 10,
        block_dangerous: bool = True,
    ):
        self.max_actions_per_iteration = max_actions_per_iteration
        self.block_dangerous = block_dangerous
        self._action_count = 0

    def reset_iteration(self) -> None:
        self._action_count = 0

    def validate(self, action: Action) -> Optional[str]:
        """
        Returns None if safe, or a reason string if the action should be blocked.
        """
        self._action_count += 1

        if self._action_count > self.max_actions_per_iteration:
            return f"Action limit exceeded ({self.max_actions_per_iteration} per iteration)"

        if not self.block_dangerous:
            return None

        if action.type == ActionType.TYPE and action.text:
            if DANGEROUS_COMMANDS.search(action.text):
                return f"Dangerous command detected in text: {action.text[:60]}"
            if BLOCKED_EXECUTABLES.search(action.text):
                return f"Blocked executable in text: {action.text[:60]}"

        if action.type == ActionType.HOTKEY and action.keys:
            normalized = tuple(k.lower().strip() for k in action.keys)
            if normalized in DANGEROUS_HOTKEYS:
                return f"Dangerous hotkey blocked: {'+'.join(action.keys)}"

        return None

    def check_or_raise(self, action: Action) -> None:
        reason = self.validate(action)
        if reason:
            log.warning("SAFETY BLOCK: %s", reason)
            raise SafetyViolation(reason, action)
