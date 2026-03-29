"""Executor edge cases: unknown action type, hotkey empty keys."""

from unittest.mock import MagicMock, patch

import pytest

from agent.errors import ErrorCategory
from agent.executor import Executor
from models.action_schema import Action, ActionPlan, ActionType


def test_run_unknown_action_type_returns_execution_error():
    ex = Executor(block_dangerous=False)
    # Bypass safety by using a type the match/case does not handle — use setattr after construction
    bad = Action(type=ActionType.CLICK, x=1, y=1)
    object.__setattr__(bad, "type", "not_an_enum")

    with patch.object(ex, "_run", wraps=ex._run):
        res = ex._run(bad)
    assert res.success is False
    assert res.error is not None
    assert res.error.category == ErrorCategory.EXECUTION


def test_hotkey_empty_keys_still_calls_pyautogui():
    ex = Executor(block_dangerous=False)
    action = Action(type=ActionType.HOTKEY, keys=[])
    with patch("agent.executor.pyautogui.hotkey", MagicMock()) as hk:
        res = ex._run(action)
    hk.assert_called_once_with()
    assert res.success
