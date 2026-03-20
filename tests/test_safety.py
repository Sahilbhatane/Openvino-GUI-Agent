"""Tests for the safety system."""

import pytest

from agent.safety import SafetyGuard, SafetyViolation
from models.action_schema import Action, ActionType


@pytest.fixture
def guard():
    return SafetyGuard(max_actions_per_iteration=3, block_dangerous=True)


class TestSafetyGuard:
    def test_safe_click_allowed(self, guard):
        action = Action(type=ActionType.CLICK, x=100, y=200)
        assert guard.validate(action) is None

    def test_safe_type_allowed(self, guard):
        action = Action(type=ActionType.TYPE, text="hello world")
        assert guard.validate(action) is None

    def test_dangerous_rm_rf_blocked(self, guard):
        action = Action(type=ActionType.TYPE, text="rm -rf /home/user")
        result = guard.validate(action)
        assert result is not None
        assert "Dangerous command" in result

    def test_dangerous_del_blocked(self, guard):
        action = Action(type=ActionType.TYPE, text="del /s /q C:\\Windows")
        result = guard.validate(action)
        assert result is not None

    def test_dangerous_format_blocked(self, guard):
        action = Action(type=ActionType.TYPE, text="format c:")
        result = guard.validate(action)
        assert result is not None

    def test_dangerous_shutdown_blocked(self, guard):
        action = Action(type=ActionType.TYPE, text="shutdown")
        result = guard.validate(action)
        assert result is not None

    def test_dangerous_hotkey_blocked(self, guard):
        action = Action(type=ActionType.HOTKEY, keys=["alt", "f4"])
        result = guard.validate(action)
        assert result is not None
        assert "hotkey" in result.lower()

    def test_safe_hotkey_allowed(self, guard):
        action = Action(type=ActionType.HOTKEY, keys=["ctrl", "c"])
        assert guard.validate(action) is None

    def test_action_limit_enforced(self, guard):
        for _ in range(3):
            action = Action(type=ActionType.CLICK, x=10, y=10)
            guard.validate(action)

        action = Action(type=ActionType.CLICK, x=10, y=10)
        result = guard.validate(action)
        assert result is not None
        assert "limit" in result.lower()

    def test_reset_iteration_clears_count(self, guard):
        for _ in range(3):
            guard.validate(Action(type=ActionType.CLICK, x=10, y=10))
        guard.reset_iteration()
        result = guard.validate(Action(type=ActionType.CLICK, x=10, y=10))
        assert result is None

    def test_check_or_raise(self, guard):
        dangerous = Action(type=ActionType.TYPE, text="rm -rf /")
        with pytest.raises(SafetyViolation):
            guard.check_or_raise(dangerous)

    def test_blocked_executable_powershell(self, guard):
        action = Action(type=ActionType.TYPE, text="powershell.exe -Command Get-Process")
        result = guard.validate(action)
        assert result is not None
        assert "Blocked executable" in result

    def test_block_dangerous_disabled(self):
        guard = SafetyGuard(max_actions_per_iteration=10, block_dangerous=False)
        action = Action(type=ActionType.TYPE, text="rm -rf /")
        assert guard.validate(action) is None
