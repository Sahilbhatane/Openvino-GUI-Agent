"""Tests for the executor's element resolution and result structure."""

import pytest
from unittest.mock import patch, MagicMock

from agent.executor import Executor, ExecutionResult
from models.action_schema import Action, ActionType


class TestExecutionResult:
    def test_success_str(self):
        r = ExecutionResult("click(100, 200)", success=True)
        assert str(r) == "click(100, 200)"

    def test_failure_str(self):
        r = ExecutionResult("click(None, None)", success=False)
        assert str(r) == "click(None, None)"
        assert r.success is False

    def test_to_dict_with_error(self):
        from agent.errors import AgentError, ErrorCategory
        err = AgentError(ErrorCategory.GROUNDING, "no coords")
        r = ExecutionResult("click", success=False, error=err)
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"]["category"] == "grounding"


class TestElementResolution:
    def test_resolve_with_valid_element(self):
        action = Action(type=ActionType.CLICK, element=3)
        elem_map = {1: (10, 20), 2: (30, 40), 3: (50, 60)}
        resolved = Executor._resolve_element(action, elem_map)
        assert resolved.x == 50
        assert resolved.y == 60

    def test_resolve_missing_element(self):
        action = Action(type=ActionType.CLICK, element=99)
        elem_map = {1: (10, 20)}
        resolved = Executor._resolve_element(action, elem_map)
        assert resolved.element == 99
        assert resolved.x is None

    def test_resolve_no_element(self):
        action = Action(type=ActionType.CLICK, x=100, y=200)
        resolved = Executor._resolve_element(action, {})
        assert resolved.x == 100

    def test_resolve_no_map(self):
        action = Action(type=ActionType.CLICK, element=1)
        resolved = Executor._resolve_element(action, None)
        assert resolved.element == 1


class TestExecutorSafety:
    @patch("pyautogui.moveTo")
    @patch("pyautogui.click")
    def test_dangerous_action_blocked(self, mock_click, mock_move):
        executor = Executor(block_dangerous=True)
        action = Action(type=ActionType.TYPE, text="rm -rf /")
        result = executor.execute_single(action)
        assert isinstance(result, ExecutionResult)
        assert "BLOCKED" in result.description
        assert result.success is False

    @patch("pyautogui.moveTo")
    @patch("pyautogui.click")
    def test_safe_click_executes(self, mock_click, mock_move):
        executor = Executor(block_dangerous=True)
        action = Action(type=ActionType.CLICK, x=100, y=200)
        result = executor.execute_single(action)
        assert isinstance(result, ExecutionResult)
        assert "click" in result.description
        assert result.success is True
        mock_move.assert_called_once()
        mock_click.assert_called_once()
