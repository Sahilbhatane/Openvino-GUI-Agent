"""Tests for execution-failure reflection hints fed back into the VLM context."""

from collections import deque

from agent.controller import StepMemory, _reflection_context


def test_reflection_empty_memory():
    assert _reflection_context(deque()) == ""


def test_reflection_skips_when_last_execution_ok():
    mem = deque(
        [
            StepMemory(
                step=1,
                thought="t",
                action_desc="click 3",
                result="ok",
                screen_changed=True,
                execution_ok=True,
            ),
        ]
    )
    assert _reflection_context(mem) == ""


def test_reflection_after_single_failure():
    mem = deque(
        [
            StepMemory(
                step=1,
                thought="t",
                action_desc="click 99",
                result="BLOCKED click 99",
                screen_changed=False,
                execution_ok=False,
            ),
        ]
    )
    text = _reflection_context(mem)
    assert "REFLECTION (execution failure):" in text
    assert "click 99" in text
    assert "BLOCKED click 99" in text
    assert "consecutive execution failures" not in text


def test_reflection_escalates_after_two_failures():
    mem = deque(
        [
            StepMemory(1, "a", "act1", "fail1", False, execution_ok=False),
            StepMemory(2, "b", "act2", "fail2", False, execution_ok=False),
        ]
    )
    text = _reflection_context(mem)
    assert "2 consecutive execution failures" in text
