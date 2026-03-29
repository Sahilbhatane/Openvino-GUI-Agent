"""Action and ActionPlan schema behavior."""

import pytest

from models.action_schema import Action, ActionPlan, ActionType


def test_action_summary_click_with_coords():
    a = Action(type=ActionType.CLICK, x=10, y=20)
    assert a.summary() == "click(10, 20)"


def test_action_summary_click_with_element():
    a = Action(type=ActionType.CLICK, element=3, x=100, y=200)
    assert "elem=3" in a.summary()
    assert "100" in a.summary()


def test_action_summary_type_scroll_wait():
    assert "hello" in Action(type=ActionType.TYPE, text="hello").summary()
    assert Action(type=ActionType.SCROLL, amount=-3).summary() == "scroll(-3)"
    assert Action(type=ActionType.WAIT, seconds=1.5).summary() == "wait(1.5s)"


def test_action_summary_keys():
    assert "enter" in Action(type=ActionType.PRESS_KEY, key="enter").summary()
    hk = Action(type=ActionType.HOTKEY, keys=["ctrl", "c"])
    assert "ctrl" in hk.summary() and "c" in hk.summary()


def test_action_plan_defaults():
    p = ActionPlan()
    assert p.thought == ""
    assert p.actions == []
    assert p.task_complete is False


def test_action_plan_roundtrip():
    p = ActionPlan(
        thought="ok",
        actions=[Action(type=ActionType.WAIT, seconds=0.1)],
        task_complete=True,
    )
    d = p.model_dump()
    p2 = ActionPlan(**d)
    assert p2.task_complete and len(p2.actions) == 1
