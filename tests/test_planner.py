"""Tests for the planner's JSON parsing logic."""

import pytest

from agent.planner import Planner
from models.action_schema import ActionType


class TestParseResponse:
    def test_valid_json(self):
        raw = '{"thought": "clicking button", "action": {"type": "click", "element": 3}, "task_complete": false}'
        plan = Planner._parse_response(raw)
        assert plan.thought == "clicking button"
        assert len(plan.actions) == 1
        assert plan.actions[0].type == ActionType.CLICK
        assert plan.actions[0].element == 3
        assert plan.task_complete is False

    def test_bare_action_object_wrapped(self):
        """Some VLMs emit only the action dict without thought/action/task_complete."""
        raw = '{"type":"click","element":24}'
        plan = Planner._parse_response(raw)
        assert len(plan.actions) == 1
        assert plan.actions[0].type == ActionType.CLICK
        assert plan.actions[0].element == 24

    def test_json_in_code_fence(self):
        raw = '```json\n{"thought": "typing text", "actions": [{"type": "type", "text": "hello", "element": 5}], "task_complete": false}\n```'
        plan = Planner._parse_response(raw)
        assert len(plan.actions) == 1
        assert plan.actions[0].type == ActionType.TYPE
        assert plan.actions[0].text == "hello"

    def test_task_complete(self):
        raw = '{"thought": "all done", "actions": [], "task_complete": true}'
        plan = Planner._parse_response(raw)
        assert plan.task_complete is True
        assert len(plan.actions) == 0

    def test_done_alias(self):
        raw = '{"thought": "finished", "actions": [], "done": true}'
        plan = Planner._parse_response(raw)
        assert plan.task_complete is True

    def test_single_action_dict_wrapped(self):
        raw = '{"thought": "scroll", "actions": {"type": "scroll", "amount": -3}, "task_complete": false}'
        plan = Planner._parse_response(raw)
        assert len(plan.actions) == 1
        assert plan.actions[0].type == ActionType.SCROLL
        assert plan.actions[0].amount == -3

    def test_unknown_action_type_dropped(self):
        raw = '{"thought": "test", "actions": [{"type": "fly", "target": "moon"}, {"type": "click", "element": 1}], "task_complete": false}'
        plan = Planner._parse_response(raw)
        assert len(plan.actions) == 1
        assert plan.actions[0].type == ActionType.CLICK

    def test_element_string_coerced(self):
        raw = '{"thought": "test", "actions": [{"type": "click", "element": "7"}], "task_complete": false}'
        plan = Planner._parse_response(raw)
        assert plan.actions[0].element == 7

    def test_element_dict_extracts_coords(self):
        raw = '{"thought": "test", "actions": [{"type": "click", "element": {"x": 100, "y": 200}}], "task_complete": false}'
        plan = Planner._parse_response(raw)
        assert plan.actions[0].element is None
        assert plan.actions[0].x == 100
        assert plan.actions[0].y == 200

    def test_garbage_returns_empty_plan(self):
        raw = "This is not JSON at all, just random text from the model."
        plan = Planner._parse_response(raw)
        assert len(plan.actions) == 0
        assert "[parse error]" in plan.thought

    def test_truncated_json_repair(self):
        raw = '{"thought": "test", "actions": [{"type": "click", "element": 5}], "task_complete": false'
        plan = Planner._parse_response(raw)
        assert len(plan.actions) == 1

    def test_hotkey_action(self):
        raw = '{"thought": "copy text", "actions": [{"type": "hotkey", "keys": ["ctrl", "c"]}], "task_complete": false}'
        plan = Planner._parse_response(raw)
        assert plan.actions[0].type == ActionType.HOTKEY
        assert plan.actions[0].keys == ["ctrl", "c"]

    def test_press_key_action(self):
        raw = '{"thought": "press enter", "actions": [{"type": "press_key", "key": "enter"}], "task_complete": false}'
        plan = Planner._parse_response(raw)
        assert plan.actions[0].type == ActionType.PRESS_KEY
        assert plan.actions[0].key == "enter"
