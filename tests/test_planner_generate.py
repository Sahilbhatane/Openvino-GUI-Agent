"""Planner.generate_plan with a fake VLM (no OpenVINO)."""

from PIL import Image

from agent.planner import Planner
from models.action_schema import ActionType


class _FakeVLM:
    def __init__(self, raw: str):
        self._raw = raw
        self.last_usage = None

    def analyze_screen(self, screenshot, instruction, elements_text="", history_text="", max_new_tokens=300):
        return self._raw


def test_generate_plan_parses_single_action():
    raw = '{"thought":"open menu","action":{"type":"click","x":5,"y":6},"task_complete":false}'
    planner = Planner(_FakeVLM(raw))
    plan = planner.generate_plan(Image.new("RGB", (10, 10)), "test")
    assert plan.thought == "open menu"
    assert len(plan.actions) == 1
    assert plan.actions[0].type == ActionType.CLICK
    assert plan.actions[0].x == 5


def test_generate_plan_truncates_multiple_actions_to_first():
    raw = (
        '{"thought":"multi","actions":['
        '{"type":"wait","seconds":0.1},'
        '{"type":"press_key","key":"enter"}'
        '],"task_complete":false}'
    )
    planner = Planner(_FakeVLM(raw))
    plan = planner.generate_plan(Image.new("RGB", (10, 10)), "x")
    assert len(plan.actions) == 1
    assert plan.actions[0].type == ActionType.WAIT


def test_parse_response_sanitize_element_dict_extracts_coords():
    raw = '{"thought":"t","action":{"type":"click","element":{"x":9,"y":8}},"task_complete":false}'
    plan = Planner._parse_response(raw)
    assert plan.actions[0].x == 9
    assert plan.actions[0].y == 8
    assert plan.actions[0].element is None


def test_parse_response_double_brace_wrapper():
    """Model sometimes echoes '{{...}}' from older prompt examples."""
    raw = '{{"thought":"open calc","action":{"type":"click","element":25},"task_complete":false}}'
    plan = Planner._parse_response(raw)
    assert plan.thought == "open calc"
    assert len(plan.actions) == 1
    assert plan.actions[0].type == ActionType.CLICK
    assert plan.actions[0].element == 25


def test_parse_response_markdown_fence_double_brace():
    raw = (
        '```json\n'
        '{{"thought":"x","action":{"type":"click","element":3},"task_complete":false}}\n'
        "```"
    )
    plan = Planner._parse_response(raw)
    assert plan.thought == "x"
    assert plan.actions[0].element == 3


def test_generate_plan_accepts_double_brace_from_vlm():
    raw = '{{"thought":"step","action":{"type":"wait","seconds":0.5},"task_complete":false}}'
    planner = Planner(_FakeVLM(raw))
    plan = planner.generate_plan(Image.new("RGB", (10, 10)), "task")
    assert plan.actions[0].type == ActionType.WAIT
    assert plan.actions[0].seconds == 0.5
