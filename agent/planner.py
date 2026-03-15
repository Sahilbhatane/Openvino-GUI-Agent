"""
Planner: screenshot + instruction  -->  validated ActionPlan.

Sends the image to the VLM, parses the JSON response,
and returns a Pydantic-validated ActionPlan.
"""

import json
import re

from PIL import Image

from models.action_schema import ActionPlan
from utils.logger import get_logger
from vision.vlm_inference import VLMInference

log = get_logger("planner")


class Planner:
    def __init__(self, vlm: VLMInference):
        self.vlm = vlm

    def generate_plan(
        self, screenshot: Image.Image, instruction: str
    ) -> ActionPlan:
        raw = self.vlm.analyze_screen(screenshot, instruction)
        plan = self._parse_response(raw)
        log.info(
            "Plan: thought=%r  actions=%d  task_complete=%s",
            plan.thought[:80],
            len(plan.actions),
            plan.task_complete,
        )
        return plan

    # ── response parsing ──────────────────────────────────

    @staticmethod
    def _parse_response(raw: str) -> ActionPlan:
        """Best-effort extraction of JSON from the model's raw text."""
        json_str = raw

        # Strip markdown code fences the model may wrap around JSON
        fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if fence:
            json_str = fence.group(1)

        # Find the outermost { ... } block
        brace = re.search(r"\{.*\}", json_str, re.DOTALL)
        if brace:
            json_str = brace.group(0)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Model may have been cut off mid-JSON; try to repair
            data = Planner._try_repair_json(json_str)
            if data is None:
                log.warning("JSON parse failed, raw: %s", raw[:300])
                return ActionPlan(
                    thought=f"[parse error] {raw[:200]}",
                    actions=[],
                    task_complete=False,
                )

        # The model sometimes uses "action" instead of "actions"
        if "action" in data and "actions" not in data:
            data["actions"] = data.pop("action")

        # Wrap a single action dict in a list
        if isinstance(data.get("actions"), dict):
            data["actions"] = [data["actions"]]

        # Accept both "done" and "task_complete" from the VLM
        if "done" in data and "task_complete" not in data:
            data["task_complete"] = data.pop("done")

        return ActionPlan(**data)

    @staticmethod
    def _try_repair_json(s: str) -> dict | None:
        """Attempt to close truncated JSON so it can still be parsed."""
        # Try progressively adding closing tokens
        for suffix in ["}", "]}", '"]}', '"}]}']: 
            try:
                return json.loads(s + suffix)
            except json.JSONDecodeError:
                continue
        return None
