"""
Planner: screenshot + instruction  -->  validated ActionPlan.

Sends the image to the VLM, parses the JSON response,
and returns a Pydantic-validated ActionPlan.
"""

import json
import re

from PIL import Image

from models.action_schema import ActionPlan, ActionType
from utils.logger import get_logger
from vision.vlm_inference import VLMInference

log = get_logger("planner")


class Planner:
    def __init__(self, vlm: VLMInference, max_new_tokens: int = 300):
        self.vlm = vlm
        self.max_new_tokens = max_new_tokens

    def generate_plan(
        self,
        screenshot: Image.Image,
        instruction: str,
        elements_text: str = "",
        history_text: str = "",
    ) -> ActionPlan:
        raw = self.vlm.analyze_screen(
            screenshot, instruction,
            elements_text=elements_text,
            history_text=history_text,
            max_new_tokens=self.max_new_tokens,
        )
        plan = self._parse_response(raw)
        if len(plan.actions) > 1:
            log.warning("VLM returned %d actions, taking only the first", len(plan.actions))
            plan = plan.model_copy(update={"actions": plan.actions[:1]})
        log.info(
            "Plan: thought=%r  actions=%d  task_complete=%s",
            plan.thought[:80],
            len(plan.actions),
            plan.task_complete,
        )
        return plan

    @property
    def last_token_usage(self):
        return self.vlm.last_usage

    # ── response parsing ──────────────────────────────────

    @staticmethod
    def _looks_like_action_plan(obj: dict) -> bool:
        """True if *obj* is a top-level plan, not a nested action dict."""
        return any(
            k in obj
            for k in ("thought", "action", "actions", "task_complete", "done")
        )

    @staticmethod
    def _parse_first_json_object(text: str) -> dict | None:
        """Parse the first complete JSON object in *text* (handles leading junk)."""
        dec = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _ = dec.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and Planner._looks_like_action_plan(obj):
                return obj
        return None

    @staticmethod
    def _parse_response(raw: str) -> ActionPlan:
        """Best-effort extraction of JSON from the model's raw text."""
        json_str = raw.strip()

        # Strip markdown code fences the model may wrap around JSON
        fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if fence:
            json_str = fence.group(1).strip()

        # Model sometimes echoes "{{...}}" after seeing brace-heavy examples — peel one layer
        for _ in range(3):
            if len(json_str) >= 4 and json_str.startswith("{{") and json_str.endswith("}}"):
                inner = json_str[1:-1].strip()
                if Planner._parse_first_json_object(inner) is not None:
                    json_str = inner
                    continue
            break

        data = Planner._parse_first_json_object(json_str)
        if data is None:
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                pass
        if data is None:
            data = Planner._try_repair_json(json_str)
        if data is None:
            brace = re.search(r"\{.*\}", json_str, re.DOTALL)
            if brace:
                frag = brace.group(0)
                data = Planner._parse_first_json_object(frag)
                if data is None:
                    data = Planner._try_repair_json(frag)
        if data is None:
            log.warning("JSON parse failed, raw: %s", raw[:300])
            return ActionPlan(
                thought=f"[parse error] {raw[:200]}",
                actions=[],
                task_complete=False,
            )

        valid_types = {t.value for t in ActionType}
        # VLM sometimes returns only the action object, e.g. {"type":"click","element":24}
        if isinstance(data, dict) and data.get("type") in valid_types:
            if not any(
                k in data
                for k in ("thought", "action", "actions", "task_complete", "done")
            ):
                data = {
                    "thought": "",
                    "actions": [dict(data)],
                    "task_complete": False,
                }

        # The model sometimes uses "action" instead of "actions"
        if "action" in data and "actions" not in data:
            data["actions"] = data.pop("action")

        # Wrap a single action dict in a list
        if isinstance(data.get("actions"), dict):
            data["actions"] = [data["actions"]]

        # Accept both "done" and "task_complete" from the VLM
        if "done" in data and "task_complete" not in data:
            data["task_complete"] = data.pop("done")

        # Drop actions whose type the executor doesn't support
        if isinstance(data.get("actions"), list):
            cleaned = []
            for a in data["actions"]:
                if isinstance(a, dict) and a.get("type") not in valid_types:
                    log.warning("Dropping unsupported action type: %r", a.get("type"))
                    continue
                if isinstance(a, dict):
                    a = Planner._sanitize_element_field(a)
                cleaned.append(a)
            data["actions"] = cleaned

        return ActionPlan(**data)

    @staticmethod
    def _sanitize_element_field(action: dict) -> dict:
        """Coerce the 'element' field to int, or extract coords from a hallucinated dict."""
        elem = action.get("element")
        if elem is None:
            return action
        if isinstance(elem, int):
            return action
        if isinstance(elem, str):
            try:
                action["element"] = int(elem)
            except ValueError:
                log.warning("Dropping non-integer element ref: %r", elem)
                action.pop("element", None)
            return action
        if isinstance(elem, dict):
            log.warning("Model returned element as dict, extracting coords: %s", elem)
            if "x" in elem and "y" in elem:
                action.setdefault("x", elem["x"])
                action.setdefault("y", elem["y"])
            action.pop("element", None)
            return action
        action.pop("element", None)
        return action

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
