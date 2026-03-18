"""
Executor: takes an ActionPlan and drives the desktop via PyAutoGUI.

Safety:
  - Maximum actions per step is capped (default 10).
  - pyautogui.FAILSAFE is enabled (move mouse to top-left to abort).
  - Short pause between every action.
"""

import time

import pyautogui

from models.action_schema import Action, ActionPlan, ActionType
from utils.logger import get_logger

log = get_logger("executor")


class Executor:
    def __init__(
        self,
        max_actions: int = 10,
        action_delay: float = 0.3,
        failsafe: bool = True,
        cursor_move_duration: float = 0.2,
    ):
        self.max_actions = max_actions
        self.action_delay = action_delay
        self.cursor_move_duration = cursor_move_duration
        pyautogui.FAILSAFE = failsafe
        pyautogui.PAUSE = action_delay

    def execute(
        self,
        plan: ActionPlan,
        element_map: dict[int, tuple[int, int]] | None = None,
    ) -> list[str]:
        """Run each action in the plan, return a log line per action."""
        actions = plan.actions[: self.max_actions]
        if len(plan.actions) > self.max_actions:
            log.warning(
                "Plan has %d actions, capping to %d",
                len(plan.actions),
                self.max_actions,
            )

        results: list[str] = []
        for idx, action in enumerate(actions, 1):
            resolved = self._resolve_element(action, element_map)
            result = self._run(resolved)
            log.info("  [%d/%d] %s", idx, len(actions), result)
            results.append(result)
            time.sleep(self.action_delay)
        return results

    def execute_single(
        self,
        action: Action,
        element_map: dict[int, tuple[int, int]] | None = None,
    ) -> str:
        """Resolve and execute a single action. Returns a result string."""
        resolved = self._resolve_element(action, element_map)
        result = self._run(resolved)
        log.info("  %s", result)
        return result

    # ── element resolution ─────────────────────────────────

    @staticmethod
    def _resolve_element(
        action: Action,
        element_map: dict[int, tuple[int, int]] | None,
    ) -> Action:
        """If the action references an element ID, resolve it to x,y coords."""
        if action.element is None or element_map is None:
            return action
        coords = element_map.get(action.element)
        if coords is None:
            log.warning("Element %d not found in map, keeping raw coords", action.element)
            return action
        return action.model_copy(update={"x": coords[0], "y": coords[1]})

    # ── individual action dispatch ────────────────────────

    def _run(self, action: Action) -> str:
        try:
            match action.type:
                case ActionType.CLICK:
                    pyautogui.moveTo(action.x, action.y, duration=self.cursor_move_duration)
                    pyautogui.click()
                    return f"click({action.x}, {action.y})"

                case ActionType.DOUBLE_CLICK:
                    pyautogui.moveTo(action.x, action.y, duration=self.cursor_move_duration)
                    pyautogui.doubleClick()
                    return f"double_click({action.x}, {action.y})"

                case ActionType.TYPE:
                    if action.x is not None and action.y is not None:
                        pyautogui.moveTo(action.x, action.y, duration=self.cursor_move_duration)
                        pyautogui.click()
                    self._safe_type(action.text or "")
                    return f'type("{action.text}")'

                case ActionType.SCROLL:
                    pyautogui.scroll(action.amount or 0)
                    return f"scroll({action.amount})"

                case ActionType.WAIT:
                    time.sleep(action.seconds or 1)
                    return f"wait({action.seconds}s)"

                case ActionType.PRESS_KEY:
                    key = action.key or ""
                    pyautogui.press(key)
                    return f"press_key({key})"

                case ActionType.HOTKEY:
                    keys = action.keys or []
                    pyautogui.hotkey(*keys)
                    return f"hotkey({'+'.join(keys)})"

                case _:
                    return f"unknown action: {action.type}"
        except pyautogui.FailSafeException:
            log.error("FAILSAFE triggered -- aborting")
            raise
        except Exception as exc:
            return f"FAILED {action.summary()}: {exc}"

    @staticmethod
    def _safe_type(text: str) -> None:
        """Type text, falling back to clipboard paste for non-ASCII."""
        if all(ord(c) < 128 for c in text):
            pyautogui.typewrite(text, interval=0.04)
        else:
            import pyperclip
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
