"""
Executor: takes an Action and drives the desktop via PyAutoGUI.

Features:
  - Element ID -> coordinate resolution
  - Fallback strategies (element click -> coordinate -> keyboard nav)
  - Safety validation before every action
  - OS-aware text input
"""

import sys
import time

import pyautogui

from agent.errors import AgentError, ErrorCategory
from agent.safety import SafetyGuard
from models.action_schema import Action, ActionPlan, ActionType
from utils.logger import get_logger

log = get_logger("executor")


class ExecutionResult:
    """Result of executing a single action, with structured error info."""

    def __init__(self, description: str, success: bool = True, error: AgentError | None = None):
        self.description = description
        self.success = success
        self.error = error

    def __str__(self) -> str:
        return self.description

    def to_dict(self) -> dict:
        d = {"description": self.description, "success": self.success}
        if self.error:
            d["error"] = self.error.to_dict()
        return d


class Executor:
    def __init__(
        self,
        max_actions: int = 10,
        action_delay: float = 0.3,
        failsafe: bool = True,
        cursor_move_duration: float = 0.2,
        block_dangerous: bool = True,
    ):
        self.max_actions = max_actions
        self.action_delay = action_delay
        self.cursor_move_duration = cursor_move_duration
        self.safety = SafetyGuard(
            max_actions_per_iteration=max_actions,
            block_dangerous=block_dangerous,
        )
        pyautogui.FAILSAFE = failsafe
        pyautogui.PAUSE = action_delay

    def execute(
        self,
        plan: ActionPlan,
        element_map: dict[int, tuple[int, int]] | None = None,
    ) -> list[str]:
        actions = plan.actions[: self.max_actions]
        if len(plan.actions) > self.max_actions:
            log.warning("Plan has %d actions, capping to %d", len(plan.actions), self.max_actions)

        self.safety.reset_iteration()
        results: list[str] = []
        for idx, action in enumerate(actions, 1):
            res = self._execute_with_fallback(action, element_map)
            log.info("  [%d/%d] %s", idx, len(actions), res)
            results.append(str(res))
            time.sleep(self.action_delay)
        return results

    def execute_single(
        self,
        action: Action,
        element_map: dict[int, tuple[int, int]] | None = None,
    ) -> str:
        self.safety.reset_iteration()
        res = self._execute_with_fallback(action, element_map)
        log.info("  %s", res)
        return str(res)

    def _execute_with_fallback(
        self,
        action: Action,
        element_map: dict[int, tuple[int, int]] | None,
    ) -> ExecutionResult:
        """Execute with fallback: element -> coordinate -> keyboard navigation."""
        safety_reason = self.safety.validate(action)
        if safety_reason:
            return ExecutionResult(
                description=f"BLOCKED {action.summary()}",
                success=False,
                error=AgentError(
                    category=ErrorCategory.SAFETY,
                    message=safety_reason,
                    recovery="Action was blocked by safety system",
                ),
            )

        resolved = self._resolve_element(action, element_map)
        result = self._run(resolved)

        if result.success:
            return result

        if action.element is not None and element_map:
            log.info("Primary execution failed, trying coordinate fallback")
            coord_action = resolved.model_copy(update={"element": None})
            if coord_action.x is not None and coord_action.y is not None:
                fallback = self._run(coord_action)
                if fallback.success:
                    fallback.description += " (coord fallback)"
                    return fallback

        if action.type == ActionType.CLICK:
            log.info("Click failed, trying keyboard Tab navigation fallback")
            try:
                pyautogui.press("tab")
                time.sleep(0.2)
                pyautogui.press("enter")
                return ExecutionResult(
                    description=f"keyboard_nav_fallback for {action.summary()}",
                    success=True,
                )
            except Exception as exc:
                log.warning("Keyboard fallback also failed: %s", exc)

        return result

    @staticmethod
    def _resolve_element(
        action: Action,
        element_map: dict[int, tuple[int, int]] | None,
    ) -> Action:
        if action.element is None or element_map is None:
            return action
        coords = element_map.get(action.element)
        if coords is None:
            log.warning("Element %d not found in map, keeping raw coords", action.element)
            return action
        return action.model_copy(update={"x": coords[0], "y": coords[1]})

    def _run(self, action: Action) -> ExecutionResult:
        try:
            match action.type:
                case ActionType.CLICK:
                    if action.x is None or action.y is None:
                        return ExecutionResult(
                            "click(None, None)",
                            success=False,
                            error=AgentError(
                                category=ErrorCategory.GROUNDING,
                                message="Click target has no coordinates",
                                recovery="Retrying with element resolution",
                            ),
                        )
                    pyautogui.moveTo(action.x, action.y, duration=self.cursor_move_duration)
                    pyautogui.click()
                    return ExecutionResult(f"click({action.x}, {action.y})")

                case ActionType.DOUBLE_CLICK:
                    if action.x is None or action.y is None:
                        return ExecutionResult(
                            "double_click(None, None)",
                            success=False,
                            error=AgentError(
                                category=ErrorCategory.GROUNDING,
                                message="Double-click target has no coordinates",
                            ),
                        )
                    pyautogui.moveTo(action.x, action.y, duration=self.cursor_move_duration)
                    pyautogui.doubleClick()
                    return ExecutionResult(f"double_click({action.x}, {action.y})")

                case ActionType.TYPE:
                    if action.x is not None and action.y is not None:
                        pyautogui.moveTo(action.x, action.y, duration=self.cursor_move_duration)
                        pyautogui.click()
                    self._safe_type(action.text or "")
                    return ExecutionResult(f'type("{action.text}")')

                case ActionType.SCROLL:
                    pyautogui.scroll(action.amount or 0)
                    return ExecutionResult(f"scroll({action.amount})")

                case ActionType.WAIT:
                    time.sleep(action.seconds or 1)
                    return ExecutionResult(f"wait({action.seconds}s)")

                case ActionType.PRESS_KEY:
                    key = action.key or ""
                    pyautogui.press(key)
                    return ExecutionResult(f"press_key({key})")

                case ActionType.HOTKEY:
                    keys = action.keys or []
                    pyautogui.hotkey(*keys)
                    return ExecutionResult(f"hotkey({'+'.join(keys)})")

                case _:
                    return ExecutionResult(
                        f"unknown action: {action.type}",
                        success=False,
                        error=AgentError(
                            category=ErrorCategory.EXECUTION,
                            message=f"Unknown action type: {action.type}",
                        ),
                    )

        except pyautogui.FailSafeException:
            log.error("FAILSAFE triggered -- aborting")
            raise
        except Exception as exc:
            return ExecutionResult(
                description=f"{action.summary()}: {exc}",
                success=False,
                error=AgentError(
                    category=ErrorCategory.EXECUTION,
                    message=str(exc),
                    recovery="Will retry with alternative strategy",
                    detail=f"Action: {action.summary()}",
                ),
            )

    @staticmethod
    def _safe_type(text: str) -> None:
        if all(ord(c) < 128 for c in text):
            pyautogui.typewrite(text, interval=0.04)
        else:
            import pyperclip
            pyperclip.copy(text)
            paste_key = "command" if sys.platform == "darwin" else "ctrl"
            pyautogui.hotkey(paste_key, "v")
