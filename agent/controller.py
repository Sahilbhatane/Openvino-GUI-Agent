"""
Agent controller -- step-wise ReAct loop with event-driven execution.

Pipeline per step:
  1. Capture the screen
  2. Compare with previous screenshot (event-driven change detection)
  3. Analyze accessibility tree for interactive elements
  4. Annotate screenshot with Set-of-Marks overlay
  5. Build history context from short-term memory
  6. [PLANNING] Send annotated screenshot + elements + history + instruction to VLM
  7. Retry VLM once if no valid action returned
  8. [EXECUTING] Show target highlight, pause, then execute ONE action
  9. [WAITING] Update memory, wait, repeat
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from agent.executor import Executor
from agent.planner import Planner
from agent.screen_analyzer import ScreenAnalyzer, build_element_map, build_elements_text
from vision.screen_capture import capture_screen
from vision.som_overlay import draw_som_overlay, draw_action_overlay
from utils.logger import get_logger

log = get_logger("controller")


# ── Agent status ─────────────────────────────────────────

class AgentStatus(str, Enum):
    IDLE = "IDLE"
    OBSERVING = "OBSERVING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    WAITING = "WAITING"
    DONE = "DONE"
    FAILED = "FAILED"


# ── Callback protocol ────────────────────────────────────

@dataclass
class StepCallbacks:
    """Optional callbacks the GUI (or any consumer) can register."""
    on_step_start: Callable[[int, int], None] | None = None
    on_screenshot: Callable[[Image.Image, Image.Image], None] | None = None
    on_thought: Callable[[str], None] | None = None
    on_action: Callable[[str, str], None] | None = None
    on_step_complete: Callable[[dict], None] | None = None
    on_task_complete: Callable[[dict], None] | None = None
    on_error: Callable[[str], None] | None = None
    on_status_change: Callable[[str], None] | None = None


# ── Short-term memory ────────────────────────────────────

@dataclass
class StepMemory:
    step: int
    thought: str
    action_desc: str
    result: str
    screen_changed: bool


def _format_memory(memory: deque) -> str:
    """Serialize recent steps into text the VLM can consume."""
    if not memory:
        return ""
    lines = ["PREVIOUS STEPS:"]
    for m in memory:
        changed = "screen changed" if m.screen_changed else "no visible change"
        lines.append(
            f"[Step {m.step}] Thought: \"{m.thought}\" "
            f"| Action: {m.action_desc} | Result: {m.result} ({changed})"
        )
    return "\n".join(lines)


# ── Screen change detection ──────────────────────────────

def _compute_screen_diff(img1: Image.Image, img2: Image.Image) -> float:
    """Return 0.0 (identical) to 1.0 (completely different)."""
    size = (128, 128)
    a1 = np.array(img1.resize(size), dtype=np.float32)
    a2 = np.array(img2.resize(size), dtype=np.float32)
    return float(np.abs(a1 - a2).mean() / 255.0)


# ── Controller ───────────────────────────────────────────

class AgentController:
    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        max_iterations: int = 10,
        step_delay: float = 1.5,
        memory_size: int = 5,
        change_threshold: float = 0.02,
        max_retries_no_change: int = 2,
        max_plan_retries: int = 1,
        highlight_duration: float = 0.3,
        debug: bool = False,
        debug_dir: Path | None = None,
    ):
        self.planner = planner
        self.executor = executor
        self.screen_analyzer = ScreenAnalyzer()
        self.max_iterations = max_iterations
        self.step_delay = step_delay
        self.memory_size = memory_size
        self.change_threshold = change_threshold
        self.max_retries_no_change = max_retries_no_change
        self.max_plan_retries = max_plan_retries
        self.highlight_duration = highlight_duration
        self.debug = debug
        self.debug_dir = debug_dir
        self._stop_event = threading.Event()
        self._status = AgentStatus.IDLE

    def stop(self):
        """Signal the agent loop to stop after the current step."""
        self._stop_event.set()

    @property
    def status(self) -> AgentStatus:
        return self._status

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    # ── helpers ───────────────────────────────────────────

    def _emit(self, callbacks: StepCallbacks | None, name: str, *args):
        if callbacks is None:
            return
        fn = getattr(callbacks, name, None)
        if fn is not None:
            try:
                fn(*args)
            except Exception as exc:
                log.warning("Callback %s raised: %s", name, exc)

    def _set_status(self, status: AgentStatus, callbacks: StepCallbacks | None):
        self._status = status
        log.info("Status -> %s", status.value)
        self._emit(callbacks, "on_status_change", status.value)

    # ── planning with retry ───────────────────────────────

    def _plan_with_retry(self, annotated, instruction, elem_text, history_text, callbacks):
        """Call the VLM planner. If no valid action, retry once with a hint."""
        plan = self.planner.generate_plan(
            annotated, instruction,
            elements_text=elem_text,
            history_text=history_text,
        )

        if plan.task_complete or plan.actions:
            return plan

        for attempt in range(1, self.max_plan_retries + 1):
            log.info("No valid action — retry %d/%d", attempt, self.max_plan_retries)
            self._emit(callbacks, "on_error",
                       f"No valid action from VLM — retrying ({attempt}/{self.max_plan_retries})")
            retry_hint = (
                history_text +
                "\n\nHINT: Your previous response contained no valid action. "
                "Look at the numbered elements on screen carefully and "
                "choose exactly one action to perform."
            )
            plan = self.planner.generate_plan(
                annotated, instruction,
                elements_text=elem_text,
                history_text=retry_hint,
            )
            if plan.task_complete or plan.actions:
                return plan

        return plan

    # ── main loop ─────────────────────────────────────────

    def run(self, instruction: str, callbacks: StepCallbacks | None = None) -> dict:
        self._stop_event.clear()
        self._set_status(AgentStatus.IDLE, callbacks)
        log.info("=== Task: %s ===", instruction)

        memory: deque[StepMemory] = deque(maxlen=self.memory_size)
        history: list[dict] = []
        prev_screenshot: Image.Image | None = None
        no_change_count = 0
        task_complete = False
        total_in = total_out = 0
        total_time = 0.0

        for step in range(1, self.max_iterations + 1):
            if self._stop_event.is_set():
                log.info("Stop requested — aborting.")
                break

            log.info("--- Step %d / %d ---", step, self.max_iterations)
            self._emit(callbacks, "on_step_start", step, self.max_iterations)

            # ── 1. Observe ────────────────────────────────
            self._set_status(AgentStatus.OBSERVING, callbacks)
            screenshot = capture_screen()
            elements = self.screen_analyzer.analyze()
            elem_map = build_element_map(elements)
            elem_text = build_elements_text(elements)
            log.info("Screen elements: %d detected", len(elements))

            annotated = draw_som_overlay(screenshot, elements)
            self._emit(callbacks, "on_screenshot", screenshot, annotated)

            if self.debug and self.debug_dir:
                self.debug_dir.mkdir(parents=True, exist_ok=True)
                screenshot.save(self.debug_dir / f"step_{step}_raw.png")
                annotated.save(self.debug_dir / f"step_{step}_som.png")

            # ── 2. Event-driven change detection ──────────
            screen_changed = True
            if prev_screenshot is not None:
                diff = _compute_screen_diff(prev_screenshot, screenshot)
                screen_changed = diff >= self.change_threshold
                if not screen_changed:
                    no_change_count += 1
                    log.info("Screen diff=%.4f (below threshold) — no_change streak: %d", diff, no_change_count)
                else:
                    no_change_count = 0
                    log.info("Screen diff=%.4f — change detected", diff)

            # ── 3. Build history context ──────────────────
            history_text = _format_memory(memory)
            if no_change_count > 0:
                history_text += (
                    f"\n\nWARNING: The screen has NOT changed for "
                    f"{no_change_count} consecutive step(s). "
                    f"Your previous action may not have worked. Try a different approach."
                )

            # ── 4. VLM reasoning (with retry) ─────────────
            self._set_status(AgentStatus.PLANNING, callbacks)
            plan = self._plan_with_retry(
                annotated, instruction, elem_text, history_text, callbacks,
            )

            usage = self.planner.last_token_usage
            token_info = {}
            if usage:
                token_info = {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "generation_time": round(usage.generation_time, 2),
                    "tokens_per_second": round(usage.tokens_per_second, 1),
                }
                total_in += usage.input_tokens
                total_out += usage.output_tokens
                total_time += usage.generation_time

            self._emit(callbacks, "on_thought", plan.thought)

            # ── 5. Check completion ───────────────────────
            if plan.task_complete:
                log.info("Planner says task is COMPLETE.")
                self._set_status(AgentStatus.DONE, callbacks)
                step_data = {
                    "step": step, "thought": plan.thought,
                    "task_complete": True, "elements": len(elements),
                    **token_info,
                }
                history.append(step_data)
                self._emit(callbacks, "on_step_complete", step_data)
                task_complete = True
                break

            if not plan.actions:
                log.warning("No action returned after retries — stopping.")
                self._set_status(AgentStatus.FAILED, callbacks)
                step_data = {
                    "step": step, "thought": plan.thought,
                    "actions": [], "elements": len(elements),
                    **token_info,
                }
                history.append(step_data)
                self._emit(callbacks, "on_step_complete", step_data)
                break

            # ── 6. Pre-execution highlight ────────────────
            self._set_status(AgentStatus.EXECUTING, callbacks)
            action = plan.actions[0]
            action_desc = action.summary()

            action_preview = draw_action_overlay(
                screenshot, elements,
                target_element_id=action.element,
                action_text=f"Step {step}: {action_desc}",
            )
            self._emit(callbacks, "on_screenshot", screenshot, action_preview)
            time.sleep(self.highlight_duration)

            # ── 7. Execute ONE action ─────────────────────
            result = self.executor.execute_single(action, element_map=elem_map)
            self._emit(callbacks, "on_action", action_desc, result)

            # ── 8. Post-execution overlay ─────────────────
            action_done_overlay = draw_action_overlay(
                screenshot, elements,
                target_element_id=action.element,
                action_text=f"Step {step}: {result}",
            )
            self._emit(callbacks, "on_screenshot", screenshot, action_done_overlay)

            # ── 9. Update memory ──────────────────────────
            self._set_status(AgentStatus.WAITING, callbacks)
            memory.append(StepMemory(
                step=step,
                thought=plan.thought,
                action_desc=action_desc,
                result=result,
                screen_changed=screen_changed,
            ))

            step_data = {
                "step": step,
                "thought": plan.thought,
                "actions": [action.model_dump()],
                "results": [result],
                "elements": len(elements),
                "screen_changed": screen_changed,
                **token_info,
            }
            history.append(step_data)
            self._emit(callbacks, "on_step_complete", step_data)

            # ── 10. Save reference screenshot ─────────────
            prev_screenshot = screenshot

            # ── 11. Bail if stuck ─────────────────────────
            if no_change_count >= self.max_retries_no_change:
                log.warning(
                    "Screen unchanged for %d steps — agent may be stuck.",
                    no_change_count,
                )

            time.sleep(self.step_delay)

        # ── Build result ──────────────────────────────────
        if task_complete:
            status = "completed"
        elif self._stop_event.is_set():
            status = "stopped"
        else:
            status = "max_iterations"

        if not task_complete:
            final = AgentStatus.DONE if self._stop_event.is_set() else AgentStatus.FAILED
            self._set_status(final, callbacks)

        avg_tps = total_out / total_time if total_time > 0 else 0.0
        log.info(
            "=== Finished (%s, %d steps) | tokens: %d in / %d out  %.1fs  %.1f tok/s ===",
            status, len(history), total_in, total_out, total_time, avg_tps,
        )

        result = {
            "instruction": instruction,
            "status": status,
            "iterations": len(history),
            "token_usage": {
                "total_input_tokens": total_in,
                "total_output_tokens": total_out,
                "total_generation_time": round(total_time, 2),
                "avg_tokens_per_second": round(avg_tps, 1),
            },
            "history": history,
        }
        self._emit(callbacks, "on_task_complete", result)
        self._set_status(AgentStatus.IDLE, callbacks)
        return result
