"""
Agent controller -- orchestrates the Vision -> Language -> Action loop.

Pipeline per iteration:
  1. Capture the screen
  2. Send screenshot + instruction to the planner (VLM)
  3. Execute the returned action plan
  4. Wait, then repeat until done or max iterations reached
"""

import time
from pathlib import Path

from agent.executor import Executor
from agent.planner import Planner
from vision.screen_capture import capture_screen
from utils.logger import get_logger

log = get_logger("controller")


class AgentController:
    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        max_iterations: int = 5,
        step_delay: float = 1.5,
        debug: bool = False,
        debug_dir: Path | None = None,
    ):
        self.planner = planner
        self.executor = executor
        self.max_iterations = max_iterations
        self.step_delay = step_delay
        self.debug = debug
        self.debug_dir = debug_dir

    # ── main loop ─────────────────────────────────────────

    def run(self, instruction: str) -> dict:
        log.info("=== Task: %s ===", instruction)
        history: list[dict] = []
        task_complete = False
        iteration = 0

        while not task_complete and iteration < self.max_iterations:
            iteration += 1
            log.info("--- Iteration %d / %d ---", iteration, self.max_iterations)

            screenshot = capture_screen()
            if self.debug and self.debug_dir:
                self.debug_dir.mkdir(parents=True, exist_ok=True)
                screenshot.save(self.debug_dir / f"step_{iteration}.png")

            plan = self.planner.generate_plan(screenshot, instruction)

            if plan.task_complete:
                log.info("Planner says task is COMPLETE.")
                history.append(
                    {"iteration": iteration, "thought": plan.thought, "task_complete": True}
                )
                task_complete = True
                break

            if not plan.actions:
                log.warning("No actions returned and task not complete -- stopping.")
                history.append(
                    {"iteration": iteration, "thought": plan.thought, "actions": []}
                )
                break

            results = self.executor.execute(plan)
            history.append(
                {
                    "iteration": iteration,
                    "thought": plan.thought,
                    "actions": [a.model_dump() for a in plan.actions],
                    "results": results,
                }
            )

            time.sleep(self.step_delay)

        status = "completed" if task_complete else "max_iterations"
        log.info("=== Finished (%s, %d steps) ===", status, len(history))
        return {
            "instruction": instruction,
            "status": status,
            "iterations": len(history),
            "history": history,
        }
