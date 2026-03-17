"""
Agent controller -- orchestrates the Vision -> Language -> Action loop.

Pipeline per iteration:
  1. Capture the screen
  2. Analyze accessibility tree to get interactive elements
  3. Annotate screenshot with Set-of-Marks overlay
  4. Send annotated screenshot + element list + instruction to the planner (VLM)
  5. Execute the returned action plan (resolving element IDs to coordinates)
  6. Wait, then repeat until done or max iterations reached
"""

import time
from pathlib import Path

from agent.executor import Executor
from agent.planner import Planner
from agent.screen_analyzer import ScreenAnalyzer, build_element_map, build_elements_text
from vision.screen_capture import capture_screen
from vision.som_overlay import draw_som_overlay
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
        self.screen_analyzer = ScreenAnalyzer()
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
        total_in = total_out = 0
        total_time = 0.0

        while not task_complete and iteration < self.max_iterations:
            iteration += 1
            log.info("--- Iteration %d / %d ---", iteration, self.max_iterations)

            screenshot = capture_screen()

            elements = self.screen_analyzer.analyze()
            elem_map = build_element_map(elements)
            elem_text = build_elements_text(elements)
            log.info("Screen elements: %d detected", len(elements))

            annotated = draw_som_overlay(screenshot, elements)

            if self.debug and self.debug_dir:
                self.debug_dir.mkdir(parents=True, exist_ok=True)
                screenshot.save(self.debug_dir / f"step_{iteration}_raw.png")
                annotated.save(self.debug_dir / f"step_{iteration}_som.png")

            plan = self.planner.generate_plan(annotated, instruction, elements_text=elem_text)

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

            if plan.task_complete:
                log.info("Planner says task is COMPLETE.")
                history.append(
                    {"iteration": iteration, "thought": plan.thought,
                     "task_complete": True, "elements": len(elements), **token_info}
                )
                task_complete = True
                break

            if not plan.actions:
                log.warning("No actions returned and task not complete -- stopping.")
                history.append(
                    {"iteration": iteration, "thought": plan.thought,
                     "actions": [], "elements": len(elements), **token_info}
                )
                break

            results = self.executor.execute(plan, element_map=elem_map)
            history.append(
                {
                    "iteration": iteration,
                    "thought": plan.thought,
                    "actions": [a.model_dump() for a in plan.actions],
                    "results": results,
                    "elements": len(elements),
                    **token_info,
                }
            )

            time.sleep(self.step_delay)

        status = "completed" if task_complete else "max_iterations"
        avg_tps = total_out / total_time if total_time > 0 else 0.0
        log.info(
            "=== Finished (%s, %d steps) | tokens: %d in / %d out  %.1fs  %.1f tok/s ===",
            status, len(history), total_in, total_out, total_time, avg_tps,
        )
        return {
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
