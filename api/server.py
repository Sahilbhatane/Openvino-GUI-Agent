"""
FastAPI server exposing the GUI agent as a REST endpoint.

The VLM model is loaded once during startup via lifespan context.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent.controller import AgentController
from agent.executor import Executor
from agent.planner import Planner
from config import (
    ACTION_DELAY_SECONDS,
    BLOCK_DANGEROUS_ACTIONS,
    CURSOR_MOVE_DURATION,
    DEBUG_MODE,
    DEBUG_SCREENSHOT_DIR,
    HIGHLIGHT_DURATION,
    MAX_ACTIONS_PER_STEP,
    MAX_ITERATIONS,
    MAX_PLAN_RETRIES,
    MEMORY_SIZE,
    MODEL_PATH,
    MAX_NEW_TOKENS,
    MAX_RETRIES_NO_CHANGE,
    OPENVINO_DEVICE,
    PYAUTOGUI_FAILSAFE,
    SCREEN_CHANGE_THRESHOLD,
    STEP_DELAY_SECONDS,
)
from utils.logger import get_logger
from vision.vlm_inference import VLMInference

log = get_logger("api")

_controller: AgentController | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _controller

    log.info("Loading VLM (this may take a minute) ...")
    vlm = VLMInference(MODEL_PATH, device=OPENVINO_DEVICE)
    vlm.load()

    planner = Planner(vlm)
    executor = Executor(
        max_actions=MAX_ACTIONS_PER_STEP,
        action_delay=ACTION_DELAY_SECONDS,
        failsafe=PYAUTOGUI_FAILSAFE,
        cursor_move_duration=CURSOR_MOVE_DURATION,
        block_dangerous=BLOCK_DANGEROUS_ACTIONS,
    )
    _controller = AgentController(
        planner=planner,
        executor=executor,
        max_iterations=MAX_ITERATIONS,
        step_delay=STEP_DELAY_SECONDS,
        memory_size=MEMORY_SIZE,
        change_threshold=SCREEN_CHANGE_THRESHOLD,
        max_retries_no_change=MAX_RETRIES_NO_CHANGE,
        max_plan_retries=MAX_PLAN_RETRIES,
        highlight_duration=HIGHLIGHT_DURATION,
        debug=DEBUG_MODE,
        debug_dir=DEBUG_SCREENSHOT_DIR,
    )
    log.info("GUI Agent ready.")
    yield
    log.info("Shutting down.")


app = FastAPI(title="OpenVINO GUI Agent MVP", lifespan=lifespan)


class TaskRequest(BaseModel):
    instruction: str


class TaskResponse(BaseModel):
    status: str
    instruction: str
    iterations: int
    token_usage: dict = {}
    history: list


@app.post("/run-task", response_model=TaskResponse, responses={503: {"description": "Agent not yet initialized"}})
async def run_task(req: TaskRequest):
    if _controller is None:
        raise HTTPException(503, "Agent not initialized yet")
    log.info("Received task: %s", req.instruction)
    try:
        result = _controller.run(req.instruction)
    except Exception as exc:
        log.exception("Task failed: %s", exc)
        raise HTTPException(500, f"Task execution failed: {exc}")
    return result


@app.get("/health")
async def health():
    return {
        "status": "ready" if _controller else "loading",
        "model": str(MODEL_PATH.name),
        "device": OPENVINO_DEVICE,
    }
