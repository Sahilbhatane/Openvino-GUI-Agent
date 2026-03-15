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
    DEBUG_MODE,
    DEBUG_SCREENSHOT_DIR,
    MAX_ACTIONS_PER_STEP,
    MAX_ITERATIONS,
    MODEL_PATH,
    MAX_NEW_TOKENS,
    OPENVINO_DEVICE,
    PYAUTOGUI_FAILSAFE,
    STEP_DELAY_SECONDS,
)
from utils.logger import get_logger
from vision.vlm_inference import VLMInference

log = get_logger("api")

# Shared state populated at startup
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
    )
    _controller = AgentController(
        planner=planner,
        executor=executor,
        max_iterations=MAX_ITERATIONS,
        step_delay=STEP_DELAY_SECONDS,
        debug=DEBUG_MODE,
        debug_dir=DEBUG_SCREENSHOT_DIR,
    )
    log.info("GUI Agent ready.")
    yield
    log.info("Shutting down.")


app = FastAPI(title="OpenVINO GUI Agent MVP", lifespan=lifespan)


# ── request / response schemas ────────────────────────────

class TaskRequest(BaseModel):
    instruction: str


class TaskResponse(BaseModel):
    status: str
    instruction: str
    iterations: int
    history: list


# ── endpoints ─────────────────────────────────────────────

@app.post("/run-task", response_model=TaskResponse, responses={503: {"description": "Agent not yet initialized"}})
async def run_task(req: TaskRequest):
    if _controller is None:
        raise HTTPException(503, "Agent not initialized yet")
    log.info("Received task: %s", req.instruction)
    result = _controller.run(req.instruction)
    return result


@app.get("/health")
async def health():
    return {
        "status": "ready" if _controller else "loading",
        "model": str(MODEL_PATH.name),
        "device": OPENVINO_DEVICE,
    }
