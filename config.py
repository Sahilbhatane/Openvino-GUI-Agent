from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ── Model ─────────────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "models" / "Phi-3.5-vision-instruct-int4-ov"
# AUTO lets OpenVINO pick (often Intel GPU/NPU when available; not NVIDIA CUDA).
# Check logs after load for EXECUTION_DEVICES, or: python -c "import openvino as ov; print(ov.Core().available_devices)"
OPENVINO_DEVICE = "AUTO"
MAX_NEW_TOKENS = 128      # One JSON plan is short; raise if the model truncates mid-JSON

# ── Agent loop ────────────────────────────────────────────
MAX_ACTIONS_PER_STEP = 10
MAX_ITERATIONS = 10
STEP_DELAY_SECONDS = 1.5

# ── ReAct step-wise loop ─────────────────────────────────
MEMORY_SIZE = 5
SCREEN_CHANGE_THRESHOLD = 0.02
MAX_RETRIES_NO_CHANGE = 2

# ── Executor safety ──────────────────────────────────────
PYAUTOGUI_FAILSAFE = True  # move mouse to top-left corner to abort
ACTION_DELAY_SECONDS = 0.3
CURSOR_MOVE_DURATION = 0.2  # animate cursor before click
HIGHLIGHT_DURATION = 0.3    # show target highlight before executing
MAX_PLAN_RETRIES = 1        # retry VLM if first response has no valid action
BLOCK_DANGEROUS_ACTIONS = True

# ── Screen capture ───────────────────────────────────────
MAX_VLM_DIMENSION = 720   # smaller → faster vision encode; raise toward 960–1280 if clicks miss targets

# ── Debug ─────────────────────────────────────────────────
DEBUG_MODE = False
DEBUG_SCREENSHOT_DIR = PROJECT_ROOT / "debug_screenshots"

# ── FastAPI ───────────────────────────────────────────────
API_HOST = "127.0.0.1"
API_PORT = 8000
