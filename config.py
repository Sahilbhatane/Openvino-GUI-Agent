from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ── Model ─────────────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "models" / "Phi-3.5-vision-instruct-int4-ov"
OPENVINO_DEVICE = "CPU"  # CPU, GPU, or AUTO
MAX_NEW_TOKENS = 300

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
MAX_VLM_DIMENSION = 1280

# ── Debug ─────────────────────────────────────────────────
DEBUG_MODE = True
DEBUG_SCREENSHOT_DIR = PROJECT_ROOT / "debug_screenshots"

# ── FastAPI ───────────────────────────────────────────────
API_HOST = "127.0.0.1"
API_PORT = 8000
