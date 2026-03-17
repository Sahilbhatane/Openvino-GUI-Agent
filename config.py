from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ── Model ─────────────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "models" / "Phi-3.5-vision-instruct-int4-ov"
OPENVINO_DEVICE = "CPU"  # CPU, GPU, or AUTO
MAX_NEW_TOKENS = 300

# ── Agent loop ────────────────────────────────────────────
MAX_ACTIONS_PER_STEP = 10
MAX_ITERATIONS = 5
STEP_DELAY_SECONDS = 1.5

# ── Executor safety ──────────────────────────────────────
PYAUTOGUI_FAILSAFE = True  # move mouse to top-left corner to abort
ACTION_DELAY_SECONDS = 0.3

# ── Debug ─────────────────────────────────────────────────
DEBUG_MODE = True
DEBUG_SCREENSHOT_DIR = PROJECT_ROOT / "debug_screenshots"

# ── FastAPI ───────────────────────────────────────────────
API_HOST = "127.0.0.1"
API_PORT = 8000
