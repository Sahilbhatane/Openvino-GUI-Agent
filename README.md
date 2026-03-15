# OpenVINO GUI Agent MVP

A local desktop GUI agent powered by **Phi-3.5 Vision (INT4)** running through
**OpenVINO**. The agent captures your screen, reasons about what it sees, and
executes mouse/keyboard actions to complete natural-language tasks.

## Pipeline

```
User instruction
      |
Screen capture (mss)
      |
Vision-Language Model (OpenVINO)
      |
Structured action plan (JSON)
      |
Executor (pyautogui)
      |
New screenshot --> loop
```

## Quick start

```bash
# 1. Activate the virtual environment
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Linux / macOS

# 2. Start the server (loads the model on startup)
python main.py

# 3. Send a task via curl or any HTTP client
curl -X POST http://127.0.0.1:8000/run-task \
  -H "Content-Type: application/json" \
  -d "{\"instruction\": \"open the calculator\"}"
```

## CLI usage

With the server running in one terminal, open another and run:

```bash
python cli_agent.py "open calculator"
```

The CLI sends the instruction to the server's `/run-task` endpoint and prints
each iteration's reasoning, actions, and results. An optional `--url` flag lets
you point to a different server address:

```bash
python cli_agent.py "enable dark mode" --url http://192.168.1.50:8000
```

If the server is not running you'll see:

```
ERROR: Agent server is not running.
Start it with:  python main.py
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/run-task` | Run a GUI task. Body: `{"instruction": "..."}` |
| GET | `/health` | Check if the model is loaded and ready |

## Project structure

```
main.py                  Entrypoint -- starts FastAPI server
cli_agent.py             CLI client -- send tasks from the terminal
config.py                All tunables (model path, limits, debug)
agent/
  controller.py          Orchestrates the capture-plan-execute loop
  planner.py             VLM call + JSON parsing into ActionPlan
  executor.py            Drives PyAutoGUI from an ActionPlan
vision/
  screen_capture.py      Fast screenshot via mss
  vlm_inference.py       OpenVINO Phi-3.5 Vision wrapper
models/
  action_schema.py       Pydantic schemas (Action, ActionPlan)
api/
  server.py              FastAPI app with /run-task endpoint
utils/
  logger.py              Shared logging setup
```

## Debug mode

Set `DEBUG_MODE = True` in `config.py` (default). Each iteration saves a
screenshot to `debug_screenshots/step_N.png` so you can see exactly what the
model was looking at.

## Safety

- **Failsafe**: move mouse to the top-left corner of the screen to abort at
  any time (PyAutoGUI failsafe).
- **Action cap**: max 10 actions per iteration (configurable).
- **Iteration cap**: max 5 loop iterations per task (configurable).
