# OpenVINO GUI Agent

A local desktop GUI agent powered by **Phi-3.5 Vision (INT4)** running through
**OpenVINO**. The agent captures your screen, identifies interactive UI elements
via the Windows accessibility tree, and executes mouse/keyboard actions to
complete natural-language tasks -- all running locally on your machine.

## Architecture

```
User instruction
      │
Screen capture (mss)
      │
Accessibility tree (pywinauto)
      │          │
Element list   Set-of-Marks overlay
      │          │
      ├──────────┘
      │
Vision-Language Model (OpenVINO)
      │
Structured action plan (JSON)
      │
Executor (pyautogui) ── resolves element IDs → coordinates
      │
New screenshot → loop
```

Each iteration the agent:
1. Captures a screenshot
2. Walks the Windows UI Automation tree to find interactive elements (buttons, text fields, links, etc.)
3. Draws numbered red badges on the screenshot (Set-of-Marks overlay)
4. Sends the annotated screenshot + element list + instruction to the VLM
5. Parses the VLM's JSON action plan (which references elements by number)
6. Executes actions via PyAutoGUI, resolving element IDs to screen coordinates
7. Repeats until the task is complete or the iteration limit is reached

## Quick start

```bash
# 1. Activate the virtual environment
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server (loads the model on startup)
python main.py

# 4. Send a task via curl or any HTTP client
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
each iteration's reasoning, actions, results, and **token usage stats**:

```
Status:     max_iterations
Iterations: 5
Tokens:     9385 in / 492 out  (509.3s, 1.0 tok/s)

--- Iteration 1 ---  [1877 in / 86 out  123.09s  0.7 tok/s]
  Thought: To open the calculator, I need to find the appropriate button...
  Action: click  ->  click(712, 1050)
```

An optional `--url` flag lets you point to a different server address:

```bash
python cli_agent.py "enable dark mode" --url http://192.168.1.50:8000
```

## Endpoints

| Method | Path        | Description                                    |
| ------ | ----------- | ---------------------------------------------- |
| POST   | `/run-task` | Run a GUI task. Body: `{"instruction": "..."}` |
| GET    | `/health`   | Check if the model is loaded and ready         |

## Action types

The agent supports 7 action types:

| Action         | JSON format                                    | Description                       |
| -------------- | ---------------------------------------------- | --------------------------------- |
| `click`        | `{"type":"click","element":3}`                 | Click an element by SoM ID        |
| `double_click` | `{"type":"double_click","element":3}`          | Double-click an element            |
| `type`         | `{"type":"type","text":"hello","element":3}`   | Type text into an element          |
| `scroll`       | `{"type":"scroll","amount":-3}`                | Scroll (positive=up, negative=down)|
| `wait`         | `{"type":"wait","seconds":2}`                  | Wait before next action            |
| `press_key`    | `{"type":"press_key","key":"enter"}`           | Press a single key                 |
| `hotkey`       | `{"type":"hotkey","keys":["ctrl","c"]}`        | Press a key combination            |

Actions can reference elements by numbered ID (from the accessibility tree) or
fall back to raw `x`/`y` pixel coordinates.

## Project structure

```
main.py                  Entrypoint -- starts FastAPI server
cli_agent.py             CLI client -- send tasks from the terminal
config.py                All tunables (model path, limits, debug)

agent/
  controller.py          Orchestrates the capture → analyze → plan → execute loop
  planner.py             VLM call + JSON parsing into ActionPlan
  executor.py            Drives PyAutoGUI, resolves element IDs to coordinates
  screen_analyzer.py     Walks Windows UIA accessibility tree for interactive elements

vision/
  screen_capture.py      Fast screenshot via mss (downscaled to 1280px max)
  vlm_inference.py       OpenVINO Phi-3.5 Vision wrapper with token usage tracking
  som_overlay.py         Set-of-Marks: draws numbered badges on screenshots

models/
  action_schema.py       Pydantic schemas (Action, ActionPlan, ActionType)

api/
  server.py              FastAPI app with /run-task endpoint

utils/
  logger.py              Shared logging setup
```

## Screen grounding

The agent uses **pywinauto** to read the Windows UI Automation accessibility
tree each iteration. This gives it a structured list of every visible button,
text field, link, menu item, etc. with bounding boxes.

These elements are:
- **Numbered** sequentially (1, 2, 3, ...)
- **Overlaid** on the screenshot as red badges (Set-of-Marks)
- **Listed** as text in the VLM prompt so the model can reference them by ID

This means the model says `"element": 1` (click the Start button) instead of
guessing raw pixel coordinates.

## Debug mode

Set `DEBUG_MODE = True` in `config.py` (default). Each iteration saves:
- `debug_screenshots/step_N_raw.png` -- the original screenshot
- `debug_screenshots/step_N_som.png` -- the annotated screenshot with SoM badges

## Configuration

Key settings in `config.py`:

| Setting              | Default | Description                          |
| -------------------- | ------- | ------------------------------------ |
| `OPENVINO_DEVICE`    | `CPU`   | OpenVINO device (`CPU`, `GPU`, `AUTO`) |
| `MAX_NEW_TOKENS`     | `300`   | Max tokens the VLM generates per call |
| `MAX_ITERATIONS`     | `5`     | Max loop iterations per task          |
| `MAX_ACTIONS_PER_STEP` | `10` | Max actions the executor runs per step |
| `DEBUG_MODE`         | `True`  | Save debug screenshots each iteration |

## Safety

- **Failsafe**: move mouse to the top-left corner of the screen to abort at
  any time (PyAutoGUI failsafe).
- **Action cap**: max 10 actions per iteration (configurable).
- **Iteration cap**: max 5 loop iterations per task (configurable).
- **Error handling**: invalid VLM responses (unknown action types, malformed
  JSON, hallucinated element formats) are caught and logged gracefully instead
  of crashing the server.
