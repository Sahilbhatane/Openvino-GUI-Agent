# OpenVINO GUI Agent

A cross-platform desktop GUI agent powered by a local Vision-Language Model (Phi-3.5 Vision INT4 via OpenVINO). The agent observes the screen, reasons about it, and executes real OS-level actions to complete user instructions.

## Architecture

The agent follows a strict **Perception -> Planning -> Action -> Feedback** loop:

```
User Instruction
      |
      v
[1. Screen Capture] -----> mss (cross-platform)
      |
[2. Accessibility]  -----> pywinauto (Win) / AT-SPI (Linux) / AXUIElement (macOS)
      |
[3. Grounding]      -----> Set-of-Marks overlay (numbered badges on screenshot)
      |
[4. VLM Planning]   -----> OpenVINO Phi-3.5 Vision INT4 -> JSON action plan
      |
[5. Safety Check]   -----> Dangerous action blocking, action limits
      |
[6. Execution]      -----> PyAutoGUI with fallback strategies
      |
[7. Feedback]       -----> Screen diff detection, retry if stuck
      |
      v
  Loop or Done
```

## Features

- **Cross-platform**: Windows, Linux, macOS accessibility backends
- **Local inference**: Runs entirely on CPU via OpenVINO (no cloud API needed)
- **Observable**: 3-panel UI shows live screen, agent reasoning, and control metrics
- **Safety system**: Blocks dangerous commands (rm -rf, format, shutdown), hotkeys (Alt+F4), and executables (powershell, cmd)
- **Structured errors**: Every error is categorized (model/execution/os/grounding/safety) with recovery info
- **Fallback strategies**: Element click -> coordinate fallback -> keyboard navigation
- **Debug mode**: Saves raw/annotated screenshots, JSON plans, and action logs per iteration

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the model

```bash
huggingface-cli download OpenVINO/Phi-3.5-vision-instruct-int4-ov --local-dir models/Phi-3.5-vision-instruct-int4-ov
```

### 3. Launch

**Desktop GUI** (recommended):
```bash
python gui_app.py
```

**API server**:
```bash
python main.py
# Then send tasks via HTTP:
curl -X POST http://localhost:8000/run-task -H "Content-Type: application/json" -d '{"instruction": "open calculator"}'
```

**CLI client**:
```bash
python cli_agent.py "open calculator and compute 42 x 7"
```

## UI Layout

| Panel | Content |
|-------|---------|
| **Left** | Instruction input, Run/Stop buttons, iteration counter, status, token usage, speed, UI element count, debug toggle |
| **Center** | Live screenshot stream with Set-of-Marks overlay and target element highlighting |
| **Right** | Per-iteration reasoning: Thought, Action Plan (JSON), Executed action, Result, Errors |

## Project Structure

```
OpenVINO GUI-Agent/
├── main.py                    # FastAPI server entry point
├── cli_agent.py               # CLI client
├── gui_app.py                 # PySide6 3-panel desktop UI
├── config.py                  # All tunables (model path, limits, delays)
├── agent/
│   ├── controller.py          # ReAct loop orchestrator
│   ├── planner.py             # VLM -> JSON action plan parser
│   ├── executor.py            # PyAutoGUI execution with fallbacks
│   ├── screen_analyzer.py     # Platform-dispatching accessibility extraction
│   ├── safety.py              # Dangerous action blocking
│   ├── errors.py              # Structured error categories
│   ├── os_actions.py          # Cross-platform OS actions (open_app, etc.)
│   └── platform/
│       ├── __init__.py         # Auto-detects OS, returns correct backend
│       ├── windows.py          # pywinauto UIA backend
│       ├── linux.py            # AT-SPI backend
│       └── macos.py            # AXUIElement backend
├── api/
│   └── server.py              # FastAPI REST endpoint
├── models/
│   └── action_schema.py       # Pydantic action/plan schemas
├── vision/
│   ├── vlm_inference.py       # OpenVINO model loading and inference
│   ├── screen_capture.py      # mss screen capture + downscaling
│   └── som_overlay.py         # Set-of-Marks badge drawing
├── utils/
│   └── logger.py              # Shared logging
├── tests/
│   ├── test_planner.py        # JSON parsing tests (12 cases)
│   ├── test_executor.py       # Element resolution + safety tests
│   ├── test_safety.py         # Dangerous action blocking tests (13 cases)
│   ├── test_screen_analyzer.py # UIElement + builder tests
│   └── test_errors.py         # Error categorization tests
└── requirements.txt
```

## Action Types

| Type | Fields | Description |
|------|--------|-------------|
| `click` | `element` or `x, y` | Click a UI element or coordinate |
| `double_click` | `element` or `x, y` | Double-click |
| `type` | `text`, optional `element` | Type text into an input |
| `scroll` | `amount` | Scroll (positive=up, negative=down) |
| `wait` | `seconds` | Pause execution |
| `press_key` | `key` | Press a single key (enter, tab, escape, etc.) |
| `hotkey` | `keys` | Key combination (e.g. ctrl+c) |

## Safety

- **PyAutoGUI failsafe**: Move mouse to top-left corner to abort
- **Global Stop button**: Immediately halts the agent loop
- **Dangerous command detection**: Blocks rm -rf, format, shutdown, del /s, etc.
- **Dangerous hotkey detection**: Blocks Alt+F4, Ctrl+Alt+Delete
- **Executable blocking**: Blocks powershell, cmd, bash, wscript, etc.
- **Action limit**: Configurable cap per iteration (default: 10)

## Configuration

All tunables are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENVINO_DEVICE` | `CPU` | OpenVINO device (CPU, GPU, AUTO) |
| `MAX_ITERATIONS` | `10` | Maximum steps per task |
| `MAX_NEW_TOKENS` | `300` | VLM generation limit |
| `STEP_DELAY_SECONDS` | `1.5` | Pause between iterations |
| `MEMORY_SIZE` | `5` | Steps kept in short-term memory |
| `SCREEN_CHANGE_THRESHOLD` | `0.02` | Minimum diff to detect screen change |
| `DEBUG_MODE` | `True` | Save debug artifacts per step |
| `BLOCK_DANGEROUS_ACTIONS` | `True` | Enable safety system |

## Platform Requirements

| OS | Accessibility Backend | Install |
|----|----------------------|---------|
| Windows | pywinauto (UIA) | `pip install pywinauto` (included in requirements.txt) |
| Linux | AT-SPI (pyatspi) | `sudo apt install python3-pyatspi` |
| macOS | AXUIElement | `pip install pyobjc-framework-ApplicationServices pyobjc-framework-Quartz pyobjc-framework-Cocoa` |

## Running Tests

```bash
python -m pytest tests/ -v
```

## Debug Output

When `DEBUG_MODE = True`, each iteration saves to `debug_screenshots/`:
- `step_N_raw.png` -- raw screenshot
- `step_N_som.png` -- annotated screenshot with SoM overlay
- `step_N_plan.json` -- VLM action plan
- `step_N_actions.json` -- executed actions and results
