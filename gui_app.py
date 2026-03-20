"""
PySide6 desktop GUI for the OpenVINO GUI Agent.

Launch:  python gui_app.py

Three-panel layout:
  LEFT   - User control (instruction, run/stop, iteration counter, status)
  CENTER - Live screenshot with Set-of-Marks overlay and target highlight
  RIGHT  - Agent reasoning log (UL-TARS style: thought, plan, action, result, errors)
"""

import json
import sys
import time

from PySide6.QtCore import Qt, Signal, Slot, QThread, QSize, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QTextCursor, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QLabel,
    QSplitter,
    QStatusBar,
    QCheckBox,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QGroupBox,
)
from PIL import Image

from agent.controller import AgentController, StepCallbacks
from agent.executor import Executor
from agent.planner import Planner
from config import (
    ACTION_DELAY_SECONDS,
    CURSOR_MOVE_DURATION,
    DEBUG_MODE,
    DEBUG_SCREENSHOT_DIR,
    HIGHLIGHT_DURATION,
    MAX_ACTIONS_PER_STEP,
    MAX_ITERATIONS,
    MAX_PLAN_RETRIES,
    MEMORY_SIZE,
    MODEL_PATH,
    OPENVINO_DEVICE,
    PYAUTOGUI_FAILSAFE,
    SCREEN_CHANGE_THRESHOLD,
    MAX_RETRIES_NO_CHANGE,
    STEP_DELAY_SECONDS,
)
from vision.vlm_inference import VLMInference


# ── Theme constants ──────────────────────────────────────

DEEP_NAVY = "#0A2463"
STEEL_BLUE = "#3E92CC"
OFF_WHITE = "#F0F2F5"
DARK_BG = "#071A40"
PANEL_BG = "#0D2B5E"
BORDER_COLOR = "#1A3A7A"
TEXT_PRIMARY = "#E8ECF1"
TEXT_SECONDARY = "#8DA4C7"
TEXT_MUTED = "#5A7BA8"
ACCENT_GREEN = "#4CAF50"
ACCENT_RED = "#E53935"
ACCENT_YELLOW = "#FFC107"
ACCENT_BLUE = "#2196F3"

INITIAL_TOKEN_DISPLAY = "0 / 0"


def pil_to_qpixmap(pil_image: Image.Image) -> QPixmap:
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    data = pil_image.tobytes("raw", "RGB")
    qimg = QImage(data, pil_image.width, pil_image.height,
                  3 * pil_image.width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


STATUS_COLORS = {
    "IDLE":      TEXT_MUTED,
    "LOADING":   ACCENT_YELLOW,
    "OBSERVING": STEEL_BLUE,
    "PLANNING":  ACCENT_BLUE,
    "EXECUTING": "#FF9800",
    "WAITING":   ACCENT_YELLOW,
    "DONE":      ACCENT_GREEN,
    "FAILED":    ACCENT_RED,
}


STYLESHEET = f"""
QMainWindow {{
    background-color: {DARK_BG};
}}
QWidget#central {{
    background-color: {DARK_BG};
}}

/* ── Panels ─────────────────────────────────────── */
QFrame#left_panel, QFrame#center_panel, QFrame#right_panel {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
}}

/* ── Input ──────────────────────────────────────── */
QLineEdit {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 13px;
    selection-background-color: {STEEL_BLUE};
}}
QLineEdit:focus {{
    border-color: {STEEL_BLUE};
}}
QLineEdit:disabled {{
    color: {TEXT_MUTED};
}}

/* ── Buttons ────────────────────────────────────── */
QPushButton {{
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: bold;
}}
QPushButton#run_btn {{
    background-color: {ACCENT_GREEN};
    color: #ffffff;
}}
QPushButton#run_btn:hover {{
    background-color: #66BB6A;
}}
QPushButton#run_btn:pressed {{
    background-color: #388E3C;
}}
QPushButton#stop_btn {{
    background-color: {ACCENT_RED};
    color: #ffffff;
}}
QPushButton#stop_btn:hover {{
    background-color: #EF5350;
}}
QPushButton#stop_btn:pressed {{
    background-color: #C62828;
}}
QPushButton:disabled {{
    background-color: {BORDER_COLOR};
    color: {TEXT_MUTED};
}}

/* ── Text panels ────────────────────────────────── */
QTextEdit {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 8px;
    font-size: 12px;
}}

/* ── Screenshot label ───────────────────────────── */
QLabel#screenshot_label {{
    background-color: {DARK_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
}}

/* ── Status bar ─────────────────────────────────── */
QStatusBar {{
    background-color: {DARK_BG};
    color: {TEXT_MUTED};
    font-size: 11px;
    padding: 2px 8px;
    border-top: 1px solid {BORDER_COLOR};
}}

/* ── Splitter ───────────────────────────────────── */
QSplitter::handle {{
    background-color: {BORDER_COLOR};
    width: 2px;
}}

/* ── Section headers ────────────────────────────── */
QLabel#section_header {{
    color: {TEXT_SECONDARY};
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 1px;
    padding: 6px 0px 4px 0px;
}}

/* ── Checkbox ───────────────────────────────────── */
QCheckBox {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid {BORDER_COLOR};
    background-color: {DARK_BG};
}}
QCheckBox::indicator:checked {{
    background-color: {STEEL_BLUE};
    border-color: {STEEL_BLUE};
}}

/* ── Labels ─────────────────────────────────────── */
QLabel#metric_label {{
    color: {TEXT_SECONDARY};
    font-size: 11px;
    padding: 2px 0px;
}}
QLabel#metric_value {{
    color: {TEXT_PRIMARY};
    font-size: 14px;
    font-weight: bold;
    padding: 0px 0px 4px 0px;
}}
QLabel#status_label {{
    font-size: 12px;
    font-weight: bold;
    padding: 4px 8px;
    border-radius: 4px;
}}

/* ── GroupBox ────────────────────────────────────── */
QGroupBox {{
    color: {TEXT_SECONDARY};
    font-size: 10px;
    font-weight: bold;
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 14px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}
"""


# ── Status Indicator Widget ──────────────────────────────

class StatusIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._dot = QLabel()
        self._dot.setFixedSize(10, 10)
        layout.addWidget(self._dot)

        self._label = QLabel("IDLE")
        self._label.setFixedWidth(90)
        layout.addWidget(self._label)

        self.set_status("IDLE")

    def set_status(self, status: str):
        color = STATUS_COLORS.get(status, TEXT_MUTED)
        self._dot.setStyleSheet(
            f"background-color: {color}; border-radius: 5px; border: none;"
        )
        self._label.setText(status)
        self._label.setStyleSheet(
            f"color: {color}; font-size: 12px; font-weight: bold; border: none;"
        )


# ── Model Loader Thread ──────────────────────────────────

class ModelLoader(QThread):
    progress = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def run(self):
        try:
            self.progress.emit("Loading VLM model -- this may take a minute ...")
            vlm = VLMInference(MODEL_PATH, device=OPENVINO_DEVICE)
            vlm.load()
            self.progress.emit("Building agent controller ...")
            planner = Planner(vlm)
            executor = Executor(
                max_actions=MAX_ACTIONS_PER_STEP,
                action_delay=ACTION_DELAY_SECONDS,
                failsafe=PYAUTOGUI_FAILSAFE,
                cursor_move_duration=CURSOR_MOVE_DURATION,
            )
            controller = AgentController(
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
            self.finished.emit(controller)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Agent Worker Thread ──────────────────────────────────

class AgentWorker(QThread):
    step_started = Signal(int, int)
    screenshot_ready = Signal(object, object)
    thought_ready = Signal(str)
    action_done = Signal(str, str)
    plan_ready = Signal(dict)
    step_complete = Signal(dict)
    iteration_detail = Signal(dict)
    task_complete = Signal(dict)
    error_occurred = Signal(str)
    status_changed = Signal(str)

    def __init__(self, controller: AgentController, instruction: str, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.instruction = instruction

    def run(self):
        try:
            callbacks = StepCallbacks(
                on_step_start=self.step_started.emit,
                on_screenshot=self.screenshot_ready.emit,
                on_thought=self.thought_ready.emit,
                on_action=self.action_done.emit,
                on_plan_ready=self.plan_ready.emit,
                on_step_complete=self.step_complete.emit,
                on_iteration_detail=self.iteration_detail.emit,
                on_task_complete=self.task_complete.emit,
                on_error=self.error_occurred.emit,
                on_status_change=self.status_changed.emit,
            )
            self.controller.run(self.instruction, callbacks=callbacks)
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def stop(self):
        self.controller.stop()


# ── Main Window ──────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenVINO GUI Agent")
        self.resize(1440, 860)
        self.setMinimumSize(QSize(1100, 640))

        self._controller: AgentController | None = None
        self._worker: AgentWorker | None = None
        self._current_pixmap: QPixmap | None = None
        self._step_count = 0
        self._total_tokens_in = 0
        self._total_tokens_out = 0
        self._task_start_time = 0.0

        self._build_ui()
        self.setStyleSheet(STYLESHEET)
        self._set_ui_state("loading")
        self._start_model_load()

    # ── UI construction ───────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 4)
        root.setSpacing(8)

        # Title bar
        title_row = QHBoxLayout()
        title_row.setSpacing(12)
        title = QLabel("OpenVINO GUI Agent")
        title.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 16px; font-weight: bold; border: none;"
        )
        title_row.addWidget(title)

        version_label = QLabel("v1.0 MVP")
        version_label.setStyleSheet(
            f"color: {TEXT_MUTED}; font-size: 11px; border: none; padding-top: 3px;"
        )
        title_row.addWidget(version_label)
        title_row.addStretch()

        self.status_indicator = StatusIndicator()
        title_row.addWidget(self.status_indicator)
        root.addLayout(title_row)

        # Main 3-panel splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)

        # LEFT PANEL -- User Control
        left_panel = self._build_left_panel()
        splitter.addWidget(left_panel)

        # CENTER PANEL -- Live Screen
        center_panel = self._build_center_panel()
        splitter.addWidget(center_panel)

        # RIGHT PANEL -- Agent Reasoning
        right_panel = self._build_right_panel()
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([260, 600, 400])

        root.addWidget(splitter, stretch=1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Initializing ...")

    def _build_left_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("left_panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Section header
        header = QLabel("CONTROL")
        header.setObjectName("section_header")
        layout.addWidget(header)

        # Instruction input
        instr_label = QLabel("Instruction")
        instr_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px; border: none;")
        layout.addWidget(instr_label)

        self.instruction_input = QLineEdit()
        self.instruction_input.setPlaceholderText('e.g. "open calculator and compute 42 x 7"')
        self.instruction_input.returnPressed.connect(self._on_run)
        layout.addWidget(self.instruction_input)

        # Run / Stop buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("run_btn")
        self.run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self.stop_btn)

        layout.addLayout(btn_row)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background-color: {BORDER_COLOR}; border: none; max-height: 1px;")
        layout.addWidget(sep)

        # Metrics
        metrics_header = QLabel("METRICS")
        metrics_header.setObjectName("section_header")
        layout.addWidget(metrics_header)

        # Iteration counter
        self.iter_label = QLabel("Iteration")
        self.iter_label.setObjectName("metric_label")
        layout.addWidget(self.iter_label)
        self.iter_value = QLabel(INITIAL_TOKEN_DISPLAY)
        self.iter_value.setObjectName("metric_value")
        layout.addWidget(self.iter_value)

        # Status display
        self.status_label_text = QLabel("Status")
        self.status_label_text.setObjectName("metric_label")
        layout.addWidget(self.status_label_text)
        self.status_display = QLabel("IDLE")
        self.status_display.setObjectName("status_label")
        self.status_display.setStyleSheet(
            f"color: {TEXT_MUTED}; background-color: {DARK_BG}; "
            f"border: 1px solid {BORDER_COLOR}; border-radius: 4px; padding: 4px 8px;"
        )
        layout.addWidget(self.status_display)

        # Token usage
        self.token_label = QLabel("Tokens (in / out)")
        self.token_label.setObjectName("metric_label")
        layout.addWidget(self.token_label)
        self.token_value = QLabel(INITIAL_TOKEN_DISPLAY)
        self.token_value.setObjectName("metric_value")
        layout.addWidget(self.token_value)

        # Generation speed
        self.speed_label = QLabel("Speed")
        self.speed_label.setObjectName("metric_label")
        layout.addWidget(self.speed_label)
        self.speed_value = QLabel("-- tok/s")
        self.speed_value.setObjectName("metric_value")
        layout.addWidget(self.speed_value)

        # Iteration time
        self.time_label = QLabel("Iteration Time")
        self.time_label.setObjectName("metric_label")
        layout.addWidget(self.time_label)
        self.time_value = QLabel("--")
        self.time_value.setObjectName("metric_value")
        layout.addWidget(self.time_value)

        # Elements count
        self.elem_label = QLabel("UI Elements")
        self.elem_label.setObjectName("metric_label")
        layout.addWidget(self.elem_label)
        self.elem_value = QLabel("0")
        self.elem_value.setObjectName("metric_value")
        layout.addWidget(self.elem_value)

        layout.addStretch()

        # Options
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"background-color: {BORDER_COLOR}; border: none; max-height: 1px;")
        layout.addWidget(sep2)

        self.debug_cb = QCheckBox("Debug mode")
        self.debug_cb.setChecked(DEBUG_MODE)
        layout.addWidget(self.debug_cb)

        self.show_reasoning_cb = QCheckBox("Show reasoning")
        self.show_reasoning_cb.setChecked(True)
        layout.addWidget(self.show_reasoning_cb)

        return panel

    def _build_center_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("center_panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        header = QLabel("LIVE SCREEN + GROUNDING")
        header.setObjectName("section_header")
        layout.addWidget(header)

        self.screenshot_label = QLabel()
        self.screenshot_label.setObjectName("screenshot_label")
        self.screenshot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.screenshot_label.setMinimumSize(QSize(400, 300))
        self.screenshot_label.setText("No screenshot yet")
        self.screenshot_label.setStyleSheet(
            self.screenshot_label.styleSheet() + f" color: {TEXT_MUTED};"
        )
        layout.addWidget(self.screenshot_label, stretch=1)

        return panel

    def _build_right_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("right_panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        header = QLabel("AGENT REASONING")
        header.setObjectName("section_header")
        layout.addWidget(header)

        self.reasoning_panel = QTextEdit()
        self.reasoning_panel.setReadOnly(True)
        mono = QFont("Cascadia Code", 10)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.reasoning_panel.setFont(mono)
        layout.addWidget(self.reasoning_panel, stretch=1)

        return panel

    # ── UI state machine ──────────────────────────────────

    def _set_ui_state(self, state: str):
        is_running = state == "running"
        is_idle = state == "idle"

        self.instruction_input.setEnabled(is_idle)
        self.run_btn.setEnabled(is_idle)
        self.stop_btn.setEnabled(is_running)

    # ── Model loading ─────────────────────────────────────

    def _start_model_load(self):
        self.status_indicator.set_status("LOADING")
        self._loader = ModelLoader()
        self._loader.progress.connect(self._on_load_progress)
        self._loader.finished.connect(self._on_model_loaded)
        self._loader.error.connect(self._on_load_error)
        self._loader.start()

    @Slot(str)
    def _on_load_progress(self, msg: str):
        self.status_bar.showMessage(msg)
        self._log_system(msg)

    @Slot(object)
    def _on_model_loaded(self, controller: AgentController):
        self._controller = controller
        self._set_ui_state("idle")
        self.status_indicator.set_status("IDLE")
        self.status_bar.showMessage("Ready -- enter an instruction and press Run")
        self._log_system("Model loaded. Agent ready.")

    @Slot(str)
    def _on_load_error(self, msg: str):
        self.status_indicator.set_status("FAILED")
        self.status_bar.showMessage(f"Load error: {msg}")
        self._log_error("Model", f"Failed to load model: {msg}")

    # ── Run / Stop ────────────────────────────────────────

    @Slot()
    def _on_run(self):
        instruction = self.instruction_input.text().strip()
        if not instruction or self._controller is None:
            return
        self._set_ui_state("running")
        self._step_count = 0
        self._total_tokens_in = 0
        self._total_tokens_out = 0
        self._task_start_time = time.time()
        self.reasoning_panel.clear()
        self.iter_value.setText(f"0 / {MAX_ITERATIONS}")
        self.token_value.setText(INITIAL_TOKEN_DISPLAY)
        self.speed_value.setText("-- tok/s")
        self.time_value.setText("--")
        self.elem_value.setText("0")
        self._log_system(f"Task: {instruction}")
        self.status_bar.showMessage("Running ...")

        self._worker = AgentWorker(self._controller, instruction, parent=self)
        self._worker.step_started.connect(self._on_step_start)
        self._worker.screenshot_ready.connect(self._on_screenshot)
        self._worker.thought_ready.connect(self._on_thought)
        self._worker.action_done.connect(self._on_action)
        self._worker.plan_ready.connect(self._on_plan_ready)
        self._worker.step_complete.connect(self._on_step_complete)
        self._worker.iteration_detail.connect(self._on_iteration_detail)
        self._worker.task_complete.connect(self._on_task_complete)
        self._worker.error_occurred.connect(self._on_agent_error)
        self._worker.status_changed.connect(self._on_status_change)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    @Slot()
    def _on_stop(self):
        if self._worker is not None:
            self._worker.stop()
            self._log_system("Stop requested -- waiting for current step to finish ...")

    # ── Agent event slots ─────────────────────────────────

    @Slot(str)
    def _on_status_change(self, status: str):
        self.status_indicator.set_status(status)
        color = STATUS_COLORS.get(status, TEXT_MUTED)
        self.status_display.setText(status)
        self.status_display.setStyleSheet(
            f"color: {color}; background-color: {DARK_BG}; "
            f"border: 1px solid {color}; border-radius: 4px; padding: 4px 8px;"
        )

    @Slot(int, int)
    def _on_step_start(self, step: int, max_steps: int):
        self._step_count = step
        self.iter_value.setText(f"{step} / {max_steps}")
        self.status_bar.showMessage(f"Step {step} / {max_steps}")
        self._log_step_header(step, max_steps)

    @Slot(object, object)
    def _on_screenshot(self, raw_img: Image.Image, annotated_img: Image.Image):
        pixmap = pil_to_qpixmap(annotated_img)
        self._current_pixmap = pixmap
        self._update_screenshot_label()

    @Slot(str)
    def _on_thought(self, thought: str):
        if self.show_reasoning_cb.isChecked():
            self._log_thought(thought)

    @Slot(dict)
    def _on_plan_ready(self, plan_data: dict):
        if self.show_reasoning_cb.isChecked():
            self._log_plan(plan_data)

    @Slot(str, str)
    def _on_action(self, desc: str, result: str):
        self._log_action(desc, result)

    @Slot(dict)
    def _on_step_complete(self, data: dict):
        tps = data.get("tokens_per_second", 0)
        gen_time = data.get("generation_time", 0)
        elems = data.get("elements", 0)
        changed = data.get("screen_changed", True)
        step = data.get("step", self._step_count)
        iter_time = data.get("iteration_time", 0)
        in_tok = data.get("input_tokens", 0)
        out_tok = data.get("output_tokens", 0)

        self._total_tokens_in += in_tok
        self._total_tokens_out += out_tok

        self.token_value.setText(f"{self._total_tokens_in} / {self._total_tokens_out}")
        self.speed_value.setText(f"{tps} tok/s")
        self.time_value.setText(f"{iter_time}s")
        self.elem_value.setText(str(elems))

        parts = [f"Step {step}", f"{elems} elem", f"{gen_time}s", f"{tps} tok/s"]
        if not changed:
            parts.append("no screen change")
        self.status_bar.showMessage("  |  ".join(parts))

        # Show errors in reasoning panel
        errors = data.get("errors", [])
        for err in errors:
            self._log_error(err.get("category", "unknown"), err.get("message", ""))

    @Slot(dict)
    def _on_iteration_detail(self, data: dict):
        # Reserved for future per-iteration detail rendering
        _ = data

    @Slot(dict)
    def _on_task_complete(self, result: dict):
        status = result.get("status", "unknown")
        steps = result.get("iterations", 0)
        usage = result.get("token_usage", {})
        tps = usage.get("avg_tokens_per_second", 0)
        total_t = usage.get("total_generation_time", 0)
        total_in = usage.get("total_input_tokens", 0)
        total_out = usage.get("total_output_tokens", 0)

        if status == "completed":
            color = ACCENT_GREEN
            label = "COMPLETED"
        elif status == "stopped":
            color = ACCENT_YELLOW
            label = "STOPPED"
        else:
            color = ACCENT_RED
            label = "INCOMPLETE"

        self.token_value.setText(f"{total_in} / {total_out}")

        self._log_html(
            f'<br><div style="border-left: 3px solid {color}; padding: 6px 12px; margin: 8px 0;">'
            f'<p style="color:{color}; font-weight:bold; margin:0;">'
            f'{label} -- {steps} step(s), {total_t}s total, {tps} tok/s avg</p></div>'
        )
        self.status_bar.showMessage(
            f"{label}  |  {steps} steps  |  {total_t}s  |  {tps} tok/s"
        )

    @Slot(str)
    def _on_agent_error(self, msg: str):
        self._log_error("agent", msg)

    @Slot()
    def _on_worker_finished(self):
        self._set_ui_state("idle")

    # ── Log helpers (reasoning panel) ─────────────────────

    def _log_html(self, html: str):
        self.reasoning_panel.moveCursor(QTextCursor.MoveOperation.End)
        self.reasoning_panel.insertHtml(html)
        self.reasoning_panel.moveCursor(QTextCursor.MoveOperation.End)

    def _log_system(self, text: str):
        self._log_html(
            f'<p style="color:{TEXT_MUTED}; margin:2px 0;">{_esc(text)}</p>'
        )

    def _log_error(self, category: str, text: str):
        self._log_html(
            f'<div style="border-left: 3px solid {ACCENT_RED}; padding: 4px 10px; margin: 4px 0;">'
            f'<p style="color:{ACCENT_RED}; margin:0;">'
            f'<b>[{_esc(category.upper())}]</b> {_esc(text)}</p></div>'
        )

    def _log_step_header(self, step: int, max_steps: int):
        self._log_html(
            f'<div style="border-top: 1px solid {BORDER_COLOR}; margin-top: 10px; padding-top: 8px;">'
            f'<p style="color:{STEEL_BLUE}; font-weight:bold; margin:0;">'
            f'Iteration {step} / {max_steps}</p></div>'
        )

    def _log_thought(self, thought: str):
        self._log_html(
            f'<div style="margin: 4px 0 4px 8px;">'
            f'<p style="color:{TEXT_SECONDARY}; margin:0 0 2px 0; font-size:10px;">THOUGHT</p>'
            f'<p style="color:{TEXT_PRIMARY}; margin:0 0 0 4px;">{_esc(thought)}</p></div>'
        )

    def _log_plan(self, plan_data: dict):
        actions = plan_data.get("actions", [])
        if not actions:
            return
        plan_json = json.dumps(actions, indent=2)
        self._log_html(
            f'<div style="margin: 4px 0 4px 8px;">'
            f'<p style="color:{TEXT_SECONDARY}; margin:0 0 2px 0; font-size:10px;">ACTION PLAN</p>'
            f'<pre style="color:{ACCENT_BLUE}; margin:0 0 0 4px; font-size:11px;">'
            f'{_esc(plan_json)}</pre></div>'
        )

    def _log_action(self, desc: str, result: str):
        self._log_html(
            f'<div style="margin: 4px 0 4px 8px;">'
            f'<p style="color:{TEXT_SECONDARY}; margin:0 0 2px 0; font-size:10px;">EXECUTED</p>'
            f'<p style="color:{ACCENT_GREEN}; margin:0 0 0 4px;">{_esc(desc)}</p>'
            f'<p style="color:{TEXT_SECONDARY}; margin:0 0 0 4px; font-size:10px;">RESULT</p>'
            f'<p style="color:{TEXT_PRIMARY}; margin:0 0 0 4px;">{_esc(result)}</p></div>'
        )

    # ── Screenshot display ────────────────────────────────

    def _update_screenshot_label(self):
        if self._current_pixmap is None:
            return
        lbl = self.screenshot_label
        scaled = self._current_pixmap.scaled(
            lbl.width() - 4, lbl.height() - 4,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        lbl.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_screenshot_label()


# ── Entry point ──────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
