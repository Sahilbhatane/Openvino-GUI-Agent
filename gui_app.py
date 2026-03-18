"""
PySide6 desktop GUI for the OpenVINO GUI Agent.

Launch:  python gui_app.py

Features:
  - Instruction input + Run / Stop controls
  - Live screenshot preview with SoM + action overlay
  - Step-by-step reasoning log (thought + action)
  - Real-time status indicator (IDLE / PLANNING / EXECUTING / ...)
  - "Show reasoning" toggle
  - Model loaded once at startup, reused across tasks
"""

import sys

from PySide6.QtCore import Qt, Signal, Slot, QThread, QSize
from PySide6.QtGui import QImage, QPixmap, QFont, QTextCursor
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


# ── Helpers ───────────────────────────────────────────────

def pil_to_qpixmap(pil_image: Image.Image) -> QPixmap:
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    data = pil_image.tobytes("raw", "RGB")
    qimg = QImage(data, pil_image.width, pil_image.height,
                  3 * pil_image.width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _esc(text: str) -> str:
    """Minimal HTML escaping for log messages."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ── Status colors ────────────────────────────────────────

STATUS_COLORS = {
    "IDLE":      "#666680",
    "LOADING":   "#cfcf5b",
    "OBSERVING": "#5ba8cf",
    "PLANNING":  "#5b6acf",
    "EXECUTING": "#cf8f5b",
    "WAITING":   "#cfcf5b",
    "DONE":      "#6fcf6f",
    "FAILED":    "#cf5b5b",
}


STYLESHEET = """
QMainWindow {
    background-color: #141522;
}
QWidget#central {
    background-color: #141522;
}

/* ── Input bar ─────────────────────────────────── */
QLineEdit {
    background-color: #1e2035;
    color: #e4e4ef;
    border: 1px solid #2d2f4a;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 14px;
    selection-background-color: #5b6acf;
}
QLineEdit:focus {
    border-color: #5b6acf;
}
QLineEdit:disabled {
    color: #555;
}

/* ── Buttons ───────────────────────────────────── */
QPushButton {
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-size: 13px;
    font-weight: bold;
}
QPushButton#run_btn {
    background-color: #5b6acf;
    color: #ffffff;
}
QPushButton#run_btn:hover {
    background-color: #6f7de0;
}
QPushButton#run_btn:pressed {
    background-color: #4a58b5;
}
QPushButton#stop_btn {
    background-color: #cf5b5b;
    color: #ffffff;
}
QPushButton#stop_btn:hover {
    background-color: #e06f6f;
}
QPushButton#stop_btn:pressed {
    background-color: #b54a4a;
}
QPushButton:disabled {
    background-color: #252740;
    color: #555;
}

/* ── Log panel ─────────────────────────────────── */
QTextEdit#log_panel {
    background-color: #181a2e;
    color: #c8c8d8;
    border: 1px solid #252740;
    border-radius: 8px;
    padding: 10px;
    font-size: 12px;
}

/* ── Screenshot label ──────────────────────────── */
QLabel#screenshot_label {
    background-color: #181a2e;
    border: 1px solid #252740;
    border-radius: 8px;
}

/* ── Status bar ────────────────────────────────── */
QStatusBar {
    background-color: #10111e;
    color: #666680;
    font-size: 12px;
    padding: 2px 8px;
}

/* ── Splitter ──────────────────────────────────── */
QSplitter::handle {
    background-color: #252740;
    width: 2px;
}

/* ── Section headers ───────────────────────────── */
QLabel#section_header {
    color: #8888a8;
    font-size: 11px;
    font-weight: bold;
    padding: 4px 0px;
}

/* ── Checkbox ──────────────────────────────────── */
QCheckBox {
    color: #8888a8;
    font-size: 12px;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid #3a3b5e;
    background-color: #1e2035;
}
QCheckBox::indicator:checked {
    background-color: #5b6acf;
    border-color: #5b6acf;
}
"""


# ── Status Indicator Widget ──────────────────────────────

class StatusIndicator(QWidget):
    """Colored dot + label showing the current agent state."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(6)

        self._dot = QLabel()
        self._dot.setFixedSize(10, 10)
        layout.addWidget(self._dot)

        self._label = QLabel("IDLE")
        self._label.setFixedWidth(90)
        layout.addWidget(self._label)

        self.set_status("IDLE")

    def set_status(self, status: str):
        color = STATUS_COLORS.get(status, "#666680")
        self._dot.setStyleSheet(
            f"background-color: {color}; border-radius: 5px; border: none;"
        )
        self._label.setText(status)
        self._label.setStyleSheet(
            f"color: {color}; font-size: 12px; font-weight: bold; border: none;"
        )


# ── Model Loader Thread ──────────────────────────────────

class ModelLoader(QThread):
    """Loads the VLM model in a background thread so the GUI stays responsive."""
    progress = Signal(str)
    finished = Signal(object)
    error = Signal(str)

    def run(self):
        try:
            self.progress.emit("Loading VLM model — this may take a minute ...")
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
    """Runs the agent loop in a background thread, emitting Qt signals for each event."""
    step_started = Signal(int, int)
    screenshot_ready = Signal(object, object)
    thought_ready = Signal(str)
    action_done = Signal(str, str)
    step_complete = Signal(dict)
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
                on_step_complete=self.step_complete.emit,
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
        self.resize(1280, 780)
        self.setMinimumSize(QSize(900, 560))

        self._controller: AgentController | None = None
        self._worker: AgentWorker | None = None
        self._current_pixmap: QPixmap | None = None
        self._step_count = 0

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
        root.setContentsMargins(16, 14, 16, 8)
        root.setSpacing(10)

        # Title row
        title_row = QHBoxLayout()
        title = QLabel("OpenVINO GUI Agent")
        title.setStyleSheet("color: #e4e4ef; font-size: 18px; font-weight: bold;")
        title_row.addWidget(title)
        title_row.addStretch()
        self.status_indicator = StatusIndicator()
        title_row.addWidget(self.status_indicator)
        root.addLayout(title_row)

        # ── Input row ─────────────────────────────────────
        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        self.instruction_input = QLineEdit()
        self.instruction_input.setPlaceholderText(
            'Enter instruction  (e.g. "open calculator and compute 42 \u00d7 7")'
        )
        self.instruction_input.returnPressed.connect(self._on_run)
        input_row.addWidget(self.instruction_input, stretch=1)

        self.run_btn = QPushButton("Run")
        self.run_btn.setObjectName("run_btn")
        self.run_btn.clicked.connect(self._on_run)
        input_row.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.clicked.connect(self._on_stop)
        input_row.addWidget(self.stop_btn)

        root.addLayout(input_row)

        # ── Main split: screenshot | log ──────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)

        # Left — Screenshot
        left_frame = QWidget()
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        lbl_screen = QLabel("LIVE SCREEN")
        lbl_screen.setObjectName("section_header")
        left_layout.addWidget(lbl_screen)

        self.screenshot_label = QLabel()
        self.screenshot_label.setObjectName("screenshot_label")
        self.screenshot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.screenshot_label.setMinimumSize(QSize(400, 300))
        self.screenshot_label.setText("No screenshot yet")
        self.screenshot_label.setStyleSheet(
            self.screenshot_label.styleSheet() + " color: #555;"
        )
        left_layout.addWidget(self.screenshot_label, stretch=1)

        splitter.addWidget(left_frame)

        # Right — Log
        right_frame = QWidget()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Header row: label + toggle
        log_header = QHBoxLayout()
        log_header.setSpacing(0)
        lbl_log = QLabel("AGENT LOG")
        lbl_log.setObjectName("section_header")
        log_header.addWidget(lbl_log)
        log_header.addStretch()
        self.show_reasoning_cb = QCheckBox("Show reasoning")
        self.show_reasoning_cb.setChecked(True)
        log_header.addWidget(self.show_reasoning_cb)
        right_layout.addLayout(log_header)

        self.log_panel = QTextEdit()
        self.log_panel.setObjectName("log_panel")
        self.log_panel.setReadOnly(True)
        mono = QFont("Cascadia Code", 10)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.log_panel.setFont(mono)
        right_layout.addWidget(self.log_panel, stretch=1)

        splitter.addWidget(right_frame)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        root.addWidget(splitter, stretch=1)

        # ── Status bar ────────────────────────────────────
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Initializing ...")

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
        self.status_bar.showMessage("Ready — enter an instruction and press Run")
        self._log_system("Model loaded. Agent ready.")

    @Slot(str)
    def _on_load_error(self, msg: str):
        self.status_indicator.set_status("FAILED")
        self.status_bar.showMessage(f"Load error: {msg}")
        self._log_error(f"Failed to load model: {msg}")

    # ── Run / Stop ────────────────────────────────────────

    @Slot()
    def _on_run(self):
        instruction = self.instruction_input.text().strip()
        if not instruction or self._controller is None:
            return
        self._set_ui_state("running")
        self._step_count = 0
        self.log_panel.clear()
        self._log_system(f"Task: {instruction}")
        self.status_bar.showMessage("Running ...")

        self._worker = AgentWorker(self._controller, instruction, parent=self)
        self._worker.step_started.connect(self._on_step_start)
        self._worker.screenshot_ready.connect(self._on_screenshot)
        self._worker.thought_ready.connect(self._on_thought)
        self._worker.action_done.connect(self._on_action)
        self._worker.step_complete.connect(self._on_step_complete)
        self._worker.task_complete.connect(self._on_task_complete)
        self._worker.error_occurred.connect(self._on_agent_error)
        self._worker.status_changed.connect(self._on_status_change)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    @Slot()
    def _on_stop(self):
        if self._worker is not None:
            self._worker.stop()
            self._log_system("Stop requested — waiting for current step to finish ...")

    # ── Agent event slots ─────────────────────────────────

    @Slot(str)
    def _on_status_change(self, status: str):
        self.status_indicator.set_status(status)

    @Slot(int, int)
    def _on_step_start(self, step: int, max_steps: int):
        self._step_count = step
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

        parts = [f"Step {step}", f"{elems} elements", f"{gen_time}s", f"{tps} tok/s"]
        if not changed:
            parts.append("no screen change")
        self.status_bar.showMessage("  |  ".join(parts))

    @Slot(dict)
    def _on_task_complete(self, result: dict):
        status = result.get("status", "unknown")
        steps = result.get("iterations", 0)
        usage = result.get("token_usage", {})
        tps = usage.get("avg_tokens_per_second", 0)
        total_t = usage.get("total_generation_time", 0)

        if status == "completed":
            color = "#6fcf6f"
            icon = "DONE"
        elif status == "stopped":
            color = "#cfcf5b"
            icon = "STOPPED"
        else:
            color = "#cf5b5b"
            icon = "INCOMPLETE"

        self._log_html(
            f'<br><p style="color:{color}; font-weight:bold; margin:8px 0 2px 0;">'
            f'{icon} — Task {status} in {steps} step(s)  '
            f'({total_t}s total, {tps} tok/s avg)</p>'
        )
        self.status_bar.showMessage(
            f"{icon}  |  {steps} steps  |  {total_t}s  |  {tps} tok/s"
        )

    @Slot(str)
    def _on_agent_error(self, msg: str):
        self._log_error(msg)

    @Slot()
    def _on_worker_finished(self):
        self._set_ui_state("idle")

    # ── Log helpers ───────────────────────────────────────

    def _log_html(self, html: str):
        self.log_panel.moveCursor(QTextCursor.MoveOperation.End)
        self.log_panel.insertHtml(html)
        self.log_panel.moveCursor(QTextCursor.MoveOperation.End)

    def _log_system(self, text: str):
        self._log_html(
            f'<p style="color:#6688aa; margin:2px 0;">{_esc(text)}</p>'
        )

    def _log_error(self, text: str):
        self._log_html(
            f'<p style="color:#cf5b5b; margin:2px 0;">'
            f'<span style="color:#cf5b5b;">\u26a0</span> {_esc(text)}</p>'
        )

    def _log_step_header(self, step: int, max_steps: int):
        self._log_html(
            f'<p style="color:#5b6acf; font-weight:bold; margin:10px 0 2px 0;">'
            f'[Step {step}/{max_steps}]</p>'
        )

    def _log_thought(self, thought: str):
        self._log_html(
            f'<p style="color:#c8c8d8; margin:1px 0 1px 12px;">'
            f'<span style="color:#7878a8;">Thought:</span> {_esc(thought)}</p>'
        )

    def _log_action(self, desc: str, result: str):
        self._log_html(
            f'<p style="color:#6fcf6f; margin:1px 0 1px 12px;">'
            f'<span style="color:#7878a8;">Action:</span> {_esc(desc)}</p>'
            f'<p style="color:#5ba8cf; margin:1px 0 1px 24px;">'
            f'<span style="color:#7878a8;">\u2192</span> {_esc(result)}</p>'
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
