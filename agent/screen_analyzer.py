"""
Screen Analyzer: extracts visible interactive UI elements via Windows UI Automation.

Uses pywinauto's UIA backend to walk the accessibility tree and returns
a filtered, numbered list of elements the VLM can reference.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.logger import get_logger

log = get_logger("screen_analyzer")

# Control types we consider "interactive" and worth showing to the VLM.
INTERACTIVE_CONTROL_TYPES = frozenset({
    "Button",
    "CheckBox",
    "ComboBox",
    "Edit",
    "Hyperlink",
    "Link",
    "ListItem",
    "MenuItem",
    "MenuBarItem",
    "RadioButton",
    "TabItem",
    "TreeItem",
    "ToolBar",
    "DataItem",
})

# Container types we recurse into but don't add to the element list themselves.
CONTAINER_TYPES = frozenset({
    "Pane",
    "Window",
    "Group",
    "Custom",
    "Tab",
    "ToolBar",
    "Menu",
    "MenuBar",
    "StatusBar",
    "TitleBar",
    "Header",
    "ScrollBar",
    "Document",
    "List",
    "Tree",
    "Table",
    "DataGrid",
    "",
})

MAX_ELEMENTS = 30
MIN_ELEMENT_SIZE = 5


@dataclass
class UIElement:
    """A single interactive element detected on screen."""
    id: int
    control_type: str
    name: str
    bbox: tuple[int, int, int, int]  # (left, top, right, bottom)

    @property
    def center(self) -> tuple[int, int]:
        l, t, r, b = self.bbox
        return ((l + r) // 2, (t + b) // 2)

    def prompt_line(self) -> str:
        cx, cy = self.center
        label = self.name[:50] if self.name else ""
        return f"  [{self.id}] {self.control_type} \"{label}\" at ({cx},{cy})"


class ScreenAnalyzer:
    """Walks the Windows UIA accessibility tree and returns interactive elements."""

    def __init__(self, max_elements: int = MAX_ELEMENTS):
        self.max_elements = max_elements

    def analyze(self) -> list[UIElement]:
        """Return visible interactive elements on the current desktop."""
        try:
            from pywinauto import Desktop
        except ImportError:
            log.warning("pywinauto not installed -- screen analysis disabled")
            return []

        elements: list[UIElement] = []
        try:
            desktop = Desktop(backend="uia")
            for win in desktop.windows():
                if len(elements) >= self.max_elements:
                    break
                self._walk(win, elements, depth=0)
        except Exception as exc:
            log.warning("Accessibility tree walk failed: %s", exc)

        for idx, elem in enumerate(elements[: self.max_elements], 1):
            elem.id = idx

        result = elements[: self.max_elements]
        log.info("Detected %d interactive elements", len(result))
        return result

    def _walk(self, ctrl, elements: list[UIElement], depth: int) -> None:
        """Recursively walk the UIA tree. ctrl is already a UIAWrapper."""
        if len(elements) >= self.max_elements or depth > 10:
            return

        ctype = _safe_control_type(ctrl)
        name = _safe_name(ctrl)
        bbox = _safe_rect(ctrl)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        is_interactive = ctype in INTERACTIVE_CONTROL_TYPES
        is_big_enough = w >= MIN_ELEMENT_SIZE and h >= MIN_ELEMENT_SIZE
        has_identity = bool(name) or ctype in ("Button", "Edit", "ComboBox", "CheckBox")

        if is_interactive and is_big_enough and has_identity:
            elements.append(UIElement(id=0, control_type=ctype, name=name, bbox=bbox))

        should_recurse = (
            ctype in INTERACTIVE_CONTROL_TYPES
            or ctype in CONTAINER_TYPES
        )
        if not should_recurse:
            return

        try:
            children = ctrl.children()
        except Exception:
            return

        for child in children:
            if len(elements) >= self.max_elements:
                return
            self._walk(child, elements, depth + 1)


def _safe_control_type(ctrl) -> str:
    try:
        return ctrl.element_info.control_type or ""
    except Exception:
        return ""


def _safe_name(ctrl) -> str:
    try:
        name = ctrl.element_info.name or ""
        # Strip Unicode control characters (LTR marks, etc.) that break logging
        name = name.encode("ascii", errors="ignore").decode("ascii")
        return name.strip()[:80]
    except Exception:
        return ""


def _safe_rect(ctrl) -> tuple[int, int, int, int]:
    try:
        r = ctrl.rectangle()
        return (r.left, r.top, r.right, r.bottom)
    except Exception:
        return (0, 0, 0, 0)


def build_element_map(elements: list[UIElement]) -> dict[int, tuple[int, int]]:
    """Create a mapping from element ID to center (x, y) for the executor."""
    return {e.id: e.center for e in elements}


def build_elements_text(elements: list[UIElement]) -> str:
    """Format elements as text for the VLM prompt."""
    if not elements:
        return "No interactive elements detected."
    lines = [e.prompt_line() for e in elements]
    return "Elements on screen:\n" + "\n".join(lines)
