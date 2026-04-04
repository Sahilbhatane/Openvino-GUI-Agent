"""
Screen Analyzer: extracts visible interactive UI elements via platform-specific
accessibility APIs.

Dispatches to Windows (pywinauto), Linux (AT-SPI), or macOS (AXUIElement)
depending on the current OS. Returns a normalized list of UIElement objects
that the rest of the pipeline consumes uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.logger import get_logger

log = get_logger("screen_analyzer")

MAX_ELEMENTS = 24


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

    @property
    def role(self) -> str:
        """Normalized role string for prompt output."""
        mapping = {
            "Button": "button", "CheckBox": "checkbox", "ComboBox": "combobox",
            "Edit": "input", "Hyperlink": "link", "Link": "link",
            "ListItem": "listitem", "MenuItem": "menuitem",
            "MenuBarItem": "menuitem", "RadioButton": "radio",
            "TabItem": "tab", "TreeItem": "treeitem",
            "ToolBar": "toolbar", "DataItem": "dataitem",
        }
        return mapping.get(self.control_type, self.control_type.lower())

    def prompt_line(self) -> str:
        cx, cy = self.center
        label = self.name[:50] if self.name else ""
        return f'  [{self.id}] {self.role} "{label}" at ({cx},{cy})'

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "role": self.role,
            "name": self.name,
            "bbox": list(self.bbox),
        }


class ScreenAnalyzer:
    """Platform-dispatching screen analyzer."""

    def __init__(self, max_elements: int = MAX_ELEMENTS):
        self.max_elements = max_elements
        self._backend = None

    def _get_backend(self):
        if self._backend is None:
            from agent.platform import get_backend
            self._backend = get_backend()
        return self._backend

    def analyze(self) -> list[UIElement]:
        """Return visible interactive elements on the current desktop."""
        backend = self._get_backend()
        elements = backend.get_elements(self.max_elements)
        log.info("Detected %d interactive elements", len(elements))
        return elements


def build_element_map(elements: list[UIElement]) -> dict[int, tuple[int, int]]:
    """Create a mapping from element ID to center (x, y) for the executor."""
    return {e.id: e.center for e in elements}


def build_elements_text(elements: list[UIElement]) -> str:
    """Format elements as text for the VLM prompt."""
    if not elements:
        return "No interactive elements detected."
    lines = [e.prompt_line() for e in elements]
    return "Elements on screen:\n" + "\n".join(lines)
