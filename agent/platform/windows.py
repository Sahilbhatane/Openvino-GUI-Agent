"""
Windows accessibility backend using pywinauto UIA.
"""

from agent.screen_analyzer import UIElement
from utils.logger import get_logger

log = get_logger("platform.windows")

INTERACTIVE_CONTROL_TYPES = frozenset({
    "Button", "CheckBox", "ComboBox", "Edit", "Hyperlink", "Link",
    "ListItem", "MenuItem", "MenuBarItem", "RadioButton", "TabItem",
    "TreeItem", "ToolBar", "DataItem",
})

CONTAINER_TYPES = frozenset({
    "Pane", "Window", "Group", "Custom", "Tab", "ToolBar", "Menu",
    "MenuBar", "StatusBar", "TitleBar", "Header", "ScrollBar",
    "Document", "List", "Tree", "Table", "DataGrid", "",
})

MIN_ELEMENT_SIZE = 5


class WindowsAccessibility:
    def get_elements(self, max_elements: int = 30) -> list[UIElement]:
        try:
            from pywinauto import Desktop
        except ImportError:
            log.warning("pywinauto not installed")
            return []

        elements: list[UIElement] = []
        try:
            desktop = Desktop(backend="uia")
            for win in desktop.windows():
                if len(elements) >= max_elements:
                    break
                self._walk(win, elements, max_elements, depth=0)
        except Exception as exc:
            log.warning("UIA tree walk failed: %s", exc)

        for idx, elem in enumerate(elements[:max_elements], 1):
            elem.id = idx
        return elements[:max_elements]

    def _walk(self, ctrl, elements: list[UIElement], max_elements: int, depth: int) -> None:
        if len(elements) >= max_elements or depth > 10:
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

        should_recurse = ctype in INTERACTIVE_CONTROL_TYPES or ctype in CONTAINER_TYPES
        if not should_recurse:
            return

        try:
            children = ctrl.children()
        except Exception:
            return

        for child in children:
            if len(elements) >= max_elements:
                return
            self._walk(child, elements, max_elements, depth + 1)


def _safe_control_type(ctrl) -> str:
    try:
        return ctrl.element_info.control_type or ""
    except Exception:
        return ""


def _safe_name(ctrl) -> str:
    try:
        name = ctrl.element_info.name or ""
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
