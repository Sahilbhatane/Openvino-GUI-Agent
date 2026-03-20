"""
Linux accessibility backend using AT-SPI (pyatspi2).

Requires: python3-pyatspi (system package) or pyatspi via pip.
AT-SPI daemon must be running (usually started by the desktop environment).
"""

from agent.screen_analyzer import UIElement
from utils.logger import get_logger

log = get_logger("platform.linux")

INTERACTIVE_ROLES = frozenset({
    "ROLE_PUSH_BUTTON", "ROLE_TOGGLE_BUTTON", "ROLE_CHECK_BOX",
    "ROLE_RADIO_BUTTON", "ROLE_COMBO_BOX", "ROLE_TEXT", "ROLE_ENTRY",
    "ROLE_PASSWORD_TEXT", "ROLE_LINK", "ROLE_MENU_ITEM",
    "ROLE_CHECK_MENU_ITEM", "ROLE_RADIO_MENU_ITEM", "ROLE_LIST_ITEM",
    "ROLE_TAB", "ROLE_TREE_ITEM", "ROLE_SLIDER", "ROLE_SPIN_BUTTON",
    "ROLE_TOOL_BAR",
})

CONTAINER_ROLES = frozenset({
    "ROLE_APPLICATION", "ROLE_FRAME", "ROLE_DIALOG", "ROLE_PANEL",
    "ROLE_FILLER", "ROLE_SCROLL_PANE", "ROLE_VIEWPORT", "ROLE_SPLIT_PANE",
    "ROLE_MENU_BAR", "ROLE_MENU", "ROLE_TOOL_BAR", "ROLE_STATUS_BAR",
    "ROLE_PAGE_TAB_LIST", "ROLE_TABLE", "ROLE_TREE_TABLE", "ROLE_TREE",
    "ROLE_LIST", "ROLE_DOCUMENT_FRAME", "ROLE_DESKTOP_FRAME",
    "ROLE_LAYERED_PANE", "ROLE_INTERNAL_FRAME",
})

MIN_ELEMENT_SIZE = 5


class LinuxAccessibility:
    def get_elements(self, max_elements: int = 30) -> list[UIElement]:
        try:
            import pyatspi
        except ImportError:
            log.warning(
                "pyatspi not installed. Install via: "
                "sudo apt install python3-pyatspi (Debian/Ubuntu) or "
                "pip install pyatspi"
            )
            return []

        elements: list[UIElement] = []
        try:
            desktop = pyatspi.Registry.getDesktop(0)
            for i in range(desktop.childCount):
                if len(elements) >= max_elements:
                    break
                app = desktop.getChildAtIndex(i)
                if app is None:
                    continue
                self._walk(app, elements, max_elements, depth=0)
        except Exception as exc:
            log.warning("AT-SPI tree walk failed: %s", exc)

        for idx, elem in enumerate(elements[:max_elements], 1):
            elem.id = idx
        return elements[:max_elements]

    def _walk(self, node, elements: list[UIElement], max_elements: int, depth: int) -> None:
        if len(elements) >= max_elements or depth > 10:
            return

        try:
            import pyatspi
            role = node.getRoleName().upper().replace(" ", "_")
            role_key = f"ROLE_{role}"
        except Exception:
            return

        name = _safe_name(node)
        bbox = _safe_extents(node)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        is_interactive = role_key in INTERACTIVE_ROLES
        is_big_enough = w >= MIN_ELEMENT_SIZE and h >= MIN_ELEMENT_SIZE
        has_identity = bool(name) or is_interactive

        if is_interactive and is_big_enough and has_identity:
            ctype = _role_to_control_type(role_key)
            elements.append(UIElement(id=0, control_type=ctype, name=name, bbox=bbox))

        should_recurse = role_key in INTERACTIVE_ROLES or role_key in CONTAINER_ROLES
        if not should_recurse:
            return

        try:
            for i in range(node.childCount):
                if len(elements) >= max_elements:
                    return
                child = node.getChildAtIndex(i)
                if child is not None:
                    self._walk(child, elements, max_elements, depth + 1)
        except Exception:
            return


def _safe_name(node) -> str:
    try:
        name = node.name or ""
        name = name.encode("ascii", errors="ignore").decode("ascii")
        return name.strip()[:80]
    except Exception:
        return ""


def _safe_extents(node) -> tuple[int, int, int, int]:
    try:
        import pyatspi
        component = node.queryComponent()
        ext = component.getExtents(pyatspi.DESKTOP_COORDS)
        return (ext.x, ext.y, ext.x + ext.width, ext.y + ext.height)
    except Exception:
        return (0, 0, 0, 0)


def _role_to_control_type(role_key: str) -> str:
    mapping = {
        "ROLE_PUSH_BUTTON": "Button",
        "ROLE_TOGGLE_BUTTON": "Button",
        "ROLE_CHECK_BOX": "CheckBox",
        "ROLE_RADIO_BUTTON": "RadioButton",
        "ROLE_COMBO_BOX": "ComboBox",
        "ROLE_TEXT": "Edit",
        "ROLE_ENTRY": "Edit",
        "ROLE_PASSWORD_TEXT": "Edit",
        "ROLE_LINK": "Hyperlink",
        "ROLE_MENU_ITEM": "MenuItem",
        "ROLE_CHECK_MENU_ITEM": "MenuItem",
        "ROLE_RADIO_MENU_ITEM": "MenuItem",
        "ROLE_LIST_ITEM": "ListItem",
        "ROLE_TAB": "TabItem",
        "ROLE_TREE_ITEM": "TreeItem",
        "ROLE_SLIDER": "Edit",
        "ROLE_SPIN_BUTTON": "Edit",
        "ROLE_TOOL_BAR": "ToolBar",
    }
    return mapping.get(role_key, "Button")
