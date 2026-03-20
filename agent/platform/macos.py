"""
macOS accessibility backend using ApplicationServices / AXUIElement.

Requires: pyobjc-framework-ApplicationServices (pip installable).
System Preferences > Privacy & Security > Accessibility must grant access.
"""

from agent.screen_analyzer import UIElement
from utils.logger import get_logger

log = get_logger("platform.macos")

INTERACTIVE_ROLES = frozenset({
    "AXButton", "AXCheckBox", "AXRadioButton", "AXPopUpButton",
    "AXComboBox", "AXTextField", "AXTextArea", "AXSecureTextField",
    "AXLink", "AXMenuItem", "AXMenuBarItem", "AXTab", "AXRow",
    "AXCell", "AXSlider", "AXIncrementor", "AXColorWell",
    "AXDisclosureTriangle", "AXToolbar",
})

CONTAINER_ROLES = frozenset({
    "AXApplication", "AXWindow", "AXSheet", "AXDrawer", "AXDialog",
    "AXGroup", "AXScrollArea", "AXSplitGroup", "AXTabGroup",
    "AXMenuBar", "AXMenu", "AXToolbar", "AXList", "AXOutline",
    "AXTable", "AXBrowser", "AXLayoutArea", "AXLayoutItem",
})

MIN_ELEMENT_SIZE = 5


class MacOSAccessibility:
    def get_elements(self, max_elements: int = 30) -> list[UIElement]:
        try:
            from ApplicationServices import (
                AXUIElementCreateSystemWide,
                AXUIElementCopyAttributeValue,
            )
            from CoreFoundation import CFRelease
            import Quartz
        except ImportError:
            log.warning(
                "pyobjc not installed. Install via: "
                "pip install pyobjc-framework-ApplicationServices pyobjc-framework-Quartz"
            )
            return []

        elements: list[UIElement] = []
        try:
            system_wide = AXUIElementCreateSystemWide()
            err, apps_ref = AXUIElementCopyAttributeValue(
                system_wide, "AXFocusedApplication", None
            )
            if err == 0 and apps_ref is not None:
                self._walk_ax(apps_ref, elements, max_elements, depth=0)
            else:
                self._walk_all_apps(elements, max_elements)
        except Exception as exc:
            log.warning("AX tree walk failed: %s", exc)

        for idx, elem in enumerate(elements[:max_elements], 1):
            elem.id = idx
        return elements[:max_elements]

    def _walk_all_apps(self, elements: list[UIElement], max_elements: int) -> None:
        try:
            import Quartz
            from ApplicationServices import AXUIElementCreateApplication
        except ImportError:
            return

        workspace_apps = Quartz.NSWorkspace.sharedWorkspace().runningApplications()
        for app in workspace_apps:
            if len(elements) >= max_elements:
                break
            pid = app.processIdentifier()
            if pid <= 0:
                continue
            try:
                ax_app = AXUIElementCreateApplication(pid)
                self._walk_ax(ax_app, elements, max_elements, depth=0)
            except Exception:
                continue

    def _walk_ax(self, element, elements: list[UIElement], max_elements: int, depth: int) -> None:
        if len(elements) >= max_elements or depth > 10:
            return

        try:
            from ApplicationServices import AXUIElementCopyAttributeValue
        except ImportError:
            return

        role = _ax_attr(element, "AXRole") or ""
        name = _ax_attr(element, "AXTitle") or _ax_attr(element, "AXDescription") or ""
        name = str(name).encode("ascii", errors="ignore").decode("ascii").strip()[:80]

        bbox = _ax_position_size(element)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        is_interactive = role in INTERACTIVE_ROLES
        is_big_enough = w >= MIN_ELEMENT_SIZE and h >= MIN_ELEMENT_SIZE
        has_identity = bool(name) or is_interactive

        if is_interactive and is_big_enough and has_identity:
            ctype = _role_to_control_type(role)
            elements.append(UIElement(id=0, control_type=ctype, name=name, bbox=bbox))

        should_recurse = role in INTERACTIVE_ROLES or role in CONTAINER_ROLES
        if not should_recurse:
            return

        err, children = AXUIElementCopyAttributeValue(element, "AXChildren", None)
        if err != 0 or children is None:
            return

        try:
            for child in children:
                if len(elements) >= max_elements:
                    return
                self._walk_ax(child, elements, max_elements, depth + 1)
        except Exception:
            return


def _ax_attr(element, attr: str):
    try:
        from ApplicationServices import AXUIElementCopyAttributeValue
        err, value = AXUIElementCopyAttributeValue(element, attr, None)
        return value if err == 0 else None
    except Exception:
        return None


def _ax_position_size(element) -> tuple[int, int, int, int]:
    try:
        pos = _ax_attr(element, "AXPosition")
        size = _ax_attr(element, "AXSize")
        if pos is None or size is None:
            return (0, 0, 0, 0)
        import Quartz
        point = Quartz.CGPoint()
        Quartz.AXValueGetValue(pos, Quartz.kAXValueCGPointType, point)
        sz = Quartz.CGSize()
        Quartz.AXValueGetValue(size, Quartz.kAXValueCGSizeType, sz)
        x, y = int(point.x), int(point.y)
        return (x, y, x + int(sz.width), y + int(sz.height))
    except Exception:
        return (0, 0, 0, 0)


def _role_to_control_type(role: str) -> str:
    mapping = {
        "AXButton": "Button",
        "AXCheckBox": "CheckBox",
        "AXRadioButton": "RadioButton",
        "AXPopUpButton": "ComboBox",
        "AXComboBox": "ComboBox",
        "AXTextField": "Edit",
        "AXTextArea": "Edit",
        "AXSecureTextField": "Edit",
        "AXLink": "Hyperlink",
        "AXMenuItem": "MenuItem",
        "AXMenuBarItem": "MenuBarItem",
        "AXTab": "TabItem",
        "AXRow": "ListItem",
        "AXCell": "DataItem",
        "AXSlider": "Edit",
        "AXIncrementor": "Button",
        "AXToolbar": "ToolBar",
    }
    return mapping.get(role, "Button")
