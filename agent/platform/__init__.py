"""
Cross-platform accessibility extraction.

Dispatches to the correct backend based on the current OS:
  - Windows: pywinauto (UIA)
  - Linux:   AT-SPI (pyatspi2)
  - macOS:   ApplicationServices (AXUIElement)
"""

import sys
from typing import Protocol

from agent.screen_analyzer import UIElement
from utils.logger import get_logger

log = get_logger("platform")


class AccessibilityBackend(Protocol):
    def get_elements(self, max_elements: int) -> list[UIElement]:
        ...


def get_backend() -> AccessibilityBackend:
    """Return the appropriate accessibility backend for the current platform."""
    if sys.platform == "win32":
        from agent.platform.windows import WindowsAccessibility
        return WindowsAccessibility()
    elif sys.platform == "linux":
        from agent.platform.linux import LinuxAccessibility
        return LinuxAccessibility()
    elif sys.platform == "darwin":
        from agent.platform.macos import MacOSAccessibility
        return MacOSAccessibility()
    else:
        log.warning("Unsupported platform %s, returning empty elements", sys.platform)
        return _NullBackend()


class _NullBackend:
    def get_elements(self, max_elements: int = 30) -> list[UIElement]:  # noqa: ARG002
        return []
