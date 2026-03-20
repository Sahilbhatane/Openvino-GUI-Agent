"""
Cross-platform OS abstraction layer.

Provides unified high-level actions (open_app, etc.) that dispatch
to the correct OS-specific implementation.
"""

import sys
import time
import subprocess

import pyautogui

from utils.logger import get_logger

log = get_logger("os_actions")


def open_app(app_name: str) -> str:
    """
    Open an application by name using OS-native search/launch.
    Returns a status string.
    """
    if sys.platform == "win32":
        return _open_app_windows(app_name)
    elif sys.platform == "darwin":
        return _open_app_macos(app_name)
    elif sys.platform == "linux":
        return _open_app_linux(app_name)
    return f"Unsupported platform: {sys.platform}"


def _open_app_windows(app_name: str) -> str:
    """Win key -> type name -> Enter."""
    try:
        pyautogui.press("win")
        time.sleep(0.8)
        pyautogui.typewrite(app_name, interval=0.05)
        time.sleep(0.5)
        pyautogui.press("enter")
        time.sleep(1.0)
        return f"open_app({app_name}) via Start Menu"
    except Exception as exc:
        return f"FAILED open_app({app_name}): {exc}"


def _open_app_macos(app_name: str) -> str:
    """Cmd+Space (Spotlight) -> type name -> Enter."""
    try:
        pyautogui.hotkey("command", "space")
        time.sleep(0.5)
        pyautogui.typewrite(app_name, interval=0.05)
        time.sleep(0.5)
        pyautogui.press("enter")
        time.sleep(1.0)
        return f"open_app({app_name}) via Spotlight"
    except Exception as exc:
        return f"FAILED open_app({app_name}): {exc}"


def _open_app_linux(app_name: str) -> str:
    """Try xdg-open, then fall back to keyboard-based launcher search."""
    well_known = {
        "calculator": ["gnome-calculator", "kcalc", "xfce4-calculator"],
        "browser": ["xdg-open", "firefox", "google-chrome", "chromium-browser"],
        "terminal": ["gnome-terminal", "konsole", "xfce4-terminal", "xterm"],
        "files": ["nautilus", "dolphin", "thunar", "pcmanfm"],
        "text editor": ["gedit", "kate", "mousepad", "xed"],
    }

    lowered = app_name.lower().strip()
    candidates = well_known.get(lowered, [app_name])

    for cmd in candidates:
        try:
            subprocess.Popen(
                [cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(1.0)
            return f"open_app({app_name}) via {cmd}"
        except FileNotFoundError:
            continue
        except Exception as exc:
            log.warning("Failed to launch %s: %s", cmd, exc)
            continue

    try:
        pyautogui.press("super")
        time.sleep(0.8)
        pyautogui.typewrite(app_name, interval=0.05)
        time.sleep(0.5)
        pyautogui.press("enter")
        time.sleep(1.0)
        return f"open_app({app_name}) via desktop search"
    except Exception as exc:
        return f"FAILED open_app({app_name}): {exc}"


def type_text(text: str) -> str:
    """Platform-aware text input."""
    try:
        if all(ord(c) < 128 for c in text):
            pyautogui.typewrite(text, interval=0.04)
        else:
            import pyperclip
            pyperclip.copy(text)
            paste_key = "command" if sys.platform == "darwin" else "ctrl"
            pyautogui.hotkey(paste_key, "v")
        return f'type_text("{text[:40]}")'
    except Exception as exc:
        return f"FAILED type_text: {exc}"
