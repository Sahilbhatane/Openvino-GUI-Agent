"""OS actions: mocked PyAutoGUI so no real keyboard automation runs."""

from unittest.mock import MagicMock, patch

import pytest

import agent.os_actions as os_actions


@pytest.fixture
def mock_pyautogui():
    with patch.object(os_actions.pyautogui, "press", MagicMock()):
        with patch.object(os_actions.pyautogui, "typewrite", MagicMock()):
            with patch.object(os_actions.pyautogui, "hotkey", MagicMock()):
                yield


def test_open_app_windows_uses_start_menu_flow(mock_pyautogui):
    with patch.object(os_actions.sys, "platform", "win32"):
        msg = os_actions.open_app("notepad")
    assert "notepad" in msg.lower()
    assert "FAILED" not in msg


def test_type_text_ascii_uses_typewrite(mock_pyautogui):
    with patch.object(os_actions.pyautogui, "typewrite", MagicMock()) as tw:
        msg = os_actions.type_text("abc")
    tw.assert_called_once()
    assert "abc" in msg


def test_open_app_unsupported_platform():
    with patch.object(os_actions.sys, "platform", "vxworks"):
        msg = os_actions.open_app("x")
    assert "Unsupported" in msg
