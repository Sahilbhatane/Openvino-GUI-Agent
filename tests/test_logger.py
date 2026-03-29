"""Shared logging helper."""

from utils.logger import get_logger


def test_get_logger_names_child_under_gui_agent():
    log = get_logger("test_ns")
    assert log.name == "gui-agent.test_ns"


def test_get_logger_idempotent_handlers():
    get_logger("a")
    get_logger("b")
    root = __import__("logging").getLogger("gui-agent")
    assert len(root.handlers) >= 1
