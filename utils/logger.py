"""
Lightweight logging setup shared across all modules.
"""

import logging
import sys

_FMT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_DATEFMT = "%H:%M:%S"

_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root = logging.getLogger("gui-agent")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the gui-agent namespace."""
    _configure_root()
    return logging.getLogger(f"gui-agent.{name}")
