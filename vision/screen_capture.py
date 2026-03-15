"""
Fast screen capture using MSS.

Returns PIL Images suitable for direct VLM input.
"""

import mss
from PIL import Image


def capture_screen() -> Image.Image:
    """Grab the primary monitor and return an RGB PIL Image."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw = sct.grab(monitor)
        # mss gives BGRA bytes; .rgb converts to RGB on the fly
        img = Image.frombytes("RGB", raw.size, raw.rgb)
        return img


def capture_screen_with_metadata() -> tuple[Image.Image, dict]:
    """Return the screenshot plus monitor geometry for coordinate mapping."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, raw.rgb)
        meta = {
            "width": monitor["width"],
            "height": monitor["height"],
            "left": monitor["left"],
            "top": monitor["top"],
        }
        return img, meta
