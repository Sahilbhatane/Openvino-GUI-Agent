"""
Fast screen capture using MSS.

Returns PIL Images suitable for direct VLM input.
"""

import mss
from PIL import Image

MAX_VLM_DIMENSION = 1280


def _downscale(img: Image.Image, max_dim: int = MAX_VLM_DIMENSION) -> Image.Image:
    """Shrink image so the longest side is at most *max_dim* pixels."""
    w, h = img.size
    if w <= max_dim and h <= max_dim:
        return img
    scale = max_dim / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)


def capture_screen() -> Image.Image:
    """Grab the primary monitor and return an RGB PIL Image."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, raw.rgb)
        return _downscale(img)


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
        return _downscale(img), meta
