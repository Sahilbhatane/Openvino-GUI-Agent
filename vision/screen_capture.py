"""
Fast screen capture using MSS.

Returns PIL Images suitable for direct VLM input.
"""

import mss
from PIL import Image

from config import MAX_VLM_DIMENSION


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
    """Return the screenshot plus geometry for coordinate mapping.

    ``width`` / ``height`` match the returned PIL image (after downscaling).
    ``monitor_*`` fields describe the full capture region before scale.
    """
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, raw.rgb)
        scaled = _downscale(img)
        sx = scaled.width / raw.size[0] if raw.size[0] else 1.0
        sy = scaled.height / raw.size[1] if raw.size[1] else 1.0
        meta = {
            "width": scaled.width,
            "height": scaled.height,
            "left": monitor["left"],
            "top": monitor["top"],
            "monitor_width": monitor["width"],
            "monitor_height": monitor["height"],
            "scale_x": sx,
            "scale_y": sy,
        }
        return scaled, meta
