"""Controller pure helpers: memory formatting and screen diff."""

from collections import deque

from PIL import Image

from agent.controller import StepMemory, _compute_screen_diff, _format_memory


def test_format_memory_empty():
    assert _format_memory(deque()) == ""


def test_format_memory_includes_steps():
    mem = deque(
        [
            StepMemory(1, "think", "click 1", "ok", True, execution_ok=True),
        ],
        maxlen=5,
    )
    text = _format_memory(mem)
    assert "Step 1" in text
    assert "think" in text
    assert "screen changed" in text


def test_compute_screen_diff_identical_zero():
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    assert _compute_screen_diff(img, img) == 0.0


def test_compute_screen_diff_different_images_positive():
    a = Image.new("RGB", (64, 64), color=(0, 0, 0))
    b = Image.new("RGB", (64, 64), color=(255, 255, 255))
    assert _compute_screen_diff(a, b) > 0.1
