"""Set-of-Marks and action overlay drawing."""

from PIL import Image

from agent.screen_analyzer import UIElement
from vision.som_overlay import draw_action_overlay, draw_som_overlay


def _sample_elements():
    return [
        UIElement(id=1, control_type="Button", name="OK", bbox=(10, 10, 50, 50)),
        UIElement(id=2, control_type="Edit", name="", bbox=(100, 100, 200, 120)),
    ]


def test_draw_som_overlay_same_dimensions_and_rgb():
    base = Image.new("RGB", (300, 250), color=(30, 30, 30))
    out = draw_som_overlay(base, _sample_elements())
    assert out.size == base.size
    assert out.mode == "RGB"


def test_draw_som_overlay_skips_out_of_bounds_center():
    base = Image.new("RGB", (50, 50), color=(0, 0, 0))
    bad = [UIElement(id=99, control_type="Button", name="x", bbox=(1000, 1000, 1010, 1010))]
    out = draw_som_overlay(base, bad)
    assert out.size == base.size


def test_draw_action_overlay_with_target_and_caption():
    base = Image.new("RGB", (120, 80), color=(40, 40, 40))
    elems = [UIElement(id=1, control_type="Button", name="Go", bbox=(10, 10, 60, 40))]
    out = draw_action_overlay(base, elems, target_element_id=1, action_text="Step 1: click")
    assert out.size == base.size
    assert out.mode == "RGB"
