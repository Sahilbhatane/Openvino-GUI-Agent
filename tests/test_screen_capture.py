"""Screen capture helpers (unit tests; no real display required for _downscale)."""

from PIL import Image

from vision import screen_capture as sc


def test_downscale_leaves_small_image_unchanged():
    img = Image.new("RGB", (100, 80), color=(1, 2, 3))
    out = sc._downscale(img, max_dim=1280)
    assert out.size == (100, 80)
    assert out.getpixel((0, 0)) == (1, 2, 3)


def test_downscale_scales_down_long_edge():
    img = Image.new("RGB", (2000, 1000), color=(9, 9, 9))
    out = sc._downscale(img, max_dim=1000)
    assert max(out.size) == 1000
    assert out.size[0] == 1000
    assert out.size[1] == 500


def test_capture_screen_with_metadata_image_matches_meta_dimensions():
    """Returned PIL size must match meta width/height (VLM / coordinate contract)."""
    from unittest.mock import MagicMock, patch

    fake_raw = MagicMock()
    fake_raw.size = (400, 200)
    fake_raw.rgb = b"\xff" * (400 * 200 * 3)

    fake_monitor = {"left": 0, "top": 0, "width": 400, "height": 200}
    fake_sct = MagicMock()
    fake_sct.__enter__ = MagicMock(return_value=fake_sct)
    fake_sct.__exit__ = MagicMock(return_value=False)
    fake_sct.monitors = [MagicMock(), fake_monitor]
    fake_sct.grab = MagicMock(return_value=fake_raw)

    with patch("vision.screen_capture.mss.mss", return_value=fake_sct):
        img, meta = sc.capture_screen_with_metadata()

    assert img.width == meta["width"]
    assert img.height == meta["height"]
    assert meta["monitor_width"] == 400
    assert meta["monitor_height"] == 200
