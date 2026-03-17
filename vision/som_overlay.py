"""
Set-of-Marks overlay: draws numbered badges on a screenshot for each detected element.

The annotated image helps the VLM visually match numbered elements to the
structured text list, dramatically improving grounding accuracy.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from agent.screen_analyzer import UIElement

BADGE_RADIUS = 12
BADGE_COLOR = (220, 40, 40)
TEXT_COLOR = (255, 255, 255)
OUTLINE_COLOR = (0, 0, 0)
BBOX_COLOR = (220, 40, 40, 80)


def draw_som_overlay(
    image: Image.Image,
    elements: list[UIElement],
) -> Image.Image:
    """Return a copy of *image* with numbered badges drawn at each element center."""
    annotated = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font = _get_font(BADGE_RADIUS)

    for elem in elements:
        cx, cy = elem.center
        if cx < 0 or cy < 0 or cx >= image.width or cy >= image.height:
            continue

        _draw_bbox_highlight(draw, elem.bbox, image.size)

        r = BADGE_RADIUS
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=BADGE_COLOR,
            outline=OUTLINE_COLOR,
            width=1,
        )

        label = str(elem.id)
        bb = font.getbbox(label)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text(
            (cx - tw // 2, cy - th // 2 - 1),
            label,
            fill=TEXT_COLOR,
            font=font,
        )

    annotated = Image.alpha_composite(annotated, overlay)
    return annotated.convert("RGB")


def _draw_bbox_highlight(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    img_size: tuple[int, int],
) -> None:
    l, t, r, b = bbox
    l = max(0, l)
    t = max(0, t)
    r = min(img_size[0], r)
    b = min(img_size[1], b)
    if r > l and b > t:
        draw.rectangle([l, t, r, b], outline=BADGE_COLOR, width=2)


def _get_font(badge_radius: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(badge_radius, 10)
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()
