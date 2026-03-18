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


def draw_action_overlay(
    image: Image.Image,
    elements: list[UIElement],
    target_element_id: int | None = None,
    action_text: str = "",
) -> Image.Image:
    """Draw SoM overlay plus highlight the target element and show action text."""
    annotated = draw_som_overlay(image, elements)

    if target_element_id is None and not action_text:
        return annotated

    annotated = annotated.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if target_element_id is not None:
        for elem in elements:
            if elem.id == target_element_id:
                l, t, r, b = elem.bbox
                l = max(0, l)
                t = max(0, t)
                r = min(annotated.width, r)
                b = min(annotated.height, b)
                for i in range(3):
                    draw.rectangle(
                        [l - i, t - i, r + i, b + i],
                        outline=(0, 255, 100, 220),
                    )
                break

    if action_text:
        font = _get_font(14)
        draw.rectangle([0, 0, annotated.width, 32], fill=(0, 0, 0, 180))
        draw.text((10, 7), action_text, fill=(0, 255, 100, 255), font=font)

    annotated = Image.alpha_composite(annotated, overlay)
    return annotated.convert("RGB")


def _get_font(badge_radius: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(badge_radius, 10)
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()
