from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# directory holding bundled .ttf files
FONTS_DIR = Path(__file__).parent / "fonts"

# map of font key -> filename
FONTS = {
    "Merriweather":      "Merriweather-Regular.ttf",
    "Roboto":            "Roboto-Regular.ttf",
    "JetBrainsMono":     "JetBrainsMono-Regular.ttf",
    "PlayfairDisplay":   "PlayfairDisplay-Regular.ttf",
    "BebasNeue":         "BebasNeue-Regular.ttf",
    "Lobster":           "Lobster-Regular.ttf",
}


def list_fonts() -> list[str]:
    # return font keys for available bundled fonts
    return [k for k, fname in FONTS.items() if (FONTS_DIR / fname).exists()]


def render_glyph(char: str, font_key: str, size: int = 512, padding: int = 40) -> Image.Image:
    # white canvas, black glyph centered, auto-fit font size
    if font_key not in FONTS:
        raise ValueError(f"unknown font: {font_key}")
    font_path = FONTS_DIR / FONTS[font_key]

    canvas = Image.new("RGB", (size, size), color="white")
    draw = ImageDraw.Draw(canvas)

    # binary search font size so glyph fits within (size - 2*padding)
    target = size - 2 * padding
    lo, hi = 10, 800
    best_font = ImageFont.truetype(str(font_path), 10)
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(str(font_path), mid)
        bbox = draw.textbbox((0, 0), char, font=f)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= target and h <= target:
            best_font = f
            lo = mid + 1
        else:
            hi = mid - 1

    # center using bbox offset
    bbox = draw.textbbox((0, 0), char, font=best_font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - w) / 2 - bbox[0]
    y = (size - h) / 2 - bbox[1]
    draw.text((x, y), char, fill="black", font=best_font)
    return canvas
