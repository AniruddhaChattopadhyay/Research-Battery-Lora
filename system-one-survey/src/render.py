"""Make an image of an input text (added on 2026-10-16, exploratory).

The image test sends the same input as text and as an image of the text. Everything
else stays the same: the item, the question card and the system. The image must be
readable and must fit the limits of all systems (OpenRouter image guide, 2026-10-16):

  Jev     PNG or JPEG; 256 to 1,568 pixels on each side. In our test, Jev 1.13 and 1.14
          rejected 1,000 x 256 ("Invalid image dimensions") and accepted 1,000 x 300 and
          more. Thus, each image is at least 400 pixels high.
  Clef    at most about 300 KB for each image

Settings: DejaVu Sans (the font of matplotlib, Bitstream Vera license), black text on a
white background, 1,000 pixels wide. The font size is 24 pixels; a long text gets 20 or
16 pixels. render() returns None for a text that does not fit at 16 pixels, or that has a
character that the font does not have (for example an emoji). Such items are left out.
"""
import base64
import hashlib
import io
import pathlib

import matplotlib
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = pathlib.Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
WIDTH = 1000
MARGIN = 32
MIN_HEIGHT = 400
MAX_HEIGHT = 1568
SIZES = (24, 20, 16)
LINE = 1.35  # line height as a multiple of the font size
MAX_BYTES = 300_000

_CMAP = set(TTFont(FONT_PATH)["cmap"].getBestCmap())


def covered(text):
    """True if the font has a glyph for each character of the text."""
    return all(ord(c) in _CMAP for c in text if c not in "\n\r\t")


def _wrap(text, font, width):
    lines = []
    for paragraph in text.replace("\r", "").split("\n"):
        line = ""
        for word in paragraph.replace("\t", " ").split(" "):
            candidate = word if not line else f"{line} {word}"
            if font.getlength(candidate) <= width:
                line = candidate
                continue
            if line:
                lines.append(line)
            # a word that is longer than the line: break it into parts
            while font.getlength(word) > width:
                cut = len(word)
                while cut > 1 and font.getlength(word[:cut]) > width:
                    cut -= 1
                lines.append(word[:cut])
                word = word[cut:]
            line = word
        lines.append(line)
    return lines


def render(text):
    """PNG bytes and the settings, or None if the text cannot be shown well."""
    if not covered(text):
        return None
    for size in SIZES:
        font = ImageFont.truetype(str(FONT_PATH), size)
        lines = _wrap(text, font, WIDTH - 2 * MARGIN)
        step = round(size * LINE)
        height = max(MIN_HEIGHT, 2 * MARGIN + step * len(lines))
        if height > MAX_HEIGHT:
            continue
        image = Image.new("L", (WIDTH, height), 255)
        draw = ImageDraw.Draw(image)
        for i, line in enumerate(lines):
            draw.text((MARGIN, MARGIN + i * step), line, fill=0, font=font)
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        if len(data) > MAX_BYTES:
            return None
        return data, {"font_px": size, "width": WIDTH, "height": height, "lines": len(lines), "bytes": len(data)}
    return None


def image_part(data):
    """The state part of the OpenRouter decisions API for one PNG image."""
    url = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": url}}


def sha256(data):
    return hashlib.sha256(data).hexdigest()
