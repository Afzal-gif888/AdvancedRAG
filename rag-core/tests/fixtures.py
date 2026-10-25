"""Synthetic document builders for the extraction tests (no binary fixtures are committed).

Everything is generated at test time from text, so the expected content is known exactly:
- text PDFs and mixed PDFs with reportlab,
- scanned (image-only) PDFs and PNG/JPEG images with Pillow,
- DOCX with python-docx.
"""
import io
from typing import List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont


def _font(size: int):
    return ImageFont.load_default(size=size)


def text_image(lines: Sequence[str], size: int = 34, width: int = 1400, noise: bool = False, blur: float = 0.0, scale: float = 1.0) -> Image.Image:
    """A white page image with black text lines. ``blur``/``scale``/``noise`` degrade it (low-quality scans)."""
    height = 60 + len(lines) * (size + 26)
    img = Image.new("RGB", (width, max(height, 200)), "white")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((50, 40 + i * (size + 26)), line, fill="black", font=_font(size))
    if scale != 1.0:
        small = img.resize((max(int(img.width * scale), 1), max(int(img.height * scale), 1)))
        img = small.resize(img.size)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        import random

        rnd = random.Random(7)
        px = img.load()
        for _ in range(img.width * img.height // 25):
            x, y = rnd.randrange(img.width), rnd.randrange(img.height)
            v = rnd.randrange(0, 255)
            px[x, y] = (v, v, v)
    return img


def blank_image(width: int = 1000, height: int = 600) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def image_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def scanned_pdf(page_images: List[Image.Image]) -> bytes:
    """A PDF whose pages are only pictures (no text layer), like a scan."""
    buf = io.BytesIO()
    first, rest = page_images[0], page_images[1:]
    first.save(buf, format="PDF", save_all=True, append_images=rest, resolution=150)
    return buf.getvalue()


def mixed_pdf(pages: Sequence[object]) -> bytes:
    """A PDF with real text pages (``str``) and picture pages (``PIL.Image``), in the given order.
    Text pages carry a text layer; picture pages do not (so they need OCR)."""
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(612, 792))
    for page in pages:
        if isinstance(page, str):
            y = 740
            for line in page.split("\n"):
                c.setFont("Helvetica", 12)
                c.drawString(72, y, line)
                y -= 18
        elif isinstance(page, Image.Image):
            ratio = 468 / page.width
            c.drawImage(ImageReader(page), 72, 792 - 72 - page.height * ratio, width=468, height=page.height * ratio)
        c.showPage()
    c.save()
    return buf.getvalue()


def text_pdf(pages: Sequence[str]) -> bytes:
    return mixed_pdf(list(pages))


def docx_bytes(build) -> bytes:
    import docx

    d = docx.Document()
    build(d)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def layout_pdf(pages: Sequence[Sequence[tuple]]) -> bytes:
    """A text-layer PDF with explicit layout. Each page is a list of items:
    ("text", y, "a line")                      - a paragraph line at height y (points from the bottom)
    ("table", y_top, rows, options)            - rows is a list of lists of strings; options: bordered (bool), col_x (left edges),
                                                  row_h (points), font (size)
    Used to build bordered tables (ruling lines), borderless tables (aligned text only) and tables that continue across pages."""
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(612, 792))
    for items in pages:
        for item in items:
            if item[0] == "text":
                c.setFont("Helvetica", 11)
                c.drawString(72, item[1], item[2])
            elif item[0] == "table":
                _, y_top, rows, opt = item
                col_x = opt.get("col_x") or [72 + i * 120 for i in range(len(rows[0]))]
                row_h, font = opt.get("row_h", 20), opt.get("font", 10)
                right = opt.get("right", col_x[-1] + 110)
                c.setFont("Helvetica", font)
                for r, row in enumerate(rows):
                    base = y_top - (r + 1) * row_h + 6
                    for x, cell in zip(col_x, row):
                        if cell:
                            c.drawString(x + 4, base, cell)
                if opt.get("bordered", True):
                    c.setLineWidth(0.8)
                    for r in range(len(rows) + 1):
                        y = y_top - r * row_h
                        c.line(col_x[0], y, right, y)
                    for x in list(col_x) + [right]:
                        c.line(x, y_top, x, y_top - len(rows) * row_h)
        c.showPage()
    c.save()
    return buf.getvalue()
