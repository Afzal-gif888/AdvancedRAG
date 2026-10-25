"""OCR for scanned PDF pages and PNG/JPEG images.

Design (see docs/verification/OCR.md for measurements and limits):

- Self-hostable and free: RapidOCR (PaddleOCR detection/recognition models exported to ONNX, run with the
  onnxruntime that fastembed already needs) and pypdfium2 (PDFium, bundled wheel) to rasterise PDF pages.
  No system packages (Tesseract/Poppler) and no paid API are required.
- OCR is an optional capability (``OCR_ENABLED``) because it costs CPU and RAM. When it is off, image-only
  pages are *reported* as unprocessed instead of silently disappearing.
- Every page is bounded: render size (``OCR_MAX_PIXELS``), time (``OCR_PAGE_TIMEOUT_SECONDS``), and the
  whole document (``OCR_DOCUMENT_TIMEOUT_SECONDS``, ``OCR_MAX_PAGES_PER_DOCUMENT``).
- Confidence is whatever the engine reports per text line (mean per page). It is an engine score, not a
  measured accuracy, and is never invented when the engine does not provide one.
"""
import logging
import math
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, List, Optional

log = logging.getLogger("advancerag.rag.ocr")


class OcrError(Exception):
    """OCR failed for one page or image (never raised for 'no text found', which is a normal result)."""


class OcrTimeout(OcrError):
    pass


@dataclass
class OcrResult:
    text: str
    confidence: Optional[float]  # mean of the engine's per-line scores, 0..1; None when unavailable
    lines: int


class OcrEngine(ABC):
    name: str

    @abstractmethod
    def recognize(self, image: Any) -> OcrResult:
        """Blocking OCR of a PIL image. May raise OcrError."""

    def warm_up(self) -> None:
        return None


def _rows(boxes: List[Any]) -> List[List[Any]]:
    """Group detected text boxes into visual rows (same vertical band), left to right."""
    items = []
    for box, text, conf in boxes:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append({"x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys), "text": text, "conf": conf})
    items.sort(key=lambda b: (b["y0"] + b["y1"]) / 2)
    rows: List[List[dict]] = []
    for it in items:
        mid = (it["y0"] + it["y1"]) / 2
        if rows:
            last = rows[-1]
            last_mid = sum((b["y0"] + b["y1"]) / 2 for b in last) / len(last)
            height = max(b["y1"] - b["y0"] for b in last)
            if abs(mid - last_mid) <= 0.5 * height:
                last.append(it)
                continue
        rows.append([it])
    for r in rows:
        r.sort(key=lambda b: b["x0"])
    return rows


def rows_to_text(rows: List[List[dict]]) -> str:
    """Join rows into lines. Boxes in one row separated by a wide gap are columns/cells: ' | ' keeps the
    row relationship (useful for tables); close boxes are joined with a space."""
    lines = []
    for r in rows:
        parts = [r[0]["text"]]
        for prev, cur in zip(r, r[1:]):
            height = max(prev["y1"] - prev["y0"], cur["y1"] - cur["y0"], 1)
            gap = cur["x0"] - prev["x1"]
            parts.append(" | " if gap >= 2.0 * height else " ")
            parts.append(cur["text"])
        lines.append("".join(parts).strip())
    return "\n".join(line for line in lines if line)


class RapidOcrEngine(OcrEngine):
    name = "rapidocr-onnxruntime"

    def __init__(self):
        self._engine = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._engine is None:
                from rapidocr_onnxruntime import RapidOCR  # imported lazily: heavy, optional

                log.info("ocr_engine_loading", extra={"engine": self.name})
                self._engine = RapidOCR()
        return self._engine

    def warm_up(self) -> None:
        self._load()

    def recognize(self, image: Any) -> OcrResult:
        import numpy as np

        engine = self._load()
        try:
            arr = np.asarray(image.convert("RGB"))[:, :, ::-1]  # RapidOCR expects BGR
            result, _ = engine(arr)
        except Exception as exc:  # engine/ONNX failures
            raise OcrError("ocr engine failure") from exc
        if not result:
            return OcrResult("", None, 0)
        rows = _rows(result)
        scores = []
        for _, _, c in result:
            try:
                scores.append(float(c))  # some RapidOCR versions return the score as a string
            except (TypeError, ValueError):
                pass
        conf = (sum(scores) / len(scores)) if scores else None  # None if the engine gave no usable score
        return OcrResult(rows_to_text(rows), conf, len(result))


_engine: Optional[OcrEngine] = None
_executor: Optional[ThreadPoolExecutor] = None
_engine_lock = threading.Lock()


def get_engine() -> OcrEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = RapidOcrEngine()
        return _engine


def set_engine(engine: Optional[OcrEngine]) -> None:
    """Replace the engine (tests, alternative engines). ``None`` resets to the default."""
    global _engine
    with _engine_lock:
        _engine = engine


def _pool() -> ThreadPoolExecutor:
    global _executor
    with _engine_lock:
        if _executor is None:
            # One worker: OCR is CPU-bound and a hung page must not fan out into many stuck threads.
            _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
        return _executor


def _reset_pool() -> None:
    """After a timeout the worker thread may still be busy: abandon that executor and start a fresh one."""
    global _executor
    with _engine_lock:
        old, _executor = _executor, None
    if old is not None:
        old.shutdown(wait=False, cancel_futures=True)


def recognize_with_timeout(image: Any, timeout: float) -> OcrResult:
    fut = _pool().submit(get_engine().recognize, image)
    try:
        return fut.result(timeout=timeout)
    except FutureTimeout as exc:
        fut.cancel()
        _reset_pool()
        raise OcrTimeout(f"ocr exceeded {timeout:.0f}s") from exc


# ---------------------------------------------------------------- rasterising

_PDFIUM_LOCK = threading.Lock()  # PDFium is not thread-safe


def render_pdf_page(pdf: Any, index: int, dpi: int, max_pixels: int):
    """Rasterise one page of an open pypdfium2 document, capping the pixel count."""
    with _PDFIUM_LOCK:
        page = pdf[index]
        try:
            width, height = page.get_size()  # points (1/72 inch)
            scale = dpi / 72.0
            if width * height * scale * scale > max_pixels:
                scale = math.sqrt(max_pixels / (width * height))
            bitmap = page.render(scale=scale)
            return bitmap.to_pil()
        finally:
            page.close()


def open_pdfium(source: Any):
    import pypdfium2 as pdfium

    with _PDFIUM_LOCK:
        return pdfium.PdfDocument(source)
