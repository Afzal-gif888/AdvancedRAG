"""OCR accuracy on labelled synthetic samples: character error rate (CER) per condition and per script, whether the engine's
confidence tracks real error, page-order preservation on scanned/mixed PDFs, and timing.

    python scripts/eval_ocr.py [--out evaluation/results]

What it is: every sample is rendered from KNOWN text, so the ground truth is exact. Conditions vary font, size, noise, blur,
resolution, rotation and JPEG compression. Scripts other than Latin are only tested if a suitable system font exists.
What it is not: it says nothing about handwriting, photographs of documents, complex layouts or real-world scan quality, and
the numbers belong to this machine's fonts and to the OCR model that ships with rapidocr-onnxruntime (a Chinese/English
model family; other scripts are measured here precisely to show whether they work).
CER = edit distance / reference length after Unicode NFC and whitespace collapsing; case is kept (it matters for identifiers).
"""
import argparse
import io
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "rag-core"), str(ROOT / "shared")]
os.environ.setdefault("ADVANCERAG_ENV_FILE", "none")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite://")
os.environ.setdefault("LOG_LEVEL", "WARNING")

FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"

ENGLISH = [
    ["The annual audit is performed by the firm Brandt and Partners.", "Invoice 88231 is due in 45 days, payable by bank transfer.", "Contract number 77412 expires on 30 November."],
    ["Hotel stays are reimbursed up to 180 euros per night in all European cities.", "Receipts must be submitted within 30 days of the end of the trip.", "Security incidents must be reported within 24 hours."],
    ["Battery life: 14 months. Weight: 320 grams. Rating IP67.", "Article number AUX1-4821-S ships from Rotterdam.", "Temperature range -20 to 45 degrees Celsius."],
    ["Quarterly revenue grew by eighteen percent in Europe.", "The warehouse in Rotterdam opened in March 2019.", "Contact: billing@supplier.example or +31 20 555 0188."],
]
OTHER = {
    "german_latin1": ["Die Rechnung über 1.250,00 Euro ist fällig, Größe M, Straße 12.", "Käse, Müller und Zürich liegen südlich von München."],
    "french_latin1": ["Les employés reçoivent vingt-huit jours de congé payé à l'été.", "Le réseau est opérationnel depuis décembre, garçon!"],
    "spanish_latin1": ["La garantía cubre defectos de fabricación durante veinticuatro meses.", "¿Cuántos años tiene la niña? Mañana será el cumpleaños."],
    "russian_cyrillic": ["Срок гарантии на устройство составляет двадцать четыре месяца.", "Отчёт нужно сдать до тридцатого ноября."],
    "greek": ["Η εγγύηση καλύπτει ελαττώματα κατασκευής για δύο χρόνια.", "Η αναφορά παραδίδεται μέχρι τις τριάντα Νοεμβρίου."],
    "chinese_simplified": ["保修期为二十四个月,覆盖制造缺陷。", "报告必须在十一月三十日之前提交。"],
    "korean": ["보증 기간은 24개월이며 제조 결함을 보장합니다.", "보고서는 11월 30일까지 제출해야 합니다."],
}
SCRIPT_FONT = {"chinese_simplified": "msyh.ttc", "korean": "malgun.ttf"}


def load_font(name: str, size: int):
    from PIL import ImageFont

    path = FONTS / name
    return ImageFont.truetype(str(path), size) if path.exists() else None


def render(lines, font, width=1500, noise=False, blur=0.0, scale=1.0, rotate=0.0, size=34):
    from PIL import Image, ImageDraw, ImageFilter

    h = 80 + len(lines) * (size + 30)
    img = Image.new("L", (width, h), 255)
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((50, 40 + i * (size + 30)), line, fill=0, font=font)
    if rotate:
        img = img.rotate(rotate, expand=True, fillcolor=255, resample=Image.BICUBIC)
    if scale != 1.0:
        img = img.resize((max(int(img.width * scale), 1), max(int(img.height * scale), 1))).resize(img.size)
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if noise:
        rnd = random.Random(7)
        px = img.load()
        for _ in range(img.width * img.height // 40):
            px[rnd.randrange(img.width), rnd.randrange(img.height)] = rnd.randrange(0, 255)
    return img.convert("RGB")


def jpeg(img, quality):
    from PIL import Image

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


CONDITIONS = {  # name -> (font file, size, render kwargs, post)
    "clean_arial_34": ("arial.ttf", 34, {}, None),
    "arial_22_small": ("arial.ttf", 22, {}, None),
    "courier_30": ("cour.ttf", 30, {}, None),
    "times_34": ("times.ttf", 34, {}, None),
    "noise": ("arial.ttf", 34, {"noise": True}, None),
    "blur_1.5": ("arial.ttf", 34, {"blur": 1.5}, None),
    "lowres_0.5": ("arial.ttf", 34, {"scale": 0.5}, None),
    "lowres_0.35": ("arial.ttf", 34, {"scale": 0.35}, None),
    "rotate_2deg": ("arial.ttf", 34, {"rotate": 2.0}, None),
    "rotate_5deg": ("arial.ttf", 34, {"rotate": 5.0}, None),
    "jpeg_q30": ("arial.ttf", 34, {}, 30),
}


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return None if sx == 0 or sy == 0 else round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "results"))
    args = ap.parse_args()

    from rag_core import ocr
    from rag_core.config import settings
    from rag_core.evaluation import cer, wer
    from rag_core.parsing import PDF, parse_document

    engine = ocr.RapidOcrEngine()
    engine.recognize(render(["warm up"], load_font("arial.ttf", 30)))  # load models outside the timings
    rows = []

    def run(group, name, img, truth):
        t = time.perf_counter()
        res = engine.recognize(img)
        secs = time.perf_counter() - t
        ref = "\n".join(truth)
        rows.append({"group": group, "name": name, "cer": round(cer(ref, res.text), 4), "cer_casefold": round(cer(ref, res.text, fold=True), 4),
                     "wer": round(wer(ref, res.text), 4), "confidence": None if res.confidence is None else round(res.confidence, 3), "seconds": round(secs, 2),
                     "chars": len(ref)})

    for cond, (font_name, size, kw, jq) in CONDITIONS.items():
        font = load_font(font_name, size)
        if font is None:
            print(f"skip {cond}: font {font_name} not installed")
            continue
        for k, truth in enumerate(ENGLISH):
            img = render(truth, font, size=size, **kw)
            run("english_conditions", f"{cond}#{k}", jpeg(img, jq) if jq else img, truth)
    for script, samples in OTHER.items():
        font = load_font(SCRIPT_FONT.get(script, "arial.ttf"), 34)
        if font is None:
            print(f"skip {script}: no font")
            continue
        run("scripts", script, render(samples, font), samples)

    # page order and page numbers on a scanned PDF and a mixed PDF
    sys.path.insert(0, str(ROOT / "rag-core" / "tests"))
    from fixtures import mixed_pdf, scanned_pdf

    settings.OCR_ENABLED = True
    ocr.set_engine(engine)
    markers = ["Alpha 1111", "Bravo 2222", "Charlie 3333", "Delta 4444"]
    pages = [render([f"Page marker {m}", "The quick brown fox jumps over the lazy dog."], load_font("arial.ttf", 34)) for m in markers]
    order = {}
    for label, pdf in (("scanned_pdf", scanned_pdf(pages)), ("mixed_pdf", mixed_pdf([f"Text page marker {markers[0]}", pages[1], f"Text page marker {markers[2]}", pages[3]]))):
        doc = parse_document(pdf, PDF, settings)
        found = []
        for p in doc.pages:
            hit = next((m for m in markers if m.split()[1] in p.text), None)
            found.append((p.number, hit, p.method))
        order[label] = {"pages": found, "in_order": [f[1] for f in found] == markers and [f[0] for f in found] == [1, 2, 3, 4],
                        "warnings": [w["code"] for w in doc.report.warnings], "ocr_pages": doc.report.ocr_pages, "native_pages": doc.report.native_pages}

    def agg(sel):
        sel = list(sel)
        if not sel:
            return None
        tot = sum(r["chars"] for r in sel)
        return {"samples": len(sel), "mean_cer": round(sum(r["cer"] for r in sel) / len(sel), 4), "micro_cer": round(sum(r["cer"] * r["chars"] for r in sel) / tot, 4),
                "max_cer": max(r["cer"] for r in sel), "mean_confidence": round(sum(r["confidence"] or 0 for r in sel) / len(sel), 3),
                "mean_seconds": round(sum(r["seconds"] for r in sel) / len(sel), 2)}

    by_cond = {c: agg(r for r in rows if r["group"] == "english_conditions" and r["name"].split("#")[0] == c) for c in CONDITIONS}
    scripts = {r["name"]: {k: r[k] for k in ("cer", "wer", "confidence")} for r in rows if r["group"] == "scripts"}
    eng = [r for r in rows if r["group"] == "english_conditions"]
    result = {
        "stamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "engine": engine.name, "machine": {"platform": platform.platform(), "cpus": os.cpu_count(), "python": platform.python_version()},
        "method": "synthetic samples rendered from known text; CER after NFC + whitespace collapsing, case kept; see the script docstring for limits",
        "overall_english": agg(eng), "english_by_condition": by_cond, "scripts": scripts,
        "confidence_vs_accuracy": {"pearson_confidence_vs_one_minus_cer_english": pearson([r["confidence"] or 0 for r in eng], [1 - min(r["cer"], 1) for r in eng]), "n": len(eng)},
        "page_order": order, "samples": rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}-ocr.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf8")
    print(json.dumps({k: result[k] for k in ("overall_english", "english_by_condition", "scripts", "confidence_vs_accuracy")}, indent=1, ensure_ascii=False))
    print(json.dumps({k: {"in_order": v["in_order"], "warnings": v["warnings"]} for k, v in order.items()}))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
