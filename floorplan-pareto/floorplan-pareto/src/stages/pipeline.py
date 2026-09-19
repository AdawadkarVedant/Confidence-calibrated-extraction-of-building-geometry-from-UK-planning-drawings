"""The four extraction stages: triage, scale, geometry, link.

Rule: every function here must return a StageOutput with a confidence.
No exceptions, no silent failures.

Tip before adding OCR: run `pdffonts application.pdf` first. Most
planning drawings are exported from CAD and already have a text layer
(pdfplumber can read it directly, no OCR needed). Only scanned PDFs
need OCR - flag those with a REQUIRES_OCR field so later stages know.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from contracts import PropertyRecord, Stage, StageOutput

ROOT = Path(__file__).resolve().parents[2]
TRIAGE_CHECKPOINT = ROOT / "outputs" / "triage_model.pt"

# Same values used in train_triage.py to preprocess images.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_triage_model_cache: dict | None = None


def _load_triage_model() -> dict:
    """Load the trained triage model once, then reuse it (loading is slow)."""
    global _triage_model_cache
    if _triage_model_cache is not None:
        return _triage_model_cache

    import torch
    import torch.nn as nn
    from torchvision import models

    if not TRIAGE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"no triage checkpoint at {TRIAGE_CHECKPOINT} -- run "
            "src/stages/train_triage.py first"
        )

    ckpt = torch.load(TRIAGE_CHECKPOINT, map_location="cpu")
    classes = ckpt["classes"]

    # weights=None: we load our own trained weights right after, so no
    # need to also download ImageNet's.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _triage_model_cache = {
        "model": model,
        "temperature": ckpt["temperature"],
        "classes": classes,
    }
    return _triage_model_cache


def triage(record: PropertyRecord, page_image) -> StageOutput:
    """Classify one image (floor plan? elevation? etc) with a confidence score.

    page_image can be a PIL.Image already in memory, or a file path.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    cache = _load_triage_model()
    model, temperature, classes = cache["model"], cache["temperature"], cache["classes"]

    img = page_image if isinstance(page_image, Image.Image) else Image.open(page_image)
    img = img.convert("RGB")

    # No flip here - that's a training-only trick, not for real use.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = transform(img).unsqueeze(0)  # models expect a batch, so add that dimension

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits / temperature, dim=1)

    confidence, idx = probs.max(dim=1)
    page_class = classes[idx.item()]

    return StageOutput(
        Stage.TRIAGE,
        confidence=float(confidence.item()),
        payload={"page_class": page_class},
    )


def _run_lengths(row: np.ndarray) -> list[int]:
    """Count how long each stretch of same-value pixels is, along one row."""
    runs, current, length = [], row[0], 1
    for px in row[1:]:
        if px == current:
            length += 1
        else:
            runs.append(length)
            current, length = px, 1
    runs.append(length)
    return runs


def scale(record: PropertyRecord, scale_bar_image) -> StageOutput:
    """Read the drawing scale off the scale bar image.

    Only reads the scale bar (not the title block text - that would
    need OCR, not done here).

    How: find the row that looks most like a set of evenly-spaced ticks,
    measure how wide each tick is in pixels, and convert that to a
    scale (assumes the image was rendered at 150 dpi).
    """
    if scale_bar_image is None:
        return StageOutput(Stage.SCALE, confidence=0.0, payload={},
                            notes="no scale bar region available")

    gray = np.array(scale_bar_image.convert("L"))
    binary = gray < 128  # ink is dark -- see render/sheet.py's INK constant

    # Try every row, score it by how many tick-sized (>=8px) stripes it
    # has, and keep the best one. This ignores rows that are all text
    # or all border line, and finds the actual tick pattern.
    tick_lengths: list[int] = []
    for i in range(binary.shape[0]):
        lengths = [n for n in _run_lengths(binary[i, :]) if n >= 8]
        if len(lengths) > len(tick_lengths):
            tick_lengths = lengths

    if len(tick_lengths) < 2:
        return StageOutput(Stage.SCALE, confidence=0.0, payload={},
                            notes="no tick pattern detected")

    median_tick_px = float(np.median(tick_lengths))
    spread = float(np.std(tick_lengths) / max(median_tick_px, 1e-6))

    dpi = 150
    scale_denominator = round((1000.0 * dpi) / (25.4 * median_tick_px))
    confidence = float(np.clip(1.0 - 3.0 * spread, 0.0, 0.99))

    return StageOutput(
        Stage.SCALE,
        confidence=confidence,
        payload={"scale_denominator": scale_denominator},
    )


def _count_storeys(elevation_images: list) -> tuple[int | None, float]:
    """Count storeys by counting rows of windows in the elevation drawing(s)."""
    import cv2

    counts = []
    for img in elevation_images:
        gray = np.array(img.convert("L"))
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        h_img, w_img = gray.shape
        total_area = h_img * w_img
        centres = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            aspect = w / h if h else 0
            # Windows/doors are small and taller than wide - this filters
            # out the big building outline and roofline shapes.
            if 0.002 * total_area < area < 0.08 * total_area and 0.3 < aspect < 1.0:
                centres.append(y + h / 2)

        if not centres:
            continue
        centres.sort()
        rows = [centres[0]]
        for cy in centres[1:]:
            if cy - rows[-1] > 0.08 * h_img:
                rows.append(cy)
        counts.append(len(rows))

    if not counts:
        return None, 0.0
    best, freq = Counter(counts).most_common(1)[0]
    return best, freq / len(counts)


def geometry(record: PropertyRecord, floor_plan_image,
             elevation_images: list | None = None) -> StageOutput:
    """Measure floor area, room count, and storeys from the floor plan image.

    Needs a scale from the scale() stage first - if that's missing,
    this abstains too, rather than guessing.
    """
    import cv2

    scale_denominator = record.prediction("scale_denominator")
    if not scale_denominator:
        return StageOutput(Stage.GEOMETRY, confidence=0.0, payload={},
                            notes="no recovered scale to measure against")

    gray = np.array(floor_plan_image.convert("L"))
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return StageOutput(Stage.GEOMETRY, confidence=0.0, payload={},
                            notes="no closed contour found")

    # The biggest outer shape is the building outline. Each shape
    # directly inside it is one room - so room count and floor area
    # both come from the same contour search.
    areas = [cv2.contourArea(c) for c in contours]
    hier = hierarchy[0]
    top_level = [i for i, h in enumerate(hier) if h[3] == -1]
    outer_idx = max(top_level, key=lambda i: areas[i])
    outer_area_px = areas[outer_idx]

    dpi = 150
    ppm = (1000.0 / scale_denominator) / 25.4 * dpi
    ground_floor_area_m2 = outer_area_px / (ppm ** 2)

    room_count = sum(1 for h in hier if h[3] == outer_idx)

    # If there's a second big shape almost as large as the outline, the
    # page probably has 2 drawings side by side - lower confidence.
    top_areas = sorted((areas[i] for i in top_level), reverse=True)
    ambiguous_page = len(top_areas) > 1 and top_areas[1] > 0.6 * top_areas[0]
    plausible = 15.0 <= ground_floor_area_m2 <= 400.0

    storeys, storey_conf = (
        _count_storeys(elevation_images) if elevation_images else (None, 0.0)
    )

    confidence = 0.85
    if not plausible:
        confidence -= 0.5
    if ambiguous_page:
        confidence -= 0.3
    if storeys is None:
        confidence -= 0.15
    confidence = float(np.clip(confidence, 0.0, 0.99))

    payload = {
        "ground_floor_area_m2": round(float(ground_floor_area_m2), 3),
        "room_count": int(room_count),
    }
    if storeys is not None:
        payload["storeys"] = storeys

    return StageOutput(Stage.GEOMETRY, confidence=confidence, payload=payload)


def link(record: PropertyRecord, address: str, candidates: list[str]) -> StageOutput:
    """Match an address string to the best candidate from a list.

    Confidence = gap between the best match and the second-best match.
    A close call (e.g. two near-identical addresses on the same street)
    correctly gets low confidence.

    Note: matches one record at a time. Doesn't yet stop two records
    from both claiming the same candidate - that would need to look at
    all records together in one batch.
    """
    import difflib

    def normalise(s: str) -> str:
        return " ".join(s.upper().split())

    if not candidates:
        return StageOutput(Stage.LINK, confidence=0.0, payload={},
                            notes="no candidates to match against")

    target = normalise(address)
    scored = sorted(
        ((c, difflib.SequenceMatcher(None, target, normalise(c)).ratio())
         for c in candidates),
        key=lambda cs: cs[1], reverse=True,
    )
    best_addr, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    margin = best_score - second_score

    return StageOutput(
        Stage.LINK,
        confidence=float(margin),
        payload={"matched_address": best_addr, "match_score": round(float(best_score), 4)},
    )
