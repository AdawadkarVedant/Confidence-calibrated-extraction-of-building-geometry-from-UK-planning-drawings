"""Draw a Plan as a full A3 sheet image: floor plan, elevations, site plan, etc.

Each drawn region is recorded with its class, its pixel box, and its own
scale - so the corpus comes with free labels for training a detector.

Note: the site plan is drawn at a different scale (1:1250) than the main
drawings (1:100), just like a real architect's sheet. A stage that
assumes one scale for the whole page will get the site plan area wrong.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from PIL import Image, ImageDraw, ImageFont

from plans import Plan, bounds

# A3 portrait at 150 dpi. Enough for OCR of room labels, small enough to
# render a few hundred sheets in a couple of minutes.
A3_MM = (297.0, 420.0)
DEFAULT_DPI = 150

INK = (20, 20, 20)
PAPER = (255, 255, 255)

# Region classes. These are the triage labels.
REGION_CLASSES = [
    "floor_plan", "elevation", "section", "site_plan",
    "title_block", "scale_bar", "north_arrow", "notes",
]

# Tries each font name in order so this works on Linux, Windows, or
# wherever else - not just one machine with one exact font path.
_FONT_NAMES = {
    False: ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"],
    True: ["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf"],
}
_font_cache: dict[tuple[int, bool], ImageFont.ImageFont] = {}


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a font, trying several names, cached by (size, bold)."""
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    for name in _FONT_NAMES[bold]:
        try:
            font = ImageFont.truetype(name, size)
            break
        except OSError:
            continue
    else:
        # None of those fonts found - fall back to PIL's built-in default
        # rather than crashing.
        font = ImageFont.load_default()
    _font_cache[key] = font
    return font


@dataclass
class Region:
    """One drawing region on the sheet: its class, box, and scale."""

    cls: str
    bbox: tuple[int, int, int, int]       # pixels, (x0, y0, x1, y1)
    scale_denominator: int | None = None  # None where scale is meaningless
    note: str = ""

    def yolo(self, w: int, h: int) -> str:
        """This region's box in YOLO label format (class cx cy w h, 0-1 scale)."""
        x0, y0, x1, y1 = self.bbox
        return (f"{REGION_CLASSES.index(self.cls)} "
                f"{(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")


class SheetRenderer:
    """Draws all the pieces of one A3 sheet and records where each one went."""

    def __init__(self, dpi: int = DEFAULT_DPI, seed: int | None = None):
        self.dpi = dpi
        self.rng = random.Random(seed)
        self.px_w = int(A3_MM[0] / 25.4 * dpi)
        self.px_h = int(A3_MM[1] / 25.4 * dpi)

    # ------------------------------------------------------------ helpers

    def _px_per_metre(self, scale_denominator: int) -> float:
        """How many pixels equal 1 metre, at this drawing scale and dpi."""
        mm_per_metre = 1000.0 / scale_denominator
        return mm_per_metre / 25.4 * self.dpi

    def _transform(self, reference, box, ppm):
        """Build one metres->pixels conversion, shared by every shape in a drawing.

        Must be the SAME conversion for every room, or rooms end up
        drawn nested inside each other instead of side by side.
        """
        x0, y0, x1, y1 = bounds(reference)
        bx0, by0, bx1, by1 = box
        w_px, h_px = (x1 - x0) * ppm, (y1 - y0) * ppm
        ox = bx0 + ((bx1 - bx0) - w_px) / 2
        oy = by0 + ((by1 - by0) - h_px) / 2

        def to_px(poly):
            return [(ox + (px - x0) * ppm, oy + h_px - (py - y0) * ppm)
                    for px, py in poly]

        return to_px

    @staticmethod
    def _extent(*point_lists) -> tuple[int, int, int, int]:
        """Tight bounding box around the actual drawn pixels (not the empty slot)."""
        pts = [p for lst in point_lists for p in lst]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    # ------------------------------------------------------------ drawings

    def _floor_plan(self, d: ImageDraw.ImageDraw, plan: Plan, box, scale: int,
                    with_labels: bool = True) -> tuple[int, int, int, int]:
        """Draw the floor plan: rooms, walls, labels, and dimension numbers."""
        ppm = self._px_per_metre(scale)
        wall = max(2, int(0.30 * ppm))              # 300 mm walls
        to_px = self._transform(plan.footprint, box, ppm)

        for room in plan.rooms:
            pts = to_px(room.polygon)
            d.polygon(pts, outline=INK, fill=(252, 252, 252), width=max(1, wall // 3))
            if with_labels:
                font = _font(max(7, int(ppm * 0.16)))
                room_w = max(p[0] for p in pts) - min(p[0] for p in pts)
                room_h = max(p[1] for p in pts) - min(p[1] for p in pts)
                tx0, ty0, tx1, ty1 = font.getbbox(room.label)
                # Skip the label if it wouldn't fit - better an unlabelled
                # room than text spilling into the walls.
                if tx1 - tx0 < room_w * 0.9 and ty1 - ty0 < room_h * 0.9:
                    cx = sum(p[0] for p in pts) / len(pts)
                    cy = sum(p[1] for p in pts) / len(pts)
                    d.text((cx, cy), room.label, font=font, fill=INK, anchor="mm")

        outer = to_px(plan.footprint)
        d.line(outer + [outer[0]], fill=INK, width=wall)

        # Width/depth numbers in mm, like a real drawing has.
        f = _font(max(9, int(ppm * 0.20)), bold=True)
        xs = [p[0] for p in outer]
        ys = [p[1] for p in outer]
        d.text(((min(xs) + max(xs)) / 2, max(ys) + wall * 2),
               f"{int(plan.width * 1000)}", font=f, fill=INK, anchor="ma")
        d.text((max(xs) + wall * 2, (min(ys) + max(ys)) / 2),
               f"{int(plan.depth * 1000)}", font=f, fill=INK, anchor="lm")

        return self._extent(outer)

    def _elevation(self, d: ImageDraw.ImageDraw, plan: Plan, box, scale: int,
                   title: str) -> tuple[int, int, int, int]:
        """Draw a simple building facade: walls, roof, windows, and a door."""
        ppm = self._px_per_metre(scale)
        storey_h = 2.6
        total_h = plan.storeys * storey_h
        w = plan.width

        # Include the roof in the reference outline so the whole elevation
        # fits the slot rather than overflowing it.
        roof = 1.6
        poly = [(0, 0), (w, 0), (w, total_h + roof), (0, total_h + roof)]
        to_px = self._transform(poly, box, ppm)
        body = to_px([(0, 0), (w, 0), (w, total_h), (0, total_h)])
        d.polygon(body, outline=INK, fill=(250, 250, 250), width=2)

        x0 = min(p[0] for p in body)
        y1 = max(p[1] for p in body)                # ground line
        ground = [(x0 - 10, y1), (x0 + w * ppm + 10, y1)]
        d.line(ground, fill=INK, width=3)

        extra = [body, ground]
        if self.rng.random() < 0.75:
            apex = (x0 + w * ppm / 2, y1 - (total_h + roof) * ppm)
            ridge = [(x0, y1 - total_h * ppm), apex, (x0 + w * ppm, y1 - total_h * ppm)]
            d.line(ridge, fill=INK, width=3)
            extra.append(ridge)

        for storey in range(plan.storeys):
            base = y1 - (storey + 1) * storey_h * ppm
            for i in range(2):
                wx = x0 + (0.22 + i * 0.45) * w * ppm
                d.rectangle([wx, base + 0.5 * ppm,
                             wx + 0.9 * ppm, base + 1.8 * ppm],
                            outline=INK, width=2)
            if storey == 0:
                dx = x0 + 0.70 * w * ppm
                d.rectangle([dx, y1 - 2.0 * ppm, dx + 0.9 * ppm, y1],
                            outline=INK, width=2)

        bbox = self._extent(*extra)
        d.text(((bbox[0] + bbox[2]) / 2, bbox[3] + 8), title,
               font=_font(11), fill=INK, anchor="ma")
        return bbox

    def _site_plan(self, d: ImageDraw.ImageDraw, plan: Plan, box,
                   scale: int) -> tuple[int, int, int, int]:
        """Draw the property plus a few neighbours, at a different scale than the rest of the sheet."""
        ppm = self._px_per_metre(scale)
        bx0, by0, bx1, by1 = box
        d.rectangle(box, outline=(150, 150, 150), width=1, fill=(248, 246, 240))

        y = by1 - 20
        drawn = []
        for i in range(-3, 4):
            w_m = plan.width * self.rng.uniform(0.85, 1.15)
            d_m = plan.depth * self.rng.uniform(0.85, 1.15)
            if i == 0:
                w_m, d_m = plan.width, plan.depth
            cx = (bx0 + bx1) / 2 + i * (plan.width + 1.5) * ppm
            x0, y0 = cx - w_m * ppm / 2, y - d_m * ppm
            if x0 < bx0 or cx + w_m * ppm / 2 > bx1:
                continue
            fill = (235, 120, 100) if i == 0 else (225, 220, 210)
            d.rectangle([x0, y0, x0 + w_m * ppm, y], outline=INK, fill=fill, width=1)
            drawn += [(x0, y0), (x0 + w_m * ppm, y)]

        d.line([(bx0 + 5, y + 8), (bx1 - 5, y + 8)], fill=(120, 120, 120), width=2)
        d.text(((bx0 + bx1) / 2, by1 - 4), f"SITE PLAN @ 1:{scale}",
               font=_font(10), fill=INK, anchor="md")
        return self._extent(drawn) if drawn else box

    def _scale_bar(self, d: ImageDraw.ImageDraw, box, scale: int) -> None:
        """Draw a 0-10m tick-marked scale bar."""
        ppm = self._px_per_metre(scale)
        x0, y0, x1, y1 = box
        bar_y = y0 + (y1 - y0) * 0.45
        h = max(4, int((y1 - y0) * 0.25))
        for m in range(10):
            xa, xb = x0 + m * ppm, x0 + (m + 1) * ppm
            if xb > x1:
                break
            d.rectangle([xa, bar_y, xb, bar_y + h],
                        fill=INK if m % 2 == 0 else PAPER, outline=INK, width=1)
        f = _font(9)
        for m in range(0, 11, 2):
            x = x0 + m * ppm
            if x > x1:
                break
            d.text((x, bar_y - 2), f"{m}m", font=f, fill=INK, anchor="mb")

    def _north_arrow(self, d: ImageDraw.ImageDraw, box) -> None:
        """Draw a circle with an arrow pointing north."""
        x0, y0, x1, y1 = box
        cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, min(x1 - x0, y1 - y0) / 2 - 3
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=INK, width=2)
        d.polygon([(cx, cy - r * 0.8), (cx - r * 0.3, cy + r * 0.5),
                   (cx, cy + r * 0.2), (cx + r * 0.3, cy + r * 0.5)],
                  fill=INK)

    def _title_block(self, d: ImageDraw.ImageDraw, box, plan: Plan,
                     scale: int, meta: dict) -> None:
        """Draw the title block: practice name, project, scale, date. All fake data."""
        x0, y0, x1, y1 = box
        d.rectangle(box, outline=INK, width=2, fill=PAPER)
        d.line([(x0 + (x1 - x0) * 0.42, y0), (x0 + (x1 - x0) * 0.42, y1)],
               fill=INK, width=1)
        d.text((x0 + 8, y0 + 8), meta["practice"], font=_font(13, bold=True), fill=INK)
        d.text((x0 + 8, y0 + 26), "ARCHITECTURAL DESIGN", font=_font(9), fill=INK)

        cx = x0 + (x1 - x0) * 0.42 + 8
        f = _font(9)
        d.text((cx, y0 + 6), f"PROJECT   {meta['address']}", font=f, fill=INK)
        d.text((cx, y0 + 20), f"DRAWING   {meta['drawing_title']}", font=f, fill=INK)
        d.text((cx, y0 + 34), f"SCALE  1:{scale} @ A3     "
                              f"DWG {meta['drawing_no']}     {meta['date']}",
               font=f, fill=INK)

    # ---------------------------------------------------------------- main

    def render(self, plan: Plan, meta: dict,
               main_scale: int = 100, site_scale: int = 1250,
               include_site_plan: bool = True,
               include_scale_bar: bool = True,
               include_room_labels: bool = True) -> tuple[Image.Image, list[Region]]:
        """Draw one full sheet. Returns the image and the list of regions on it."""
        img = Image.new("RGB", (self.px_w, self.px_h), PAPER)
        d = ImageDraw.Draw(img)
        W, H = self.px_w, self.px_h
        m = int(0.03 * W)                                  # margin
        regions: list[Region] = []

        d.rectangle([m, m, W - m, H - m], outline=(180, 180, 180), width=1)

        # Left column: floor plan.
        fp_box = (m + 10, m + 10, int(W * 0.46), int(H * 0.60))
        fp = self._floor_plan(d, plan, fp_box, main_scale, include_room_labels)
        d.text(((fp[0] + fp[2]) / 2, fp[3] + 62),
               "PROPOSED GROUND FLOOR PLAN", font=_font(12), fill=INK, anchor="ma")
        regions.append(Region("floor_plan", fp, main_scale))

        # Right column: elevations, stacked.
        right_x0, right_x1 = int(W * 0.50), W - m - 10
        ev1 = (right_x0, m + 20, right_x1, int(H * 0.26))
        b1 = self._elevation(d, plan, ev1, main_scale, "PROPOSED REAR ELEVATION")
        regions.append(Region("elevation", b1, main_scale))

        ev2 = (right_x0, int(H * 0.28), right_x1, int(H * 0.52))
        b2 = self._elevation(d, plan, ev2, main_scale, "PROPOSED SIDE ELEVATION")
        regions.append(Region("elevation", b2, main_scale))

        # Notes block (materials text).
        nb = (right_x0, int(H * 0.55), right_x1, int(H * 0.66))
        f = _font(10)
        d.text((nb[0], nb[1]), "PROPOSED MATERIALS", font=_font(11, bold=True), fill=INK)
        for i, line in enumerate([
            "WALLS - BRICK OUTER SKIN TO MATCH EXISTING.",
            "ROOF - FLAT ROOF WITH PARAPET UPSTAND.",
            "WINDOWS - POWDER-COATED ALUMINIUM, ANTHRACITE GREY.",
            "RAINWATER GOODS - ANTHRACITE GREY.",
        ]):
            d.text((nb[0], nb[1] + 18 + i * 15), line, font=f, fill=INK)
        regions.append(Region("notes", nb))

        if include_site_plan:
            sp = (right_x0, int(H * 0.70), right_x0 + int(W * 0.24), int(H * 0.80))
            sp = self._site_plan(d, plan, sp, site_scale)
            regions.append(Region("site_plan", sp, site_scale,
                                  note="different scale from the sheet"))

        if include_scale_bar:
            ppm = self._px_per_metre(main_scale)
            sb = (m + 20, int(H * 0.80), m + 20 + int(10 * ppm), int(H * 0.80) + 26)
            self._scale_bar(d, sb, main_scale)
            regions.append(Region("scale_bar", sb, main_scale))

        na = (m + 20, int(H * 0.86), m + 70, int(H * 0.86) + 50)
        self._north_arrow(d, na)
        regions.append(Region("north_arrow", na))

        tb = (m, H - m - 70, W - m, H - m)
        self._title_block(d, tb, plan, main_scale, meta)
        regions.append(Region("title_block", tb, main_scale))

        return img, regions


PRACTICES = [
    "HARTLAND & CO", "BRIDGE DESIGN STUDIO", "OKEHAMPTON DRAWING OFFICE",
    "M. ASHWORTH DESIGN", "CLERESTORY ARCHITECTURE", "TOPSHAM DESIGN",
]
STREETS = ["QUEENS ROAD", "MOUNT PLEASANT", "HEAVITREE ROAD", "PENNSYLVANIA RD",
           "ALPHINGTON STREET", "BLACKBOY ROAD"]


def synthetic_meta(rng: random.Random, plan: Plan) -> dict:
    """Make up fake title-block details (practice, address, date). No real data."""
    return {
        "practice": rng.choice(PRACTICES),
        "address": f"{rng.randint(1, 180)} {rng.choice(STREETS)}, EXETER",
        "drawing_title": "PROPOSED PLAN AND ELEVATIONS",
        "drawing_no": f"{rng.randint(100, 999)}/{rng.randint(1, 40):02d}",
        "date": f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.26",
    }


def region_summary(regions: list[Region]) -> str:
    """One-line summary of all regions, for debug printing."""
    return ", ".join(f"{r.cls}"
                     + (f"@1:{r.scale_denominator}" if r.scale_denominator else "")
                     for r in regions)


if __name__ == "__main__":
    from plans import procedural

    rng = random.Random(3)
    plan = procedural("DEMO0001", seed=3)
    r = SheetRenderer(seed=3)
    img, regions = r.render(plan, synthetic_meta(rng, plan))
    img.save("/tmp/demo_sheet.png")
    print(f"{img.size[0]}x{img.size[1]} px, {len(regions)} regions")
    print(region_summary(regions))
    print("truth:", plan.truth())
