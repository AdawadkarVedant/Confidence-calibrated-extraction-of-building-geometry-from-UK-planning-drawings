"""Plan geometry: a normalised internal form, a ResPlan adapter, a fallback.

The renderer never touches ResPlan directly. It consumes `Plan`, which is
deliberately minimal -- exterior footprint plus labelled room polygons, all
in metres. Two things produce a `Plan`:

  from_resplan()   adapter for the real dataset
  procedural()     self-contained generator, no dependencies

The fallback exists so the renderer and everything downstream can be built
and tested before ResPlan is downloaded -- the same reason `synthetic.py`
existed for the optimisation layer. Never let an external dependency block
the component you are actually being judged on.

ResPlan ships `resplan_utils.py` and a pickle whose exact key names should
be confirmed against the release you download. `from_resplan` is written
defensively and names every key it expects in `RESPLAN_KEYS`, so fixing it
is one edit rather than an archaeology exercise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

Point = tuple[float, float]
Polygon = list[Point]

ROOM_LABELS = [
    "LIVING ROOM", "KITCHEN", "DINING", "BEDROOM", "BATHROOM",
    "WC", "HALL", "UTILITY", "STUDY", "LOBBY",
]


def polygon_area(poly: Polygon) -> float:
    """Shoelace. Absolute, so winding order does not matter."""
    n = len(poly)
    a = sum(poly[i][0] * poly[(i + 1) % n][1] - poly[(i + 1) % n][0] * poly[i][1]
            for i in range(n))
    return abs(a) / 2.0


def bounds(poly: Polygon) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


@dataclass
class Room:
    label: str
    polygon: Polygon                     # metres

    @property
    def area(self) -> float:
        return polygon_area(self.polygon)


@dataclass
class Plan:
    """One dwelling in metres, origin at the footprint's bottom-left."""

    plan_id: str
    footprint: Polygon
    rooms: list[Room] = field(default_factory=list)
    storeys: int = 2
    source: str = "procedural"

    @property
    def ground_floor_area_m2(self) -> float:
        return polygon_area(self.footprint)

    @property
    def width(self) -> float:
        x0, _, x1, _ = bounds(self.footprint)
        return x1 - x0

    @property
    def depth(self) -> float:
        _, y0, _, y1 = bounds(self.footprint)
        return y1 - y0

    def truth(self) -> dict:
        """Exactly what the extraction pipeline is trying to recover."""
        return {
            "ground_floor_area_m2": round(self.ground_floor_area_m2, 3),
            "storeys": self.storeys,
            "room_count": len(self.rooms),
            "width_m": round(self.width, 3),
            "depth_m": round(self.depth, 3),
        }


# --------------------------------------------------------------- ResPlan

# Confirm against the release you download (see resplan_utils.py and the
# demo notebook). One edit here fixes the whole adapter.
RESPLAN_KEYS = {
    "rooms": ("rooms", "spaces"),
    "polygon": ("polygon", "poly", "coords", "points"),
    "label": ("label", "type", "category", "name"),
    "outline": ("outline", "footprint", "boundary", "exterior"),
    "id": ("id", "plan_id", "index"),
}


def _first(d: dict, names: tuple[str, ...], default=None):
    for n in names:
        if n in d:
            return d[n]
    return default


def from_resplan(raw: dict, plan_id: str | None = None) -> Plan:
    """Adapt one ResPlan record.

    ResPlan gives metric-scale coordinates in metres, so the conversion is
    thin. If your release provides pixel coordinates plus a scale factor,
    apply it here and nowhere else.
    """
    rooms: list[Room] = []
    for r in _first(raw, RESPLAN_KEYS["rooms"], []) or []:
        poly = _first(r, RESPLAN_KEYS["polygon"])
        if not poly or len(poly) < 3:
            continue
        label = str(_first(r, RESPLAN_KEYS["label"], "ROOM")).upper()
        rooms.append(Room(label, [(float(x), float(y)) for x, y in poly]))

    outline = _first(raw, RESPLAN_KEYS["outline"])
    if outline and len(outline) >= 3:
        footprint = [(float(x), float(y)) for x, y in outline]
        approximated = False
    elif rooms:
        # Bounding box of all rooms. Crude: it overstates area on L-shaped
        # plans, which is exactly the case that matters. Recorded in the
        # manifest so these records can be excluded from area evaluation.
        xs = [p[0] for rm in rooms for p in rm.polygon]
        ys = [p[1] for rm in rooms for p in rm.polygon]
        footprint = [(min(xs), min(ys)), (max(xs), min(ys)),
                     (max(xs), max(ys)), (min(xs), max(ys))]
        approximated = True
    else:
        raise ValueError("record has neither an outline nor usable rooms")

    x0, y0, _, _ = bounds(footprint)
    footprint = [(x - x0, y - y0) for x, y in footprint]
    for rm in rooms:
        rm.polygon = [(x - x0, y - y0) for x, y in rm.polygon]

    plan = Plan(
        plan_id=str(plan_id or _first(raw, RESPLAN_KEYS["id"], "resplan")),
        footprint=footprint,
        rooms=rooms,
        storeys=2,
        source="resplan-bbox" if approximated else "resplan",
    )
    return plan


def load_resplan(path: str, limit: int | None = None) -> list[Plan]:
    """Load the ResPlan pickle. Unusable records are skipped and counted."""
    import pickle

    with open(path, "rb") as fh:
        data = pickle.load(fh)
    records = data if isinstance(data, list) else list(data.values())
    if limit:
        records = records[:limit]

    plans, skipped = [], 0
    for i, raw in enumerate(records):
        try:
            plans.append(from_resplan(raw, plan_id=f"RP{i:05d}"))
        except Exception:
            skipped += 1
    if skipped:
        print(f"load_resplan: skipped {skipped}/{len(records)} unusable records")
    return plans


# ------------------------------------------------------------ procedural

def _subdivide(rect, rng, min_side, depth=0):
    """Recursive binary subdivision -- the standard way to fake a layout."""
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    if depth >= 3 or (w < 2 * min_side and h < 2 * min_side):
        return [rect]
    vertical = w > h if abs(w - h) > 0.5 else rng.random() < 0.5
    if vertical and w >= 2 * min_side:
        cut = rng.uniform(x0 + min_side, x1 - min_side)
        return (_subdivide((x0, y0, cut, y1), rng, min_side, depth + 1)
                + _subdivide((cut, y0, x1, y1), rng, min_side, depth + 1))
    if not vertical and h >= 2 * min_side:
        cut = rng.uniform(y0 + min_side, y1 - min_side)
        return (_subdivide((x0, y0, x1, cut), rng, min_side, depth + 1)
                + _subdivide((x0, cut, x1, y1), rng, min_side, depth + 1))
    return [rect]


def procedural(plan_id: str, seed: int | None = None) -> Plan:
    """A plausible UK terraced or semi-detached ground floor.

    Over half are L-shaped, because a rear extension producing a non-convex
    footprint is the common case in householder applications and it is
    precisely where naive largest-contour measurement starts to fail. If
    every plan were a rectangle the geometry stage would look far better
    than it deserves.
    """
    rng = random.Random(seed)
    w = rng.uniform(5.0, 8.5)
    d = rng.uniform(7.5, 12.0)

    l_shaped = rng.random() < 0.55
    if l_shaped:
        ext_w = rng.uniform(0.45, 0.80) * w
        ext_d = rng.uniform(2.0, 4.5)
        footprint = [(0, 0), (w, 0), (w, d),
                     (ext_w, d), (ext_w, d + ext_d), (0, d + ext_d)]
        total_d = d + ext_d
    else:
        footprint = [(0, 0), (w, 0), (w, d), (0, d)]
        total_d = d

    cells = _subdivide((0, 0, w, total_d), rng, min_side=2.2)
    labels = rng.sample(ROOM_LABELS, min(len(cells), len(ROOM_LABELS)))

    rooms: list[Room] = []
    for (x0, y0, x1, y1), label in zip(cells, labels):
        if l_shaped and y1 > d:
            if x0 >= ext_w:
                continue                       # entirely in the missing corner
            x1 = min(x1, ext_w)
            if x1 - x0 < 1.5:
                continue
        rooms.append(Room(label, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))

    return Plan(
        plan_id=plan_id,
        footprint=footprint,
        rooms=rooms,
        storeys=rng.choice([1, 2, 2, 2, 3]),
        source="procedural",
    )


def procedural_corpus(n: int, seed: int = 0) -> list[Plan]:
    return [procedural(f"PROC{i:05d}", seed=seed * 100_003 + i) for i in range(n)]


if __name__ == "__main__":
    for p in procedural_corpus(6):
        shape = "L" if len(p.footprint) > 4 else "rect"
        print(f"{p.plan_id}  {p.ground_floor_area_m2:6.1f} m²  "
              f"{len(p.rooms)} rooms  {p.storeys} storeys  "
              f"{p.width:.1f}x{p.depth:.1f} m  {shape}")
