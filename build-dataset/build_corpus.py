"""Build a rendered sheet corpus with exact ground truth.

    python src/render/build_corpus.py --n 300 --out data/rendered
    python src/render/build_corpus.py --n 300 --resplan /path/to/ResPlan.pkl
    python src/render/build_corpus.py --n 8 --debug          # bbox overlays

Emits per sheet:

    sheets/SHEET00001.png        the rendered, degraded drawing
    truth/SHEET00001.json        true area in m², storeys, every region's
                                 class, tight pixel bbox and scale
    labels/SHEET00001.txt        the same regions in YOLO format
    manifest.jsonl               one line per sheet

The region boxes are free detection and classification labels.

Degradation level is assigned per sheet from `--mix`, so a single corpus
supports stratified evaluation: report the Pareto front separately at each
level and you have the degradation experiment rather than a single number.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import degrade
import plans as plans_mod
from sheet import SheetRenderer, synthetic_meta

ROOT = Path(__file__).resolve().parents[2]


def build(n: int, out: Path, resplan: str | None, seed: int,
          mix: list[float], dpi: int, debug: bool) -> None:
    rng = random.Random(seed)

    if resplan:
        source = plans_mod.load_resplan(resplan, limit=n)
        if len(source) < n:
            print(f"only {len(source)} usable ResPlan records; topping up "
                  f"with procedural plans")
            source += plans_mod.procedural_corpus(n - len(source), seed=seed)
    else:
        source = plans_mod.procedural_corpus(n, seed=seed)

    for sub in ("sheets", "truth", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if debug:
        (out / "debug").mkdir(parents=True, exist_ok=True)

    levels = [0, 1, 2, 3]
    manifest = (out / "manifest.jsonl").open("w")
    counts = {lv: 0 for lv in levels}

    for i, plan in enumerate(source[:n]):
        sheet_id = f"SHEET{i:05d}"
        level = rng.choices(levels, weights=mix)[0]
        deg = degrade.profile(level, rng)
        counts[level] += 1

        renderer = SheetRenderer(dpi=dpi, seed=rng.randrange(1 << 30))
        meta = synthetic_meta(rng, plan)

        # Site plan scale varies, so the mixed-scale trap is not a constant
        # the model can memorise.
        site_scale = rng.choice([1250, 1250, 2500, 500])

        img, regions = renderer.render(
            plan, meta,
            main_scale=rng.choice([100, 100, 100, 50]),
            site_scale=site_scale,
            include_scale_bar=deg.include_scale_bar,
            include_room_labels=deg.include_room_labels,
        )
        main_scale = next(r.scale_denominator for r in regions
                          if r.cls == "floor_plan")

        if debug and i < 8:
            from PIL import ImageDraw
            import sheet as sheet_mod
            dbg = img.copy()
            dd = ImageDraw.Draw(dbg)
            for g in regions:
                dd.rectangle(g.bbox, outline=(220, 30, 30), width=4)
                dd.text((g.bbox[0] + 6, g.bbox[1] + 6), g.cls,
                        fill=(220, 30, 30), font=sheet_mod._font(22, True))
            dbg.save(out / "debug" / f"{sheet_id}_boxes.png")

        img = degrade.apply(img, deg, rng)
        img.save(out / "sheets" / f"{sheet_id}.png")

        w, h = img.size
        (out / "labels" / f"{sheet_id}.txt").write_text(
            "\n".join(g.yolo(w, h) for g in regions) + "\n")

        record = {
            "sheet_id": sheet_id,
            "plan_id": plan.plan_id,
            "plan_source": plan.source,
            "dpi": dpi,
            "size_px": [w, h],
            "sheet_scale_denominator": main_scale,
            "degradation": deg.to_dict(),
            "truth": plan.truth(),
            "regions": [asdict(g) for g in regions],
            "meta": meta,
        }
        (out / "truth" / f"{sheet_id}.json").write_text(json.dumps(record, indent=2))
        manifest.write(json.dumps(record) + "\n")

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}")

    manifest.close()
    print(f"\nwrote {n} sheets to {out}")
    print("degradation mix: " + ", ".join(
        f"L{lv}={counts[lv]}" for lv in levels))
    print(f"scale traps: every sheet carries a site plan at a different "
          f"scale from the main drawings")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "Confidence calibrated extraction of building geometry from UK planning drawings" / "floorplan-pareto" / "floorplan-pareto" / "data")
    ap.add_argument("--resplan", default=None,
                    help="path to the ResPlan pickle; procedural plans if omitted")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--mix", type=float, nargs=4, default=[0.20, 0.40, 0.28, 0.12],
                    help="proportion of sheets at degradation levels 0..3")
    ap.add_argument("--debug", action="store_true",
                    help="also write bbox overlays for the first 8 sheets")
    args = ap.parse_args()
    build(args.n, args.out, args.resplan, args.seed, args.mix, args.dpi, args.debug)


if __name__ == "__main__":
    main()
