"""Run every sheet through all 4 pipeline stages and save the results.

    python src/run_extraction.py                 # all sheets
    python src/run_extraction.py --limit 20       # quick test, first 20 only

Every region is classified by triage() itself - we never peek at the
manifest's ground-truth labels, since a real bundle wouldn't have any.
A stage returning 0 confidence (nothing usable found) is a normal,
expected result, not an error.

Writes data/stage_outputs.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

from contracts import PropertyRecord, Stage
from stages import pipeline

ROOT = Path(__file__).resolve().parent.parent


def load_manifest(data_dir: Path) -> list[dict]:
    """Read manifest.jsonl into a list of dicts, one per sheet."""
    lines = (data_dir / "manifest.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def fake_candidate_pool(true_address: str, rng: random.Random, n_decoys: int = 4) -> list[str]:
    """Fake address candidates for link() to match against (no real UPRN/EPC data yet).

    Makes a few decoys with nearby house numbers, on the same street as
    the true address, to test the matching logic.
    """
    house_no, _, rest = true_address.partition(" ")
    decoys = []
    for offset in rng.sample([-3, -2, -1, 1, 2, 3], k=min(n_decoys, 6)):
        decoy_no = max(1, int(house_no) + offset)
        decoys.append(f"{decoy_no} {rest}")
    pool = decoys + [true_address]
    rng.shuffle(pool)
    return pool


def process_record(sheet: dict, data_dir: Path, rng: random.Random) -> tuple[PropertyRecord, str]:
    """Run one sheet through triage -> scale -> geometry -> link.

    Returns the filled-in PropertyRecord plus a short status string.
    """
    record = PropertyRecord(
        record_id=sheet["sheet_id"],
        application_ref=f"RENDERED/{sheet['sheet_id']}",  # not a real application ref
        source_pdf=str(data_dir / "sheets" / f"{sheet['sheet_id']}.png"),
        truth=sheet["truth"],
        truth_source="rendered",  # exact truth from the renderer, not epc/hand
    )

    sheet_img = Image.open(record.source_pdf).convert("RGB")

    # Classify every region on the sheet using only triage's own guess -
    # never the manifest's ground truth.
    by_class: dict[str, list[tuple]] = {}
    for region in sheet["regions"]:
        crop = sheet_img.crop(tuple(region["bbox"]))
        out = pipeline.triage(record, crop)
        by_class.setdefault(out.payload["page_class"], []).append((out, crop))

    floor_plan_candidates = by_class.get("floor_plan", [])
    if not floor_plan_candidates:
        return record, "abstained at triage (no region classified as floor_plan)"
    best_out, floor_plan_crop = max(floor_plan_candidates, key=lambda oc: oc[0].confidence)
    record.stage_outputs[Stage.TRIAGE] = best_out

    scale_bar_candidates = by_class.get("scale_bar", [])
    scale_bar_crop = (
        max(scale_bar_candidates, key=lambda oc: oc[0].confidence)[1]
        if scale_bar_candidates else None
    )
    elevation_crops = [crop for _, crop in by_class.get("elevation", [])]

    record.stage_outputs[Stage.SCALE] = pipeline.scale(record, scale_bar_crop)
    record.stage_outputs[Stage.GEOMETRY] = pipeline.geometry(
        record, floor_plan_crop, elevation_images=elevation_crops
    )

    # LINK needs an address and a candidate list, not an image.
    address = sheet["meta"]["address"]
    candidates = fake_candidate_pool(address, rng)
    record.stage_outputs[Stage.LINK] = pipeline.link(record, address, candidates)

    return record, "ok"


def main() -> None:
    """Process every sheet, print progress, save results to JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/stage_outputs.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_dir = ROOT / args.data
    sheets = load_manifest(data_dir)
    if args.limit:
        sheets = sheets[: args.limit]

    rng = random.Random(args.seed)
    records, status_counts = [], {}
    for sheet in sheets:
        record, status = process_record(sheet, data_dir, rng)
        records.append(record)
        status_counts[status] = status_counts.get(status, 0) + 1
        conf = record.confidence_vector()
        print(f"{sheet['sheet_id']}: {status}  "
              f"[triage={conf[0]:.2f} scale={conf[1]:.2f} "
              f"geometry={conf[2]:.2f} link={conf[3]:.2f}]")

    print("\nsummary:")
    for status, count in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {count:4d}  {status}")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([r.to_dict() for r in records], indent=2))
    print(f"\nwrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
