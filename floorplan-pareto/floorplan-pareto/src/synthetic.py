"""Fake pipeline output, used to test the optimizer before real data exists.

Not just random noise - it copies real-world problems on purpose:
  - confidence scores are noisy, not perfect
  - confidence is a bit inflated (overconfident), like real ML models
  - one bad stage drags down the stages after it
  - a few records can't be recovered no matter the threshold
"""

from __future__ import annotations

import numpy as np

from contracts import PropertyRecord, Stage, StageOutput

RNG_SEED = 20260906


def _confidence_from_correctness(
    rng: np.random.Generator,
    correct: np.ndarray,
    separation: float,
    overconfidence: float,
) -> np.ndarray:
    """Fake confidence scores: high for correct records, low for wrong ones.

    Not perfectly separated (some overlap), and pushed upward a bit to
    mimic a model that's more confident than it should be.
    """
    n = len(correct)
    a_ok, b_ok = 2.0 + 4.0 * separation, 2.0
    a_bad, b_bad = 2.0, 2.0 + 4.0 * separation
    conf = np.where(
        correct,
        rng.beta(a_ok, b_ok, n),
        rng.beta(a_bad, b_bad, n),
    )
    conf = conf ** (1.0 - overconfidence)  # exponent < 1 shifts mass upward
    return np.clip(conf, 0.001, 0.999)


def generate(n: int = 400, seed: int = RNG_SEED) -> list[PropertyRecord]:
    """Make n fake PropertyRecords, each with 4 stage outputs and a truth value."""
    rng = np.random.default_rng(seed)

    # Fake ground truth for each property.
    true_area = rng.lognormal(mean=np.log(62.0), sigma=0.33, size=n)
    true_storeys = rng.choice([1, 2, 3], size=n, p=[0.10, 0.75, 0.15])

    # Drawing quality (0-1). ~12% are too poor to ever recover.
    quality = rng.beta(5.0, 2.0, size=n)
    unrecoverable = quality < 0.30

    # Did each stage get it right? Each depends on the stage before it.
    triage_ok = rng.random(n) < np.clip(0.55 + 0.45 * quality, 0, 0.99)
    scale_ok = triage_ok & (rng.random(n) < np.clip(0.45 + 0.55 * quality, 0, 0.98))
    scale_ok &= ~unrecoverable
    geom_ok = scale_ok & (rng.random(n) < np.clip(0.62 + 0.38 * quality, 0, 0.97))
    link_ok = rng.random(n) < 0.86

    # Predicted area: close to true if geometry worked, way off if not.
    rel_err = np.where(
        geom_ok,
        rng.normal(0.0, 0.045, n),
        rng.choice([-0.5, 0.5, 1.0, -0.35], size=n) + rng.normal(0, 0.15, n),
    )
    pred_area = true_area * (1.0 + rel_err)

    storey_err = (~geom_ok) & (rng.random(n) < 0.45)
    pred_storeys = np.where(storey_err, np.maximum(1, true_storeys - 1), true_storeys)

    # Confidence per stage. Triage is easy (well separated); geometry is
    # hard (confidence and correctness overlap more).
    conf = {
        Stage.TRIAGE: _confidence_from_correctness(rng, triage_ok, 2.2, 0.25),
        Stage.SCALE: _confidence_from_correctness(rng, scale_ok, 1.6, 0.30),
        Stage.GEOMETRY: _confidence_from_correctness(rng, geom_ok, 1.1, 0.35),
        Stage.LINK: _confidence_from_correctness(rng, link_ok, 1.8, 0.15),
    }

    # 30 records are marked "hand"-labelled, the rest "epc".
    hand = set(rng.choice(n, size=30, replace=False).tolist())

    records: list[PropertyRecord] = []
    for i in range(n):
        rec = PropertyRecord(
            record_id=f"SYN{i:04d}",
            application_ref=f"25/{1000 + i}/FUL",
            source_pdf=f"data/raw/SYN{i:04d}.pdf",
            truth={
                "ground_floor_area_m2": float(true_area[i]),
                "storeys": int(true_storeys[i]),
            },
            truth_source="hand" if i in hand else "epc",
        )
        rec.stage_outputs[Stage.TRIAGE] = StageOutput(
            Stage.TRIAGE,
            float(conf[Stage.TRIAGE][i]),
            {"page_class": "floor_plan" if triage_ok[i] else "elevation"},
        )
        rec.stage_outputs[Stage.SCALE] = StageOutput(
            Stage.SCALE,
            float(conf[Stage.SCALE][i]),
            {"scale_denominator": 100 if scale_ok[i] else 50},
        )
        rec.stage_outputs[Stage.GEOMETRY] = StageOutput(
            Stage.GEOMETRY,
            float(conf[Stage.GEOMETRY][i]),
            {
                "ground_floor_area_m2": float(pred_area[i]),
                "storeys": int(pred_storeys[i]),
            },
        )
        rec.stage_outputs[Stage.LINK] = StageOutput(
            Stage.LINK,
            float(conf[Stage.LINK][i]),
            {"uprn": f"1000{i:07d}" if link_ok[i] else None},
        )
        records.append(rec)

    return records


if __name__ == "__main__":
    recs = generate()
    print(f"generated {len(recs)} synthetic records")
    for s in Stage.ordered():
        c = np.array([r.confidence(s) for r in recs])
        print(f"  {s.value:9s} mean conf {c.mean():.3f}  sd {c.std():.3f}")
