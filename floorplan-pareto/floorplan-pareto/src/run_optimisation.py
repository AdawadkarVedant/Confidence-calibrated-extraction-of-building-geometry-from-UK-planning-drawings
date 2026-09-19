"""Run the optimizer and save the results (Pareto front + plots + JSON).

    python src/run_optimisation.py --source synthetic
    python src/run_optimisation.py --source data/stage_outputs.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import nsga2
import synthetic
from contracts import PropertyRecord, Stage, StageOutput
from objectives import EvaluationMatrix, best_shared_threshold, grid_baseline

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
STAGE_LABELS = ["triage", "scale", "geometry", "link"]


def load_real(path: Path) -> list[PropertyRecord]:
    """Load real records from a JSON file (undoes PropertyRecord.to_dict())."""
    records = []
    for d in json.loads(path.read_text()):
        stage_outputs = {}
        for stage_value, out in d.get("stage_outputs", {}).items():
            stage = Stage(stage_value)
            stage_outputs[stage] = StageOutput(
                stage=stage,
                confidence=out["confidence"],
                payload=out.get("payload", {}),
                notes=out.get("notes", ""),
            )
        records.append(PropertyRecord(
            record_id=d["record_id"],
            application_ref=d.get("application_ref", ""),
            source_pdf=d.get("source_pdf", ""),
            stage_outputs=stage_outputs,
            truth=d.get("truth", {}),
            truth_source=d.get("truth_source", ""),
        ))
    return records


def load(source: str):
    """"synthetic" -> fake data. Anything else -> path to a real JSON file."""
    if source == "synthetic":
        return synthetic.generate()
    path = Path(source)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    return load_real(path)


def plot_front(F: np.ndarray, baseline: np.ndarray, chosen: int, path: Path) -> None:
    """Save a 3-panel scatter plot: front vs. baseline, 3 objectives pairwise."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    pairs = [
        (1, 0, "coverage", "error rate among accepted"),
        (2, 0, "review cost per record", "error rate among accepted"),
        (1, 2, "coverage", "review cost per record"),
    ]
    for ax, (xi, yi, xl, yl) in zip(axes, pairs):
        x = 1 - F[:, xi] if xi == 1 else F[:, xi]
        y = 1 - F[:, yi] if yi == 1 else F[:, yi]
        bx = 1 - baseline[:, xi] if xi == 1 else baseline[:, xi]
        by = 1 - baseline[:, yi] if yi == 1 else baseline[:, yi]

        ax.scatter(x, y, s=26, c=F[:, 2], cmap="viridis", label="Pareto front")
        ax.plot(bx, by, "--", color="0.45", lw=1.4,
                label="single shared threshold")
        ax.scatter(x[chosen], y[chosen], s=190, facecolors="none",
                   edgecolors="crimson", lw=2.2, label="chosen operating point")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.25)

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Joint threshold tuning across a four-stage extraction pipeline "
        "(NSGA-II, colour = review cost)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_thresholds(X: np.ndarray, F: np.ndarray, chosen: int, path: Path) -> None:
    """Save a line plot: the 4 thresholds for every point on the front."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cov = 1 - F[:, 1]
    norm = plt.Normalize(cov.min(), cov.max())
    cmap = plt.get_cmap("plasma")
    for i, row in enumerate(X):
        ax.plot(range(4), row, color=cmap(norm(cov[i])), alpha=0.35, lw=1.0)
    ax.plot(range(4), X[chosen], color="crimson", lw=3.0, marker="o",
            label="chosen operating point")
    ax.set_xticks(range(4), STAGE_LABELS)
    ax.set_ylabel("confidence threshold")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="coverage")
    ax.set_title("Pareto-optimal threshold vectors by stage", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Load data, run the optimizer, print results, save plots + JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic")
    ap.add_argument("--pop", type=int, default=120)
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cost-ratio", type=float, default=20.0,
                    help="cost of a confidently wrong answer / cost of a referral")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    records = load(args.source)
    matrix = EvaluationMatrix.from_records(records)
    print(f"{matrix.n_records} records, "
          f"{matrix.correct.mean():.1%} correct before any filtering")

    res = nsga2.minimise(
        matrix.evaluate_population,
        n_var=4,
        pop_size=args.pop,
        generations=args.gens,
        seed=args.seed,
        verbose=True,
    )
    print(f"\nfront size: {len(res.F)}")

    baseline = grid_baseline(matrix)
    ref = np.array([1.0, 1.0])
    hv_front = nsga2.hypervolume_2d(res.F[:, :2], ref)
    hv_base = nsga2.hypervolume_2d(baseline[:, :2], ref)
    print(f"hypervolume (error x 1-coverage): front {hv_front:.4f} "
          f"vs shared-threshold baseline {hv_base:.4f} "
          f"({hv_front / max(hv_base, 1e-9):.2f}x)")

    chosen, cost = matrix.select_operating_point(
        res.X, cost_wrong=args.cost_ratio, cost_review=1.0
    )
    t = res.X[chosen]
    objs = res.F[chosen]
    print(f"\noperating point at cost ratio {args.cost_ratio:g}:1")
    print("  thresholds  " + "  ".join(
        f"{lab}={v:.3f}" for lab, v in zip(STAGE_LABELS, t)))
    print(f"  error rate among accepted  {objs[0]:.3f}")
    print(f"  coverage                   {1 - objs[1]:.3f}")
    print(f"  review cost per record     {objs[2]:.3f}")
    print(f"  expected cost per record   {cost:.3f}")

    sweep = matrix.sweep_cost_ratio(res.X, [2, 5, 10, 20, 50, 100])
    for row in sweep:
        t_sh, c_sh = best_shared_threshold(matrix, row["cost_ratio"])
        row["best_shared_threshold"] = round(t_sh, 3)
        row["shared_expected_cost"] = round(c_sh, 4)
        row["saving_vs_shared"] = round((c_sh - row["expected_cost"]) / c_sh, 4)

    print("\ncost ratio sweep (saving is against the best possible shared threshold)")
    print(f"{'ratio':>6} {'error':>7} {'cover':>7} {'review':>7} "
          f"{'E[cost]':>8} {'shared':>8} {'saving':>7}  thresholds")
    for row in sweep:
        th = " ".join(f"{v:.2f}" for v in row["thresholds"])
        print(f"{row['cost_ratio']:>6g} {row['error_rate']:>7.3f} "
              f"{row['coverage']:>7.3f} {row['review_cost']:>7.3f} "
              f"{row['expected_cost']:>8.3f} {row['shared_expected_cost']:>8.3f} "
              f"{row['saving_vs_shared']:>6.1%}  {th}")

    plot_front(res.F, baseline, chosen, OUT / "pareto_front.png")
    plot_thresholds(res.X, res.F, chosen, OUT / "thresholds.png")

    (OUT / "results.json").write_text(json.dumps({
        "n_records": matrix.n_records,
        "front": [
            {"thresholds": x.round(4).tolist(),
             "error_rate": round(float(f[0]), 4),
             "coverage": round(float(1 - f[1]), 4),
             "review_cost": round(float(f[2]), 4)}
            for x, f in zip(res.X, res.F)
        ],
        "baseline_shared_threshold": baseline.round(4).tolist(),
        "hypervolume": {"front": hv_front, "baseline": hv_base},
        "cost_ratio_sweep": sweep,
        "chosen": {"index": chosen, "expected_cost": cost},
    }, indent=2))
    print(f"\nwrote {OUT/'pareto_front.png'}, {OUT/'thresholds.png'}, "
          f"{OUT/'results.json'}")


if __name__ == "__main__":
    main()
