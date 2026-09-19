# Workflow

The exact sequence of commands to go from nothing to a finished Pareto
front, what each one produces, and what to check along the way. For what
each function does, see [TECHNICAL.md](TECHNICAL.md); for how the pieces
fit together structurally, see [ARCHITECTURE.md](ARCHITECTURE.md).

There are two ways to run this project: the **quick path** (fake
confidence numbers, seconds to run, no images) and the **real path**
(actual rendered images, a real trained model, several minutes to run).
Both produce the same shape of result. Start with whichever one you need;
they don't depend on each other.

---

## Quick path: test the optimiser with fake data

No corpus, no training, no images. Useful for testing changes to the
optimisation layer itself.

```bash
pip install -r requirements.txt
python src/run_optimisation.py --source synthetic
```

What happens: `synthetic.generate()` fabricates 400 fake `PropertyRecord`s
directly from probability distributions (see
[TECHNICAL.md](TECHNICAL.md#srcsyntheticpy)), then NSGA-II runs against
them. Takes a few seconds. Writes `outputs/results.json`,
`outputs/pareto_front.png`, `outputs/thresholds.png`.

This path never touches an image and never calls anything under
`src/stages/` or `src/render/`.

---

## Real path: build the corpus, train the model, extract, optimise

### Step 1 — build the rendered corpus

```bash
python src/render/build_corpus.py --n 100 --seed 11 --out data
```

- `--n 100` — how many fake sheets to generate. (The project has been run
  at both 100 and 300; the number just trades corpus size for build/train
  time — nothing downstream cares which one you pick.)
- `--seed 11` — controls plan generation, degradation-level assignment,
  and per-sheet render randomness.
- `--out data` — writes directly into the `data/` folder the rest of the
  project expects (the script's own default, `data/rendered`, is *not*
  what other scripts look for — always pass `--out data` explicitly).

**What gets written:**
```
data/sheets/SHEET00000.png ... SHEET00099.png
data/labels/SHEET00000.txt ... (YOLO format)
data/truth/SHEET00000.json ... (exact ground truth per sheet)
data/manifest.jsonl          (all of the above, one line per sheet)
```

**What to check:** the printed degradation mix (e.g.
`L0=15, L1=44, L2=31, L3=10`) should roughly match the `--mix` weights
(default `0.20 0.40 0.28 0.12`). If you want to sanity-check the render
itself, open one PNG under `data/sheets/` directly.

### Step 2 — train the triage model

```bash
python src/stages/train_triage.py --data data --epochs 12
```

- `--epochs 12` is the full default; `--epochs 3` is enough for a smoke
  test — this corpus's 8 region classes are visually distinct enough
  that validation accuracy typically hits 100% within the first couple
  of epochs either way.

**What happens:** loads `manifest.jsonl`, splits sheets (not individual
regions) into train/validation, fine-tunes a ResNet-18 on cropped
regions, fits a temperature-scaling correction on the validation set,
and saves everything needed to use the model later.

**What gets written:**
```
outputs/triage_model.pt          (weights + temperature + class list)
outputs/triage_reliability.png   (calibration check plot)
```

**What to check:** the printed per-epoch `train acc` / `val acc` should
climb toward 1.0. Open `triage_reliability.png` — the bars should sit
close to the diagonal "perfect calibration" line; the title reports ECE
(lower is better) and the fitted temperature.

**This step must run again** any time you regenerate the corpus —
`outputs/triage_model.pt` is trained against one specific set of images
and won't match a freshly rendered corpus even if `--n` is the same
(different random content).

### Step 3 — run extraction

```bash
python src/run_extraction.py
```

Optional flags:
- `--limit 20` — process only the first 20 sheets (fast sanity check
  before committing to a full run).
- `--out data/stage_outputs.json` — where to write results (this is
  already the default).

**What happens:** for every sheet, every region gets classified by the
trained model, then `scale() → geometry() → link()` run in order, each
reading whatever the previous stage produced. Prints one line per sheet:

```
SHEET00000: ok  [triage=1.00 scale=0.95 geometry=0.85 link=0.03]
```

`ok` means the record made it through the full chain (though any
individual stage may still have reported zero confidence — that's a
normal, valid outcome, not a failure). Records that never found a
`floor_plan` region abstain immediately and are reported as such.

**What gets written:**
```
data/stage_outputs.json   (every record, as far as it got, in
                           PropertyRecord.to_dict() shape)
```

**What to check:** the final summary line count (e.g. `100 ok`). If you
want to validate accuracy against ground truth before moving on, compare
`stage_outputs.json`'s `geometry.payload.room_count` /
`ground_floor_area_m2` against the matching file in `data/truth/` — a
past validation run found ~88% exact room-count matches and ~3% mean
area error, with the worst cases concentrated on the most heavily
degraded sheets.

### Step 4 — run the optimiser on real data

```bash
python src/run_optimisation.py --source data/stage_outputs.json
```

Same script as the quick path, different `--source`. Optional flags:
`--pop` (population size, default 120), `--gens` (generations, default
150), `--cost-ratio` (default 20.0 — how much worse a confidently wrong
answer is than a manual referral).

**What happens:** loads the real records via `load_real()`, runs NSGA-II,
compares the resulting front against the best possible single shared
threshold, sweeps several cost ratios, and reports the chosen operating
point.

**What gets written (overwriting the quick-path files if you ran that
first):**
```
outputs/results.json
outputs/pareto_front.png
outputs/thresholds.png
```

**What to check:** the printed `N.N% correct before any filtering` line
is your baseline extraction accuracy. `hypervolume ... front X vs
shared-threshold baseline Y (Zx)` — `Z > 1` means joint tuning beats a
single shared threshold. The cost-ratio sweep table's `saving` column
should be positive across the board; a run at `n=100` produced savings
of roughly 5–14% depending on cost ratio.

---

## The two things that are easy to mix up

**`synthetic.py` vs. `render/build_corpus.py`.** Both produce "fake
data," but they are completely independent and never feed into each
other:

| | `synthetic.py` | `render/build_corpus.py` |
|---|---|---|
| Produces | Fake confidence numbers | Fake images + exact geometry |
| Consumed by | `run_optimisation.py --source synthetic` | `train_triage.py`, `run_extraction.py` |
| Touches an image? | Never | Always |

**Re-running `build_corpus.py` invalidates the current checkpoint.** If
you regenerate the corpus (even with the same `--n` and `--seed`, if
`plans.py` or `sheet.py` changed), the images are different pixel data,
and `outputs/triage_model.pt` was trained on the old ones. Always retrain
(step 2) after rebuilding the corpus (step 1), before running extraction
(step 3).

---

## Not part of the runnable workflow yet

`src/ingest.py` organises real downloaded planning-application PDFs, but
no real data has been collected for this project — every workflow above
runs entirely on the rendered corpus. If real data is ever collected, the
intended flow is: `ingest.py` organises the PDFs → the same four
`pipeline.py` stages run against real page images instead of rendered
crops → the same `run_extraction.py` → `run_optimisation.py` chain
applies unchanged. Getting there also needs a real region-detection step
upstream of `triage()` (real pages don't come with a manifest telling you
where the floor plan is), and OCR wired into `scale()`'s title-block
route — both currently open, stated limitations, not silent gaps.
