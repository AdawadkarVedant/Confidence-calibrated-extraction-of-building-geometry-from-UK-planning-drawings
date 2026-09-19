# Technical reference

Every function, class, and constant in `src/`, organised by file. For the
big picture, see [ARCHITECTURE.md](ARCHITECTURE.md); for how to actually
run things in order, see [WORKFLOW.md](WORKFLOW.md).

Each entry: what it does, its parameters, and what it returns.

---

## `src/contracts.py`

Shared data types every other module imports. No logic beyond validation.

### `class Stage(str, Enum)`
The four pipeline stages, in the order they run.
- Members: `TRIAGE = "triage"`, `SCALE = "scale"`, `GEOMETRY = "geometry"`, `LINK = "link"`
- **`Stage.ordered() -> list[Stage]`** — classmethod. Returns all four in execution order. No parameters.

### `class StageOutput` (dataclass)
One stage's result for one record.
- **Fields:** `stage: Stage`, `confidence: float` (must be 0–1), `payload: dict = {}`, `notes: str = ""`
- **`__post_init__()`** — raises `ValueError` if `confidence` is outside `[0, 1]`. Runs automatically on construction.

### `class PropertyRecord` (dataclass)
One planning application as it moves through the pipeline.
- **Fields:** `record_id: str`, `application_ref: str = ""`, `source_pdf: str = ""`, `stage_outputs: dict[Stage, StageOutput] = {}`, `truth: dict = {}`, `truth_source: str = ""`
- **`confidence(stage: Stage) -> float`** — confidence for one stage. Returns `0.0` if that stage never ran (i.e., abstained).
- **`confidence_vector() -> list[float]`** — no parameters. Returns all 4 confidences in `Stage.ordered()` order, e.g. `[triage_conf, scale_conf, geometry_conf, link_conf]`.
- **`prediction(key: str, default=None) -> Any`** — searches every populated stage's `payload` dict for `key` (e.g. `"scale_denominator"`), returns the first match or `default`.
- **`to_dict() -> dict`** — no parameters. Converts the whole record (including nested `StageOutput`s) into a plain JSON-serialisable dict. This is what `run_extraction.py` writes to disk and what `run_optimisation.py`'s `load_real()` reads back.

### Module-level constants
- **`REVIEW_COST: dict[Stage, float]`** — `{TRIAGE: 1.0, SCALE: 2.0, GEOMETRY: 5.0, LINK: 3.0}`. Relative reviewer effort to check an abstention at each stage.
- **`MANUAL_SURVEY_COST = 1.0`** — cost of one full manual survey (an abstained record).
- **`REVIEW_EFFORT_PRICE = 0.05`** — price of one `REVIEW_COST` unit, same currency as `MANUAL_SURVEY_COST`.

---

## `src/nsga2.py`

NSGA-II multi-objective optimizer, implemented from scratch (no external library).

### `fast_non_dominated_sort(F: np.ndarray) -> list[np.ndarray]`
Splits a batch of objective-value rows into ranked fronts (front 0 = nothing beats it, front 1 = next best, etc).
- **`F`** — shape `(n_records, n_objectives)`, lower is better in every column.
- **Returns** — list of arrays of row-indices, one array per front.

### `crowding_distance(F: np.ndarray) -> np.ndarray`
Density estimate for every point within a single front (higher = more isolated = kept preferentially when trimming).
- **`F`** — objective values for one front only.
- **Returns** — array of one distance value per row; edge points get `inf`.

### `sbx_crossover(p1, p2, rng, eta=15.0, prob=0.9) -> tuple[np.ndarray, np.ndarray]`
Blends two parent vectors into two children (simulated binary crossover).
- **`p1`, `p2`** — parent decision vectors (same shape).
- **`rng`** — `np.random.Generator`.
- **`eta`** — crossover distribution index; higher = children closer to parents.
- **`prob`** — probability crossover happens at all; otherwise returns copies of the parents unchanged.
- **Returns** — `(child1, child2)`.

### `polynomial_mutation(x, rng, lower, upper, eta=20.0, prob=None) -> np.ndarray`
Randomly nudges some entries of `x` within `[lower, upper]`.
- **`x`** — one decision vector.
- **`rng`** — `np.random.Generator`.
- **`lower`, `upper`** — per-variable bounds, same shape as `x`.
- **`eta`** — mutation distribution index; higher = smaller nudges.
- **`prob`** — per-gene mutation probability; defaults to `1/len(x)` if `None`.
- **Returns** — mutated copy of `x`, clipped to bounds.

### `tournament(rank, crowd, rng) -> int`
Picks 2 random individuals, returns the index of the better one (lower rank wins; ties broken by higher crowding distance).
- **`rank`** — array of front-rank per individual.
- **`crowd`** — array of crowding distance per individual.
- **`rng`** — `np.random.Generator`.
- **Returns** — index of the winner.

### `class Result` (dataclass)
Output of `minimise()`.
- **Fields:** `X: np.ndarray` (threshold vectors on the final front), `F: np.ndarray` (their objective values), `history: list[dict]` (per-generation stats).

### `minimise(evaluate, n_var, lower=0.0, upper=1.0, pop_size=100, generations=120, seed=0, verbose=False) -> Result`
Runs the full NSGA-II loop.
- **`evaluate`** — a function mapping a `(pop_size, n_var)` array to a `(pop_size, n_objectives)` array (in this project, `EvaluationMatrix.evaluate_population`).
- **`n_var`** — number of decision variables (4, one per stage).
- **`lower`, `upper`** — scalar or per-variable bounds for the search space.
- **`pop_size`** — population size per generation.
- **`generations`** — how many generations to run.
- **`seed`** — RNG seed for reproducibility.
- **`verbose`** — if `True`, prints progress every 20 generations.
- **Returns** — a `Result` holding the final non-dominated front.

### `hypervolume_2d(F: np.ndarray, ref: np.ndarray) -> float`
Exact hypervolume (area dominated) of a 2-objective front against a reference point.
- **`F`** — shape `(n, 2)`, the two objectives to measure.
- **`ref`** — reference point, e.g. `[1.0, 1.0]`.
- **Returns** — a single float; bigger means a better (more dominant) front.

---

## `src/objectives.py`

Turns a batch of `PropertyRecord`s plus a 4-value threshold vector into 3 scores NSGA-II can minimise.

### Module constant
- **`AREA_TOLERANCE = 0.10`** — a record counts as wrong if predicted area is off by more than 10%, or storeys don't match.

### `class EvaluationMatrix` (dataclass)
All records pre-converted to numpy arrays for fast repeated scoring.
- **Fields:** `confidences: np.ndarray (n_records, 4)`, `correct: np.ndarray (n_records,) bool`, `review_cost: np.ndarray (4,)`, `record_ids: list[str]`.
- **`n_records`** — property, returns `confidences.shape[0]`.
- **`from_records(records: list[PropertyRecord]) -> EvaluationMatrix`** — classmethod. Builds the matrix: pulls `confidence_vector()` from every record, checks each against truth (`ground_floor_area_m2` within `AREA_TOLERANCE`, `storeys` exact match) to fill `correct`.
- **`route(thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]`** — decides accept/abstain for every record given a 4-value threshold vector. Returns `(accepted, abstain_stage)`; `abstain_stage` is the index of the first stage that failed, or `-1` if accepted.
- **`evaluate(thresholds: np.ndarray) -> np.ndarray`** — returns `[error_rate, 1 - coverage, review_cost]` for one threshold vector. Returns a fixed penalty `[1.0, 1.0, review_cost.max()]` if nothing is accepted (avoids the "abstain on everything" degenerate optimum).
- **`evaluate_population(pop: np.ndarray) -> np.ndarray`** — calls `evaluate` once per row of `pop`; this is what gets passed to `nsga2.minimise` as its `evaluate` argument.
- **`expected_cost(thresholds, cost_wrong=20.0, cost_review=1.0) -> float`** — average per-record cost in units of one manual survey: `0` for accepted+correct, `cost_wrong` for accepted+wrong, `1 survey + review effort` for abstained.
- **`select_operating_point(front, cost_wrong=20.0, cost_review=1.0) -> tuple[int, float]`** — evaluates `expected_cost` for every point on `front`, returns `(best_index, best_cost)`.
- **`sweep_cost_ratio(front, ratios: list[float]) -> list[dict]`** — calls `select_operating_point` at each ratio in `ratios`, returns one result row (thresholds, error, coverage, review cost, expected cost) per ratio.

### `best_shared_threshold(matrix, cost_wrong, resolution=200) -> tuple[float, float]`
Brute-forces the single best value if only one shared threshold (applied to all 4 stages equally) were allowed.
- **`matrix`** — an `EvaluationMatrix`.
- **`cost_wrong`** — cost of an accepted-but-wrong record.
- **`resolution`** — how many candidate values to try, evenly spaced 0–0.99.
- **Returns** — `(best_threshold, its_expected_cost)`.

### `grid_baseline(matrix, levels=40) -> np.ndarray`
Same idea as above but returns the whole curve, for plotting the baseline comparison line.
- **`matrix`** — an `EvaluationMatrix`.
- **`levels`** — how many threshold values to evaluate, 0–0.95.
- **Returns** — array of `[error_rate, 1-coverage, review_cost]` rows, degenerate (coverage-0) rows dropped.

---

## `src/synthetic.py`

Fabricates fake `PropertyRecord`s directly from probability distributions — no images involved. Used only when `run_optimisation.py --source synthetic` is passed.

### Module constant
- **`RNG_SEED = 20260906`**

### `_confidence_from_correctness(rng, correct, separation, overconfidence) -> np.ndarray`
Draws one array of confidence scores: high for records where `correct` is `True`, low where `False`, with realistic overlap.
- **`rng`** — `np.random.default_rng` instance.
- **`correct`** — boolean array, one per record.
- **`separation`** — how far apart the "correct" and "incorrect" distributions are (higher = cleaner separation).
- **`overconfidence`** — 0–1, how much to push the whole distribution upward (mimics real overconfident models).
- **Returns** — array of confidences, one per record, clipped to `[0.001, 0.999]`.

### `generate(n=400, seed=RNG_SEED) -> list[PropertyRecord]`
Builds `n` fake records end to end: true area/storeys, a hidden "quality" score, whether each stage secretly succeeds (chained — a scale failure drags geometry down), predicted values, and confidence per stage.
- **`n`** — how many fake records to generate. Default 400 — unrelated to the size of the rendered image corpus.
- **`seed`** — RNG seed.
- **Returns** — list of `n` fully-populated `PropertyRecord`s, all 4 stages present on every record (never abstains — this is fabricated data, not real routing).

### `if __name__ == "__main__":`
Generates 400 records and prints the mean/stddev confidence per stage, for a quick sanity check.

---

## `src/run_optimisation.py`

Loads records (real or fake), runs the optimizer, saves plots and results.

### Constants
- **`ROOT`** — repo root (`Path(__file__).resolve().parent.parent`).
- **`OUT`** — `ROOT / "outputs"`.
- **`STAGE_LABELS`** — `["triage", "scale", "geometry", "link"]`, used for axis labels.

### `load_real(path: Path) -> list[PropertyRecord]`
Reads a JSON file (in the shape `PropertyRecord.to_dict()` produces) and reconstructs full `PropertyRecord` objects, including rebuilding each `StageOutput` from its dict form.
- **`path`** — path to the JSON file (e.g. `data/stage_outputs.json`).
- **Returns** — list of `PropertyRecord`.

### `load(source: str) -> list[PropertyRecord]`
Single entry point for getting data into the pipeline.
- **`source`** — `"synthetic"` to call `synthetic.generate()`, or any file path to load real data via `load_real()`. Relative paths resolve against `ROOT`. Raises `SystemExit` if the file doesn't exist.
- **Returns** — list of `PropertyRecord`.

### `plot_front(F, baseline, chosen, path) -> None`
Saves a 3-panel scatter plot: coverage vs. error, review cost vs. error, coverage vs. review cost — front, baseline, and the chosen point all overlaid.
- **`F`** — the optimizer's front objective values.
- **`baseline`** — `grid_baseline()`'s output, for the dashed comparison line.
- **`chosen`** — index into `F` of the selected operating point (drawn as a red circle).
- **`path`** — where to save the PNG.

### `plot_thresholds(X, F, chosen, path) -> None`
Saves a parallel-coordinates line plot: the 4 threshold values for every point on the front, coloured by coverage.
- **`X`** — the optimizer's front decision vectors (thresholds).
- **`F`** — the corresponding objective values (used for the colour scale).
- **`chosen`** — index of the point to highlight in bold red.
- **`path`** — where to save the PNG.

### `main() -> None`
Parses CLI args (`--source`, `--pop`, `--gens`, `--seed`, `--cost-ratio`), loads data, runs `nsga2.minimise`, computes the baseline and hypervolume, picks an operating point, runs the cost-ratio sweep, saves both plots, and writes `outputs/results.json`. No parameters (reads `sys.argv` via `argparse`); no return value.

---

## `src/stages/pipeline.py`

The four real extraction stages.

### Constants
- **`ROOT`** — repo root (two levels up from this file).
- **`TRIAGE_CHECKPOINT`** — `ROOT / "outputs" / "triage_model.pt"`.
- **`IMAGENET_MEAN`, `IMAGENET_STD`** — normalisation constants matching `train_triage.py`.

### `_load_triage_model() -> dict`
Loads the trained triage checkpoint once and caches it in a module-level variable (`_triage_model_cache`) so repeated calls don't re-read from disk.
- No parameters.
- **Returns** — `{"model": nn.Module, "temperature": float, "classes": list[str]}`.
- Raises `FileNotFoundError` if the checkpoint doesn't exist yet.

### `triage(record: PropertyRecord, page_image) -> StageOutput`
Classifies one image crop and returns a calibrated confidence.
- **`record`** — the `PropertyRecord` (currently unused inside the function body, kept for signature consistency with the other 3 stages).
- **`page_image`** — a `PIL.Image` already in memory, or a file path.
- **Returns** — `StageOutput(Stage.TRIAGE, confidence, {"page_class": <predicted class name>})`.

### `_run_lengths(row: np.ndarray) -> list[int]`
Counts how long each consecutive run of same-value pixels is along one row of a binary (True/False) array.
- **`row`** — 1-D boolean array.
- **Returns** — list of run lengths, in order.

### `scale(record: PropertyRecord, scale_bar_image) -> StageOutput`
Recovers the drawing's scale denominator from a scale-bar image.
- **`record`** — unused directly (kept for signature consistency).
- **`scale_bar_image`** — a `PIL.Image` of the scale-bar region, or `None` if triage never found one.
- **Method:** scores every row by how many tick-width (≥8px) runs it produces after filtering, keeps the best-scoring row, takes the median run width as pixels-per-metre, converts to a scale denominator assuming 150 dpi.
- **Returns** — `StageOutput(Stage.SCALE, confidence, {"scale_denominator": int})`, or confidence `0.0` with an empty payload if no image was given or no tick pattern was found.

### `_count_storeys(elevation_images: list) -> tuple[int | None, float]`
Counts storeys by detecting rows of window/door-shaped contours across one or more elevation images.
- **`elevation_images`** — list of `PIL.Image` crops.
- **Returns** — `(most_common_storey_count, agreement_fraction)`, or `(None, 0.0)` if nothing was detected in any image.

### `geometry(record, floor_plan_image, elevation_images=None) -> StageOutput`
Measures ground-floor area, room count, and (via `_count_storeys`) storey count.
- **`record`** — used to read the scale via `record.prediction("scale_denominator")`.
- **`floor_plan_image`** — `PIL.Image` of the floor-plan region.
- **`elevation_images`** — optional list of elevation crops, passed through to `_count_storeys`.
- **Method:** thresholds the image, finds contours with `cv2.RETR_CCOMP` (splits outer boundaries from the holes/rooms inside them), takes the largest top-level contour as the building outline, counts its direct children as rooms.
- **Returns** — `StageOutput(Stage.GEOMETRY, confidence, {"ground_floor_area_m2": float, "room_count": int, "storeys": int (if found)})`. Confidence `0.0` if no scale was available or no contour was found.

### `link(record, address: str, candidates: list[str]) -> StageOutput`
Matches an address string to the best of a list of candidates.
- **`record`** — unused directly.
- **`address`** — the extracted address string to match.
- **`candidates`** — list of candidate address strings to score against.
- **Method:** normalises both sides, scores every candidate with `difflib.SequenceMatcher`, confidence = margin between the best and second-best score.
- **Returns** — `StageOutput(Stage.LINK, confidence, {"matched_address": str, "match_score": float})`, or confidence `0.0` if `candidates` is empty.

---

## `src/stages/train_triage.py`

Trains the ResNet-18 region classifier used by `pipeline.triage()`.

### Constants
- **`ROOT`**, **`CLASSES`** (8 region class names, same order as `render/sheet.py`'s `REGION_CLASSES`), **`IMAGENET_MEAN`**, **`IMAGENET_STD`**.

### `load_manifest(data_dir: Path) -> list[dict]`
Reads `manifest.jsonl` into a list of dicts, one per sheet.
- **`data_dir`** — path to the `data/` folder.
- **Returns** — list of sheet dicts.

### `region_samples(sheets: list[dict], data_dir: Path) -> list[dict]`
Expands every sheet's region list into individual training samples.
- **`sheets`** — output of `load_manifest`.
- **`data_dir`** — path to `data/`.
- **Returns** — list of `{"sheet_id", "image_path", "bbox", "label"}` dicts, one per region.

### `split_by_sheet(sheets, val_frac, seed) -> tuple[set[str], set[str]]`
Splits whole sheets (not individual regions) into train/validation sets.
- **`sheets`** — list of sheet dicts.
- **`val_frac`** — fraction of sheets held out for validation (e.g. `0.2`).
- **`seed`** — RNG seed for the shuffle.
- **Returns** — `(train_sheet_ids, val_sheet_ids)`, both sets of strings.

### `class RegionDataset(Dataset)`
PyTorch dataset that crops one region per sample and applies the ResNet preprocessing transform.
- **`__init__(samples: list[dict], train: bool)`** — `train=True` adds `RandomHorizontalFlip`.
- **`__len__() -> int`** — number of samples.
- **`_sheet(path: Path) -> Image`** — opens (and caches) a sheet image.
- **`__getitem__(i: int) -> tuple[Tensor, int]`** — crops sample `i`'s region, applies the transform, returns `(image_tensor, label_index)`.

### `build_model(num_classes: int) -> nn.Module`
Builds a ResNet-18 pretrained on ImageNet, with a new final linear layer.
- **`num_classes`** — number of output classes (8).
- **Returns** — the model, ready for training.

### `run_epoch(model, loader, device, optimizer=None) -> tuple[float, float]`
Runs one pass over `loader`.
- **`model`**, **`loader`**, **`device`** — standard PyTorch objects.
- **`optimizer`** — if given, trains (backprop + step); if `None`, only evaluates.
- **Returns** — `(average_loss, accuracy)` for the epoch.

### `collect_logits(model, loader, device) -> tuple[Tensor, Tensor]`
Runs every sample in `loader` through `model` and collects the raw (pre-softmax) outputs — needed for temperature fitting.
- **Returns** — `(all_logits, all_labels)`, concatenated across the whole loader.

### `fit_temperature(logits, labels, iters=200) -> float`
Finds the single scalar temperature `T` that minimises validation cross-entropy when logits are divided by it before softmax.
- **`logits`, `labels`** — validation-set outputs from `collect_logits`.
- **`iters`** — max LBFGS iterations.
- **Returns** — the fitted temperature (float).

### `reliability_diagram(logits, labels, temperature, path, n_bins=10) -> None`
Bins predictions by confidence, plots claimed confidence vs. real accuracy per bin, saves as PNG.
- **`logits`, `labels`** — validation outputs.
- **`temperature`** — the fitted `T`, applied before plotting.
- **`path`** — output PNG path.
- **`n_bins`** — number of confidence buckets.
- No return value; also computes and displays ECE (Expected Calibration Error) in the plot title.

### `main() -> None`
Parses CLI args (`--data`, `--epochs`, `--batch-size`, `--lr`, `--val-frac`, `--seed`), loads and splits the corpus, trains for the given number of epochs, calibrates, saves the reliability diagram and the checkpoint (`outputs/triage_model.pt`, containing `state_dict`, `temperature`, and `classes`).

---

## `src/run_extraction.py`

Driver script: runs every sheet through all 4 pipeline stages.

### Constant
- **`ROOT`** — repo root.

### `load_manifest(data_dir: Path) -> list[dict]`
Same as `train_triage.py`'s version (duplicated intentionally — the two scripts are independent entry points).

### `fake_candidate_pool(true_address: str, rng: random.Random, n_decoys=4) -> list[str]`
Builds a fake address candidate list for `link()` to match against, since no real UPRN/EPC data has been downloaded.
- **`true_address`** — the correct address, always included in the pool.
- **`rng`** — `random.Random` instance.
- **`n_decoys`** — how many decoy addresses (nearby house numbers, same street) to add.
- **Returns** — shuffled list of address strings.

### `process_record(sheet: dict, data_dir: Path, rng: random.Random) -> tuple[PropertyRecord, str]`
Runs one sheet through `triage → scale → geometry → link`.
- **`sheet`** — one sheet's dict from the manifest.
- **`data_dir`** — path to `data/`.
- **`rng`** — shared random generator (used for the fake candidate pool).
- **Method:** classifies every region via `pipeline.triage()` using only triage's own predictions (never the manifest's ground-truth class); picks the highest-confidence `floor_plan` and `scale_bar` matches; runs `scale()`, `geometry()`, `link()` in order.
- **Returns** — `(record, status_string)`. Status is `"ok"` if it completed, or an "abstained at triage" message if no region was classified as `floor_plan`.

### `main() -> None`
Parses CLI args (`--data`, `--out`, `--limit`, `--seed`), loads the manifest, processes every sheet (or the first `--limit`), prints per-sheet status and a summary, writes `data/stage_outputs.json`.

---

## `src/ingest.py`

Organises real downloaded PDF bundles (not currently exercised against any real data).

### Constants
- **`DATA`** — path to `data/`.
- **`LABEL_PATTERNS`** — ordered list of `(label, regex)` pairs used to guess a document's type from its filename.
- **`EXCLUDE_LABELS`** — `{"form", "admin"}`, document types never copied in (personal-data boundary).

### `slug(reference: str) -> str`
Converts `"25/1223/FUL"` to `"25-1223-FUL"` (filesystem-safe).

### `weak_label(filename: str) -> str`
Guesses a document's type from its filename using `LABEL_PATTERNS`. Returns `"unknown"` if nothing matches.

### `revision(filename: str) -> str | None`
Extracts a revision letter (e.g. `"A"`) from filenames like `"2608-01-A Existing site layout plan.pdf"`.

### `drawing_number(filename: str) -> str | None`
Extracts a drawing number (e.g. `"2608-01"`) from a filename.

### `requires_ocr(pdf: Path) -> bool | None`
Runs the `pdffonts` command-line tool to check if a PDF has an embedded text layer.
- **`pdf`** — path to a PDF file.
- **Returns** — `True` if no fonts found (needs OCR), `False` if fonts found, `None` if `pdffonts` isn't installed.

### `sha256(path: Path) -> str`
Returns a 16-character hash of a file's contents (for duplicate detection).

### `class Document` (dataclass)
Fields: `filename`, `weak_label`, `drawing_number`, `revision`, `superseded`, `requires_ocr`, `sha256`.

### `class Bundle` (dataclass)
Fields: `reference`, `slug`, `address`, `postcode`, `received`, `decided`, `uprn`, `epc_lodged`, `documents: list[Document]`.
- **`to_json() -> str`** — serialises the bundle (and its documents) to a JSON string.

### `mark_superseded(docs: list[Document]) -> None`
Marks all but the latest revision of each drawing number as `superseded = True`. Mutates `docs` in place; no return value.

### `ingest(reference, source_dir, address="", postcode="", received="", decided="", exclude_personal=True) -> Bundle`
Copies one downloaded bundle of PDFs into `data/raw/`, applies `weak_label` to each, skips personal-data document types, and appends a line to `data/corpus.jsonl`.
- **`reference`** — the planning application reference.
- **`source_dir`** — where the downloaded PDFs currently sit.
- **`address`, `postcode`, `received`, `decided`** — metadata not stored inside the PDFs themselves.
- **`exclude_personal`** — if `True` (default), skips `form`/`admin` documents.
- **Returns** — the constructed `Bundle`.

### `load_corpus() -> list[Bundle]`
Reads `data/corpus.jsonl` back into a list of `Bundle` objects. No parameters.

### `training_documents() -> list[tuple[str, str, str]]`
Returns `(slug, filename, label)` rows suitable for training, skipping superseded/unknown/composite documents. No parameters.

### `if __name__ == "__main__":`
Demonstrates `weak_label`, `drawing_number`, `revision`, and `mark_superseded` against a hardcoded list of example filenames.

---

## `src/render/plans.py`

Invents building layouts as plain data (no image yet).

### Types and constants
- **`Point = tuple[float, float]`**, **`Polygon = list[Point]`**
- **`ROOM_LABELS`** — the 10 possible room names.

### `polygon_area(poly: Polygon) -> float`
Area of a polygon via the shoelace formula. Works regardless of winding direction.

### `bounds(poly: Polygon) -> tuple[float, float, float, float]`
Returns `(min_x, min_y, max_x, max_y)`.

### `class Room` (dataclass)
Fields: `label: str`, `polygon: Polygon`.
- **`area`** — property, `polygon_area(self.polygon)`.

### `class Plan` (dataclass)
Fields: `plan_id`, `footprint: Polygon`, `rooms: list[Room] = []`, `storeys: int = 2`, `source: str = "procedural"`.
- **`ground_floor_area_m2`** — property.
- **`width`**, **`depth`** — properties, from `bounds(self.footprint)`.
- **`truth() -> dict`** — returns `{"ground_floor_area_m2", "storeys", "room_count", "width_m", "depth_m"}`, the exact correct answer for this plan.

### ResPlan adapter (not currently used by the corpus)
- **`RESPLAN_KEYS`** — dict of possible field-name variants per ResPlan release.
- **`_first(d, names, default=None)`** — returns the first key in `names` present in `d`.
- **`from_resplan(raw: dict, plan_id=None) -> Plan`** — converts one raw ResPlan record into a `Plan`. Falls back to the bounding box of all rooms if no outline is given (flagged via `source="resplan-bbox"`).
- **`load_resplan(path: str, limit=None) -> list[Plan]`** — loads a ResPlan pickle file, skipping (and counting) unparseable records.

### Procedural generator (what the corpus actually uses)
- **`_subdivide(rect, rng, min_side, depth=0) -> list[tuple]`** — recursively splits a rectangle into smaller rectangles. Stops at `depth >= 3` or once both dimensions are under `2 * min_side`.
- **`procedural(plan_id: str, seed=None) -> Plan`** — builds one fake layout: random width/depth, 55% chance of an L-shape (rear extension), subdivided into rooms. L-shapes are built from 2 separate rectangle subdivisions (main block + extension) so rooms always exactly tile the footprint with no gaps.
- **`procedural_corpus(n: int, seed=0) -> list[Plan]`** — calls `procedural` `n` times with derived seeds (`seed * 100_003 + i`).

### `if __name__ == "__main__":`
Generates 6 plans and prints a one-line summary of each.

---

## `src/render/sheet.py`

Draws a `Plan` as a full A3 sheet image.

### Constants
- **`A3_MM = (297.0, 420.0)`**, **`DEFAULT_DPI = 150`**
- **`INK`**, **`PAPER`** — RGB colours.
- **`REGION_CLASSES`** — the 8 possible region class names (`floor_plan`, `elevation`, `section`, `site_plan`, `title_block`, `scale_bar`, `north_arrow`, `notes`).
- **`_FONT_NAMES`** — candidate font filenames tried in order (DejaVu, Arial, Liberation), so rendering works cross-platform.

### `_font(size: int, bold=False) -> ImageFont`
Loads a font, trying each name in `_FONT_NAMES` until one works; falls back to PIL's built-in default. Cached by `(size, bold)`.

### `class Region` (dataclass)
Fields: `cls: str`, `bbox: tuple[int,int,int,int]`, `scale_denominator: int | None = None`, `note: str = ""`.
- **`yolo(w: int, h: int) -> str`** — this region's box in YOLO label format (`class cx cy width height`, all 0–1 normalised).

### `class SheetRenderer`
- **`__init__(dpi=DEFAULT_DPI, seed=None)`** — computes canvas pixel size from `A3_MM` and `dpi`.
- **`_px_per_metre(scale_denominator: int) -> float`** — how many pixels equal 1 metre at this scale and dpi.
- **`_transform(reference, box, ppm) -> callable`** — builds one metres→pixels conversion function, shared by every shape in a drawing (so rooms don't render nested inside each other).
- **`_extent(*point_lists) -> tuple[int,int,int,int]`** — static method. Tight bounding box around actual drawn points (not the allocated layout slot).
- **`_floor_plan(d, plan, box, scale, with_labels=True) -> tuple`** — draws rooms, walls, labels (skipped if a label wouldn't fit), and width/depth dimension numbers. Returns the region's tight bbox.
- **`_elevation(d, plan, box, scale, title) -> tuple`** — draws a simple facade: body, roof (75% chance), windows per storey, a ground-floor door. Returns its bbox.
- **`_site_plan(d, plan, box, scale) -> tuple`** — draws the subject property plus a few random-sized neighbours, at a different scale than the rest of the sheet.
- **`_scale_bar(d, box, scale) -> None`** — draws a 0–10m alternating tick-mark bar.
- **`_north_arrow(d, box) -> None`** — draws a circle with an arrow.
- **`_title_block(d, box, plan, scale, meta) -> None`** — draws practice name, address, drawing title, scale, date (all fake).
- **`render(plan, meta, main_scale=100, site_scale=1250, include_site_plan=True, include_scale_bar=True, include_room_labels=True) -> tuple[Image, list[Region]]`** — the main method: draws every piece onto one blank page in a fixed layout, returns the finished image and the list of every region drawn.

### Constants and functions at module end
- **`PRACTICES`**, **`STREETS`** — lists of fake names used to build title blocks.
- **`synthetic_meta(rng, plan) -> dict`** — builds fake title-block metadata (practice, address, drawing title, drawing number, date).
- **`region_summary(regions: list[Region]) -> str`** — one-line comma-separated summary, for debug printing.

### `if __name__ == "__main__":`
Renders one demo sheet and prints its size, region summary, and truth values.

---

## `src/render/degrade.py`

Applies simulated image-quality degradation to a rendered sheet.

### `class Degradation` (dataclass)
Fields: `level: int`, `include_room_labels: bool = True`, `include_scale_bar: bool = True`, `jpeg_quality: int | None = None`, `rotation_deg: float = 0.0`, `blur_radius: float = 0.0`, `noise_sigma: float = 0.0`, `contrast: float = 1.0`, `downsample: float = 1.0`.
- **`to_dict() -> dict`** — `dataclasses.asdict(self)`.

### `LEVELS: dict[int, dict]`
Fixed parameter presets for levels 0 (nothing), 1 (mild JPEG + resize), 2 (+ rotation, blur, noise, contrast), 3 (heavier versions of all of the above, room labels off).

### `profile(level: int, rng=None) -> Degradation`
Builds one `Degradation` instance for a level, with randomised rotation direction (level ≥ 2) and a chance the scale bar goes missing entirely (level 3).
- **`level`** — 0–3.
- **`rng`** — `random.Random` instance.
- **Returns** — a `Degradation`.

### `apply(img: Image, deg: Degradation, rng=None) -> Image`
Applies the actual pixel-level effects: rotation, downsample-then-upsample, Gaussian blur, contrast adjustment, Gaussian noise, JPEG re-compression — in that order, only for whichever fields in `deg` are non-default.
- **Returns** — the degraded image.

---

## `src/render/build_corpus.py`

Top-level script that builds the whole rendered corpus.

### `build(n, out, resplan, seed, mix, dpi, debug) -> None`
Generates `n` plans (procedural, or from a ResPlan pickle if `resplan` is given), renders and degrades each one, and writes `sheets/`, `truth/`, `labels/`, and `manifest.jsonl` under `out`.
- **`n`** — number of sheets to generate.
- **`out`** — output directory (e.g. `data/`).
- **`resplan`** — optional path to a ResPlan pickle; `None` uses `plans.procedural_corpus`.
- **`seed`** — top-level RNG seed (controls degradation-level assignment, scale choices, per-sheet renderer seeds — independent of each plan's own internal seed).
- **`mix`** — 4 weights for how sheets are distributed across degradation levels 0–3.
- **`dpi`** — render resolution.
- **`debug`** — if `True`, also saves bounding-box overlay images for the first 8 sheets.
- No return value; prints progress and a final summary.

### `main() -> None`
Parses CLI args (`--n`, `--out`, `--resplan`, `--seed`, `--dpi`, `--mix`, `--debug`) and calls `build()`.
