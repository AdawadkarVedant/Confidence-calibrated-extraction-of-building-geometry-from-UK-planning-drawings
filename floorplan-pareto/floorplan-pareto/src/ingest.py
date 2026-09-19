"""Copy downloaded planning-application PDFs into data/, and log metadata.

Layout:
    data/
      raw/25-1223-FUL/documents/*.pdf   <- gitignored, never published
      derived/                          <- gitignored, regenerable
      redacted/                         <- only image folder safe to publish
      corpus.jsonl                      <- committed: metadata only, no personal data

Notes:
  - One folder per application (not flat filenames), because bundles vary
    a lot in how many files they have.
  - Original filenames are kept - they're a free label (e.g. "elevations.pdf").
  - corpus.jsonl exists because the address/dates aren't inside the PDFs
    themselves.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# Guess a document's type from its filename. Order matters: "composite"
# must be checked first, or it gets wrongly matched as floor_plan/elevation.
LABEL_PATTERNS: list[tuple[str, str]] = [
    ("composite", r"plans?\s+and\s+elevations?"),
    ("site_plan", r"site\s+(location|layout)|location\s+plan|block\s+plan"),
    ("floor_plan", r"floor\s+plans?|ground\s+floor|first\s+floor|roof\s+plan"),
    ("elevation", r"elevations?"),
    ("section", r"sections?"),
    ("form", r"application\s+form|certificate|ownership"),
    ("admin", r"decision|officers?\s+report|consultation|comment|correspondence"),
]

# These document types contain personal data, so they're never copied in.
EXCLUDE_LABELS = {"form", "admin"}


def slug(reference: str) -> str:
    """25/1223/FUL -> 25-1223-FUL"""
    return re.sub(r"[^A-Za-z0-9]+", "-", reference).strip("-")


def weak_label(filename: str) -> str:
    """Guess the document type from its filename."""
    name = filename.lower()
    for label, pattern in LABEL_PATTERNS:
        if re.search(pattern, name):
            return label
    return "unknown"


def revision(filename: str) -> str | None:
    """Pull the revision suffix from '2608-01-A Existing site layout plan'."""
    m = re.search(r"\b(\d{3,4}-\d{1,2})-([A-Z])\b", filename)
    return m.group(2) if m else None


def drawing_number(filename: str) -> str | None:
    m = re.search(r"\b(\d{3,4}-\d{1,2})\b", filename)
    return m.group(1) if m else None


def requires_ocr(pdf: Path) -> bool | None:
    """True if the PDF is a scanned image with no text layer (needs OCR).

    Returns None if the pdffonts tool isn't installed.
    """
    try:
        out = subprocess.run(
            ["pdffonts", str(pdf)], capture_output=True, text=True, timeout=30
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    body = out.stdout.splitlines()[2:]
    return not any(line.strip() for line in body)


def sha256(path: Path) -> str:
    """Short hash of a file's contents, for detecting duplicates."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


@dataclass
class Document:
    filename: str                 # original, untouched
    weak_label: str
    drawing_number: str | None
    revision: str | None
    superseded: bool
    requires_ocr: bool | None
    sha256: str


@dataclass
class Bundle:
    reference: str                # 25/1223/FUL
    slug: str
    address: str = ""             # published as part of the application
    postcode: str = ""
    received: str = ""            # ISO date
    decided: str = ""
    uprn: str | None = None       # filled by the linkage stage
    epc_lodged: str | None = None # to check temporal validity of the label
    documents: list[Document] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def mark_superseded(docs: list[Document]) -> None:
    """Mark old revisions of the same drawing as superseded.

    Without this, near-duplicate drawings could end up on both sides of
    a train/test split, making accuracy look better than it really is.
    """
    latest: dict[str, str] = {}
    for d in docs:
        if d.drawing_number is None:
            continue
        rev = d.revision or ""
        if rev >= latest.get(d.drawing_number, ""):
            latest[d.drawing_number] = rev
    for d in docs:
        if d.drawing_number is not None:
            d.superseded = (d.revision or "") != latest[d.drawing_number]


def ingest(
    reference: str,
    source_dir: Path,
    address: str = "",
    postcode: str = "",
    received: str = "",
    decided: str = "",
    exclude_personal: bool = True,
) -> Bundle:
    """Copy one downloaded bundle of PDFs into data/, and log it."""
    s = slug(reference)
    dest = DATA / "raw" / s / "documents"
    dest.mkdir(parents=True, exist_ok=True)

    bundle = Bundle(
        reference=reference, slug=s, address=address,
        postcode=postcode, received=received, decided=decided,
    )

    for pdf in sorted(source_dir.glob("*.pdf")):
        label = weak_label(pdf.name)
        if exclude_personal and label in EXCLUDE_LABELS:
            print(f"  skipped ({label}): {pdf.name}")
            continue
        target = dest / pdf.name
        if not target.exists():
            shutil.copy2(pdf, target)
        bundle.documents.append(
            Document(
                filename=pdf.name,
                weak_label=label,
                drawing_number=drawing_number(pdf.name),
                revision=revision(pdf.name),
                superseded=False,
                requires_ocr=requires_ocr(target),
                sha256=sha256(target),
            )
        )

    mark_superseded(bundle.documents)

    manifest = DATA / "corpus.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a") as fh:
        fh.write(bundle.to_json() + "\n")

    return bundle


def load_corpus() -> list[Bundle]:
    """Read corpus.jsonl back into a list of Bundles."""
    manifest = DATA / "corpus.jsonl"
    if not manifest.exists():
        return []
    out = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        docs = [Document(**x) for x in d.pop("documents", [])]
        out.append(Bundle(**d, documents=docs))
    return out


def training_documents() -> list[tuple[str, str, str]]:
    """(slug, filename, label) rows for training, skipping unclear/old ones."""
    rows = []
    for b in load_corpus():
        for d in b.documents:
            if d.superseded or d.weak_label in {"unknown", "composite"}:
                continue
            rows.append((b.slug, d.filename, d.weak_label))
    return rows


if __name__ == "__main__":
    names = [
        "2608-23 Proposed first floor and roof plans.pdf",
        "2608-24 Proposed elevations.pdf",
        "Site Location Plan.pdf",
        "2608-02 Existing floor plans.pdf",
        "2608-03 Existing elevations.pdf",
        "2608-01 Existing site layout plan.pdf",
        "2608-01-A Existing site layout plan.pdf",
        "011 02a proposed plan and elevations.pdf",
        "Application Form.pdf",
    ]
    docs = [
        Document(n, weak_label(n), drawing_number(n), revision(n), False, None, "")
        for n in names
    ]
    mark_superseded(docs)
    for d in docs:
        flag = "  SUPERSEDED" if d.superseded else ""
        drop = "  [excluded]" if d.weak_label in EXCLUDE_LABELS else ""
        print(f"{d.weak_label:11s} rev={str(d.revision):4s} {d.filename}{flag}{drop}")
