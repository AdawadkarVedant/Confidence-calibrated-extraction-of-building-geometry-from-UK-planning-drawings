"""Train a classifier to recognise sheet regions (floor plan, elevation, etc).

    python src/stages/train_triage.py --data data --epochs 12

Trains on cropped regions from the rendered corpus, not whole real PDF
pages - a real page usually only has one type of drawing on it, but our
rendered sheets combine several regions into one image.

Writes outputs/triage_model.pt (weights + temperature + class list) and
outputs/triage_reliability.png.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]

# Same list as render/sheet.py's REGION_CLASSES.
CLASSES = ["floor_plan", "elevation", "section", "site_plan",
           "title_block", "scale_bar", "north_arrow", "notes"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_manifest(data_dir: Path) -> list[dict]:
    """Read manifest.jsonl into a list of dicts, one per sheet."""
    lines = (data_dir / "manifest.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def region_samples(sheets: list[dict], data_dir: Path) -> list[dict]:
    """Turn each sheet's regions into training samples (image + box + label)."""
    samples = []
    for sheet in sheets:
        img_path = data_dir / "sheets" / f"{sheet['sheet_id']}.png"
        for region in sheet["regions"]:
            samples.append({
                "sheet_id": sheet["sheet_id"],
                "image_path": img_path,
                "bbox": region["bbox"],
                "label": CLASSES.index(region["cls"]),
            })
    return samples


def split_by_sheet(sheets: list[dict], val_frac: float, seed: int) -> tuple[set[str], set[str]]:
    """Split whole sheets into train/val (not individual regions).

    Splitting by region instead would leak info: regions from the same
    sheet look similar, so validation accuracy would look better than it
    really is.
    """
    ids = [s["sheet_id"] for s in sheets]
    random.Random(seed).shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    return set(ids[n_val:]), set(ids[:n_val])


class RegionDataset(Dataset):
    """Loads one region crop per sample, resized and normalised for ResNet."""

    def __init__(self, samples: list[dict], train: bool):
        self.samples = samples
        aug = [transforms.RandomHorizontalFlip()] if train else []
        self.transform = transforms.Compose([
            *aug,
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self._cache: dict[Path, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _sheet(self, path: Path) -> Image.Image:
        # Cache so we don't re-open the same sheet image for every region on it.
        if path not in self._cache:
            self._cache[path] = Image.open(path).convert("RGB")
        return self._cache[path]

    def __getitem__(self, i: int):
        s = self.samples[i]
        crop = self._sheet(s["image_path"]).crop(tuple(s["bbox"]))
        return self.transform(crop), s["label"]


def build_model(num_classes: int) -> nn.Module:
    """A ResNet-18 pretrained on ImageNet, with a new final layer for our classes."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, device, optimizer=None) -> tuple[float, float]:
    """Run one epoch. Give an optimizer to train, or omit it to just measure accuracy."""
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Get raw model outputs for every sample in loader (needed to fit temperature)."""
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device)).cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, iters: int = 200) -> float:
    """Find one number (temperature) that fixes overconfident predictions.

    Freshly trained models are usually too confident. Dividing their
    scores by this number spreads the confidence back out to match real
    accuracy - without changing which answer is picked.
    """
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=iters)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def reliability_diagram(logits: torch.Tensor, labels: torch.Tensor, temperature: float,
                         path: Path, n_bins: int = 10) -> None:
    """Plot claimed confidence vs. real accuracy, save it as a PNG."""
    probs = F.softmax(logits / temperature, dim=1)
    conf, pred = probs.max(1)
    correct = (pred == labels).float()

    edges = torch.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_count = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            bin_acc.append(correct[mask].mean().item())
            bin_conf.append(conf[mask].mean().item())
            bin_count.append(int(mask.sum()))
        else:
            bin_acc.append(0.0)
            bin_conf.append(((lo + hi) / 2).item())
            bin_count.append(0)

    ece = sum(c * abs(a - cf) for a, cf, c in zip(bin_acc, bin_conf, bin_count)) / len(labels)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="0.6", label="perfect calibration")
    ax.bar(edges[:-1], bin_acc, width=1 / n_bins, align="edge",
           edgecolor="black", alpha=0.75, label="triage stage")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Triage reliability diagram  (ECE={ece:.3f}, T={temperature:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Load data, train the model, calibrate it, save everything."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_dir = ROOT / args.data
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    sheets = load_manifest(data_dir)
    train_ids, val_ids = split_by_sheet(sheets, args.val_frac, args.seed)
    samples = region_samples(sheets, data_dir)
    train_samples = [s for s in samples if s["sheet_id"] in train_ids]
    val_samples = [s for s in samples if s["sheet_id"] in val_ids]
    print(f"{len(sheets)} sheets -> {len(train_samples)} train / "
          f"{len(val_samples)} val region crops")

    train_loader = DataLoader(RegionDataset(train_samples, train=True),
                               batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(RegionDataset(val_samples, train=False),
                             batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(CLASSES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device)
        print(f"epoch {epoch:2d}  train loss {train_loss:.3f} acc {train_acc:.3f}  "
              f"val loss {val_loss:.3f} acc {val_acc:.3f}")

    val_logits, val_labels = collect_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    reliability_diagram(val_logits, val_labels, temperature,
                         out_dir / "triage_reliability.png")

    ckpt_path = out_dir / "triage_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "temperature": temperature,
        "classes": CLASSES,
    }, ckpt_path)
    print(f"\nwrote {ckpt_path} and {out_dir / 'triage_reliability.png'}")


if __name__ == "__main__":
    main()
