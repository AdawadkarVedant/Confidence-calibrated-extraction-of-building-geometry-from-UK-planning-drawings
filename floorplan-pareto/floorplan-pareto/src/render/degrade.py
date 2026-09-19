"""Make a rendered sheet image worse, to simulate real-world scan quality.

4 levels, each worse than the last:
  0  clean - nothing changed
  1  mild - slight JPEG compression and resizing (the common case)
  2  scanner - also rotated, blurry, noisy, lower contrast
  3  photocopy - all of the above but heavier, room labels removed,
     scale bar sometimes missing entirely
"""

from __future__ import annotations

import io
import random
from dataclasses import asdict, dataclass

from PIL import Image, ImageFilter


@dataclass
class Degradation:
    """Settings for one degradation level."""

    level: int
    include_room_labels: bool = True
    include_scale_bar: bool = True
    jpeg_quality: int | None = None
    rotation_deg: float = 0.0
    blur_radius: float = 0.0
    noise_sigma: float = 0.0
    contrast: float = 1.0
    downsample: float = 1.0          # 1.0 = full resolution

    def to_dict(self) -> dict:
        return asdict(self)


LEVELS: dict[int, dict] = {
    0: dict(),
    1: dict(jpeg_quality=88, downsample=0.92),
    2: dict(jpeg_quality=70, rotation_deg=0.8, blur_radius=0.6,
            noise_sigma=5.0, contrast=0.88, downsample=0.80),
    3: dict(jpeg_quality=42, rotation_deg=1.8, blur_radius=1.2,
            noise_sigma=11.0, contrast=0.72, downsample=0.60,
            include_room_labels=False),
}


def profile(level: int, rng: random.Random | None = None) -> Degradation:
    """Build one degradation setting for a given level, with some randomness."""
    rng = rng or random.Random()
    params = dict(LEVELS[level])
    if level >= 2:
        params["rotation_deg"] = rng.uniform(-1, 1) * params["rotation_deg"]
    if level == 3:
        # At the worst level, the scale bar sometimes goes missing entirely.
        params["include_scale_bar"] = rng.random() > 0.35
    return Degradation(level=level, **params)


def apply(img: Image.Image, deg: Degradation,
          rng: random.Random | None = None) -> Image.Image:
    """Apply rotation/blur/noise/jpeg to an image, based on a Degradation."""
    rng = rng or random.Random()
    out = img

    if deg.rotation_deg:
        out = out.rotate(deg.rotation_deg, resample=Image.BICUBIC,
                         fillcolor=(255, 255, 255), expand=False)

    if deg.downsample < 1.0:
        w, h = out.size
        small = (max(1, int(w * deg.downsample)), max(1, int(h * deg.downsample)))
        out = out.resize(small, Image.LANCZOS).resize((w, h), Image.BILINEAR)

    if deg.blur_radius:
        out = out.filter(ImageFilter.GaussianBlur(deg.blur_radius))

    if deg.contrast != 1.0:
        from PIL import ImageEnhance
        out = ImageEnhance.Contrast(out).enhance(deg.contrast)

    if deg.noise_sigma:
        import numpy as np
        a = np.asarray(out, dtype=np.float32)
        a += np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, deg.noise_sigma, a.shape)
        out = Image.fromarray(np.clip(a, 0, 255).astype("uint8"))

    if deg.jpeg_quality:
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=deg.jpeg_quality)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")

    return out
