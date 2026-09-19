"""Controlled degradation: the experiment scraped data cannot support.

Rendering gives you a knob real planning portals do not: input quality.
Render the same plans at rising degradation and you can show *how the
Pareto front moves as document quality falls*.

That is the question an insurer actually has. Not "how accurate is it" but
"how does it degrade across a national corpus where a large share of the
drawings are flattened raster with no text layer". You cannot run that
experiment on 60 scraped bundles at any sample size, because you do not
control the nuisance variable -- you can only observe whatever mix the
portal happens to contain.

Levels are cumulative and modelled on the real Exeter sheet:

  0  clean render, room labels present, scale bar present
  1  print-driver flattening: mild JPEG, slight resampling
  2  scanner: rotation, contrast loss, noise, blur
  3  photocopy: heavy JPEG, downsample, speckle, room labels dropped,
     scale bar sometimes clipped off the sheet

Level 1 is the *common* case, not the degraded one. The real bundle came
out of Vectorworks through a PDF writer that destroyed the text layer and
left 35 raster strips. Treating flattening as exotic would flatter the
pipeline.
"""

from __future__ import annotations

import io
import random
from dataclasses import asdict, dataclass

from PIL import Image, ImageFilter


@dataclass
class Degradation:
    """Render-time flags plus image-time nuisance parameters."""

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
    """Build a profile, jittered so the corpus is not four discrete clusters."""
    rng = rng or random.Random()
    params = dict(LEVELS[level])
    if level >= 2:
        params["rotation_deg"] = rng.uniform(-1, 1) * params["rotation_deg"]
    if level == 3:
        # The scale bar sometimes falls off a badly cropped photocopy. This
        # is the case that forces the pipeline to fall back to the title
        # block ratio, or to abstain.
        params["include_scale_bar"] = rng.random() > 0.35
    return Degradation(level=level, **params)


def apply(img: Image.Image, deg: Degradation,
          rng: random.Random | None = None) -> Image.Image:
    """Apply the image-time part of a profile. Render-time flags are
    consumed by SheetRenderer.render, not here."""
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
