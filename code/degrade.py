"""The controlled degradation ladder.

Deterministic given (image, spec), so degraded images are generated on the fly at
feature-extraction time rather than written to disk — exactly reproducible, and it
avoids storing ~1.45 GB per rung.

Spec strings are "family:severity", e.g. blur:3, jpeg:30, temp:0.4, crop:0.6.
"clean" is the identity. LADDER holds the rungs we actually report.

Note on `noise`: it is not a realistic phone artifact. It is the control that makes
the adversarial argument concrete — it *raises* Laplacian variance while destroying
information, so a sharpness-based quality gate scores it as a better photo.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import uniform_filter1d

# The rungs reported in the paper. Severity 0 of every family is the clean image.
LADDER: dict[str, list[float]] = {
    "blur":      [0.0, 1.0, 2.0, 3.0, 4.0, 6.0],      # Gaussian sigma, px
    "motion":    [0.0, 3.0, 5.0, 9.0, 13.0, 17.0],    # kernel length, px
    "jpeg":      [95.0, 75.0, 50.0, 30.0, 20.0, 10.0],  # libjpeg quality
    "downscale": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0],      # shrink factor, then back up
    "temp":      [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],      # colour-temperature shift
    "dark":      [1.0, 0.8, 0.6, 0.45, 0.3, 0.2],     # brightness multiplier
    "crop":      [1.0, 0.85, 0.7, 0.6, 0.5, 0.4],     # kept fraction of frame
    "noise":     [0.0, 0.02, 0.04, 0.08, 0.12, 0.16],  # Gaussian sigma, [0,1] scale
}

# "Screenshot" = the compound rung: downscale, re-compress, crop. It is the one
# that matches the user-submitted-screenshot story, and the one where attribution
# is impossible by construction — which is why the single-factor rungs exist.
SCREENSHOT = [("downscale", 2.0), ("jpeg", 40.0), ("crop", 0.8)]

# "Phone" = an imitation of a hand-held phone photo of the same plate: mild hand-shake
# blur, camera JPEG at quality 75, a warm white-balance shift, and looser framing that
# loses part of the plate rim. Nutrition5k has no phone photos, so this rung is a
# stated imitation, not a measurement of phones. Each factor is mild on its own; the
# point is the compound.
PHONE = [("blur", 1.0), ("jpeg", 75.0), ("temp", 0.1), ("crop", 0.85)]


def parse_spec(spec: str) -> list[tuple[str, float]]:
    """'clean' -> []; 'blur:3' -> [('blur', 3.0)]; 'blur:3+jpeg:40' -> both, in order."""
    spec = spec.strip()
    if not spec or spec == "clean":
        return []
    if spec == "screenshot":
        return list(SCREENSHOT)
    if spec == "phone":
        return list(PHONE)
    out = []
    for part in spec.split("+"):
        family, _, sev = part.partition(":")
        family = family.strip()
        if family not in LADDER:
            raise ValueError(f"unknown degradation family {family!r}; known: {sorted(LADDER)}")
        out.append((family, float(sev) if sev else LADDER[family][1]))
    return out


def _jpeg(img: Image.Image, quality: float) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(round(quality)))
    buf.seek(0)
    out = Image.open(buf)
    out.load()  # decode now so the buffer can be released
    return out


def _motion_blur(img: Image.Image, length: float) -> Image.Image:
    """Horizontal camera shake: a 1-D box average along the x axis.

    Done with scipy rather than ImageFilter.Kernel, which only accepts 3x3 and 5x5.
    """
    k = max(int(round(length)), 1)
    if k <= 1:
        return img
    a = np.asarray(img, dtype=np.float32)
    a = uniform_filter1d(a, size=k, axis=1, mode="nearest")
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _downscale(img: Image.Image, factor: float) -> Image.Image:
    if factor <= 1:
        return img
    w, h = img.size
    small = img.resize((max(int(w / factor), 1), max(int(h / factor), 1)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def _temp_shift(img: Image.Image, amount: float) -> Image.Image:
    """Warm the image: scale R up and B down. amount in [0, 1]."""
    if amount == 0:
        return img
    a = np.asarray(img, dtype=np.float32)
    a[..., 0] *= 1.0 + amount
    a[..., 2] *= 1.0 - amount
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _crop(img: Image.Image, keep: float) -> Image.Image:
    """Centre-crop to `keep` of each dimension, then resize back to the original size.

    Cropping the frame removes the plate rim and table edge — the only scale
    references in a fixed-camera overhead shot — so this is the rung we expect to
    hurt the mass head far more than the naming head.
    """
    if keep >= 1.0:
        return img
    w, h = img.size
    nw, nh = max(int(w * keep), 1), max(int(h * keep), 1)
    left, top = (w - nw) // 2, (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.BILINEAR)


def _noise(img: Image.Image, sigma: float, seed: int = 0) -> Image.Image:
    if sigma <= 0:
        return img
    rng = np.random.default_rng(seed)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = a + rng.normal(0.0, sigma, a.shape).astype(np.float32)
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


_APPLY = {
    "blur":      lambda im, s: im if s <= 0 else im.filter(ImageFilter.GaussianBlur(radius=s)),
    "motion":    _motion_blur,
    "jpeg":      _jpeg,
    "downscale": _downscale,
    "temp":      _temp_shift,
    "dark":      lambda im, s: im if s >= 1.0 else ImageEnhance.Brightness(im).enhance(s),
    "crop":      _crop,
    "noise":     _noise,
}


def apply_degradation(img: Image.Image, spec: str | list[tuple[str, float]]) -> Image.Image:
    """Apply a spec (string or parsed list) to a PIL image. Deterministic."""
    steps = parse_spec(spec) if isinstance(spec, str) else spec
    for family, severity in steps:
        img = _APPLY[family](img.convert("RGB"), severity)
    return img.convert("RGB")


def rungs(family: str) -> list[str]:
    """Spec strings for every severity of one family, e.g. ['blur:0.0', ...]."""
    return [f"{family}:{s}" for s in LADDER[family]]


def all_rungs(include_screenshot: bool = True) -> list[str]:
    """Every rung worth running: clean once, then each family's non-baseline severities.

    Severity index 0 of every family is the identity (clean), so it is dropped here rather
    than matched on the formatted string — suffix matching silently ate `blur:1.0` too.
    """
    out = ["clean"]
    for family, severities in LADDER.items():
        out.extend(f"{family}:{s}" for s in severities[1:])
    if include_screenshot:
        out.extend(["screenshot", "phone"])
    return out


if __name__ == "__main__":
    import sys

    import n5k

    ids = n5k.usable_split("test", local_only=True)
    if not ids:
        sys.exit("no local images — run code/fetch_images.py --split test first")
    src = Image.open(n5k.IMAGES / ids[0] / "rgb.png").convert("RGB")
    specs = all_rungs()
    print(f"{len(specs)} rungs; demo on {ids[0]} ({src.size[0]}x{src.size[1]})")
    outdir = n5k.RESULTS / "degradation_examples"
    outdir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        out = apply_degradation(src, spec)
        out.save(outdir / f"{spec.replace(':', '_')}.png")
    print(f"wrote {len(specs)} example images to {outdir}")
