"""No-model image-quality metrics — the conditioning variables for Mondrian conformal.

These must be computable by the app on a *received* photo, with no knowledge of what
was done to it and without running the model. That is the whole point: the abstention
gate has to work on an image whose provenance is unknown.

  python code/quality.py --sample 60      # metric response across the whole ladder

Everything here is cheap (a few ms per image) and deterministic.
"""
from __future__ import annotations

import argparse
import io

import numpy as np
from PIL import Image


def _gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def laplacian_variance(img: Image.Image) -> float:
    """The standard blur metric: variance of the 4-neighbour Laplacian.

    Low = blurry. Note this is a *sharpness* detector, so additive high-frequency
    noise raises it — see the `noise` rung in degrade.py.
    """
    g = _gray(img)
    lap = (
        -4.0 * g[1:-1, 1:-1]
        + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    )
    return float(lap.var())


def tenengrad(img: Image.Image) -> float:
    """Sobel gradient energy — a second sharpness view, less noise-sensitive than Laplacian."""
    g = _gray(img)
    gx = g[1:-1, 2:] - g[1:-1, :-2]
    gy = g[2:, 1:-1] - g[:-2, 1:-1]
    return float(np.mean(gx * gx + gy * gy))


def blockiness(img: Image.Image) -> float:
    """Blind JPEG-artifact estimate: excess discontinuity on the 8x8 grid.

    Compares mean |difference| across column boundaries that fall on the JPEG block
    grid against those that do not. ~0 for a clean image, grows as quality drops.
    """
    g = _gray(img)
    d = np.abs(np.diff(g, axis=1))
    cols = np.arange(d.shape[1])
    on = d[:, (cols % 8) == 7]
    off = d[:, (cols % 8) != 7]
    if off.size == 0 or off.mean() == 0:
        return 0.0
    return float(on.mean() / off.mean() - 1.0)


def jpeg_quant_score(data: bytes | None) -> float:
    """Mean of the luminance quantization table when the source really is a JPEG.

    Higher = coarser quantization = lower quality. Returns nan for non-JPEG input,
    which is the honest answer: a re-encoded PNG carries no such evidence.
    """
    if not data:
        return float("nan")
    try:
        im = Image.open(io.BytesIO(data))
        table = getattr(im, "quantization", None)
        if not table:
            return float("nan")
        return float(np.mean(table[0]))
    except Exception:
        return float("nan")


def brightness(img: Image.Image) -> float:
    return float(_gray(img).mean())


def rms_contrast(img: Image.Image) -> float:
    return float(_gray(img).std())


def colour_ratio_rb(img: Image.Image) -> float:
    """Mean R / mean B — a white-balance / colour-temperature proxy. ~1 is neutral."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    b = a[..., 2].mean()
    return float(a[..., 0].mean() / b) if b > 0 else float("nan")


def saturation(img: Image.Image) -> float:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    mx, mn = a.max(axis=2), a.min(axis=2)
    return float(np.mean((mx - mn) / np.maximum(mx, 1.0)))


def high_freq_ratio(img: Image.Image) -> float:
    """Share of AC spectral energy sitting in the upper band. Falls under blur, rises under noise.

    The DC term carries almost all the energy in a photo, so it is excluded — with it
    included every rung reads ~0.00 and the metric is useless.
    """
    g = _gray(img)
    f = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    h, w = f.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    total = f.sum()
    return float(f[r > 0.25].sum() / total) if total > 0 else float("nan")


METRICS = {
    "lap_var": laplacian_variance,
    "tenengrad": tenengrad,
    "blockiness": blockiness,
    "brightness": brightness,
    "rms_contrast": rms_contrast,
    "colour_rb": colour_ratio_rb,
    "saturation": saturation,
    "hf_ratio": high_freq_ratio,
}


def measure(img: Image.Image, jpeg_bytes: bytes | None = None) -> dict[str, float]:
    out = {name: fn(img) for name, fn in METRICS.items()}
    out["jpeg_quant"] = jpeg_quant_score(jpeg_bytes)
    return out


def main() -> int:
    import n5k
    from degrade import all_rungs, apply_degradation

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60, help="dishes to average over")
    args = ap.parse_args()

    ids = n5k.usable_split("test", local_only=True)[: args.sample]
    if not ids:
        print("no local images — run code/fetch_images.py first")
        return 1
    imgs = [Image.open(n5k.IMAGES / i / "rgb.png").convert("RGB") for i in ids]

    specs = all_rungs()
    print(f"metric response over {len(imgs)} dishes, {len(specs)} rungs\n")
    cols = ["lap_var", "tenengrad", "blockiness", "hf_ratio", "brightness", "colour_rb"]
    print(f"{'rung':<18}" + "".join(f"{c:>12}" for c in cols))
    print("-" * (18 + 12 * len(cols)))
    base = None
    for spec in specs:
        vals = {c: [] for c in cols}
        for im in imgs:
            m = measure(apply_degradation(im, spec))
            for c in cols:
                vals[c].append(m[c])
        means = {c: float(np.mean(vals[c])) for c in cols}
        if base is None:
            base = means
        rel = "".join(f"{means[c]:>12.2f}" for c in cols)
        print(f"{spec:<18}{rel}")
    print("\nlap_var relative to clean (x): a value > 1 means the metric thinks the")
    print("degraded photo is SHARPER than the original — the failure mode the")
    print("adversarial stress test exploits.")
    for spec in ["blur:3.0", "jpeg:10.0", "noise:0.08", "noise:0.16"]:
        v = float(np.mean([measure(apply_degradation(im, spec))["lap_var"] for im in imgs]))
        print(f"  {spec:<14} {v / base['lap_var']:>6.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
