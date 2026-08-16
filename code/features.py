"""Frozen-backbone feature extraction — the one inference pass.

Everything downstream (naming head, mass head, conformal, abstention) is a groupby
on what this produces, which is what keeps a stalled teammate off the critical path.

  python code/features.py --split test --limit 50          # smoke test
  python code/features.py --split train                    # cache clean train features
  python code/features.py --split test --degradation blur:3 # a rung of the ladder

Features saved per dish: [CLS ; mean(patch tokens) ; std(patch tokens)].
The patch statistics matter because the mass head needs spatial extent, which a
pooled CLS embedding alone represents poorly.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

import n5k
from degrade import apply_degradation, parse_spec

DEFAULT_MODEL = "vit_base_patch16_clip_224.openai"


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_backbone(name: str, device: torch.device):
    """Load a frozen, eval-mode backbone plus its matching preprocessing transform."""
    model = timm.create_model(name, pretrained=True, num_classes=0)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    return model, transform, cfg


@torch.no_grad()
def embed_batch(model, batch: torch.Tensor) -> np.ndarray:
    """[CLS ; mean(patches) ; std(patches)] for a batch of preprocessed images."""
    tokens = model.forward_features(batch)          # (B, N, D) for ViTs
    if tokens.ndim == 4:                            # conv backbones: (B, D, H, W)
        tokens = tokens.flatten(2).transpose(1, 2)
    n_prefix = getattr(model, "num_prefix_tokens", 1)
    cls = tokens[:, 0] if n_prefix else tokens.mean(1)
    patches = tokens[:, n_prefix:] if n_prefix else tokens
    feat = torch.cat([cls, patches.mean(1), patches.std(1)], dim=1)
    return feat.float().cpu().numpy()


def extract(
    dish_ids: list[str],
    model_name: str = DEFAULT_MODEL,
    degradation: str = "clean",
    device: str = "auto",
    batch_size: int = 32,
) -> tuple[np.ndarray, list[str]]:
    dev = pick_device(device)
    model, transform, cfg = load_backbone(model_name, dev)
    spec = parse_spec(degradation)
    print(f"backbone {model_name} on {dev} | input {cfg['input_size']} | degradation {degradation}")

    feats: list[np.ndarray] = []
    kept: list[str] = []
    buf: list[torch.Tensor] = []
    buf_ids: list[str] = []
    t0 = time.time()

    def flush():
        if not buf:
            return
        feats.append(embed_batch(model, torch.stack(buf).to(dev)))
        kept.extend(buf_ids)
        buf.clear()
        buf_ids.clear()

    for i, did in enumerate(dish_ids):
        path = n5k.IMAGES / did / "rgb.png"
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        img = apply_degradation(img, spec)
        buf.append(transform(img))
        buf_ids.append(did)
        if len(buf) == batch_size:
            flush()
            n = len(kept)
            print(f"  {n}/{len(dish_ids)}  {n / max(time.time() - t0, 1e-9):.1f} img/s", end="\r")
    flush()

    dt = time.time() - t0
    print(f"\n{len(kept)} dishes embedded in {dt:.1f}s ({len(kept) / max(dt, 1e-9):.1f} img/s)")
    return np.concatenate(feats) if feats else np.zeros((0, 0), np.float32), kept


def cache_path(model_name: str, split: str, degradation: str) -> Path:
    tag = degradation.replace(":", "").replace(".", "p")
    return n5k.CACHE / f"{model_name.split('.')[0]}__{split}__{tag}.npz"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "fit", "calib"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--degradation", default="clean", help="e.g. clean, blur:3, jpeg:30, temp:0.4")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.split in ("fit", "calib"):
        fit, calib = n5k.fit_calib_split(local_only=True)
        ids = fit if args.split == "fit" else calib
    else:
        ids = n5k.usable_split(args.split, local_only=True)
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        print("no dishes with local images — run code/fetch_images.py first")
        return 1

    feats, kept = extract(ids, args.model, args.degradation, args.device, args.batch_size)
    n5k.CACHE.mkdir(parents=True, exist_ok=True)
    out = cache_path(args.model, args.split, args.degradation)
    np.savez_compressed(
        out, features=feats, dish_ids=np.array(kept),
        model=args.model, degradation=args.degradation,
    )
    print(f"wrote {out}  shape={feats.shape}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
