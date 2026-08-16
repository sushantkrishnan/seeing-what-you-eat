"""Download Nutrition5k overhead RGB images over plain HTTPS.

No gsutil, no gcloud, no auth — the bucket is public (see CLAUDE.md).

  python code/fetch_images.py --manifest        # rebuild the bucket listing
  python code/fetch_images.py --split test      # ~507 dishes, ~0.2 GB
  python code/fetch_images.py --all             # 3,490 dishes, ~1.45 GB
  python code/fetch_images.py --split test --limit 50   # smoke test

Resumable: a dish whose rgb.png already exists at the right size is skipped, so
re-running after an interruption costs only the missing files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import n5k

LIST_API = "https://storage.googleapis.com/storage/v1/b/nutrition5k_dataset/o"
PREFIX = "nutrition5k_dataset/imagery/realsense_overhead/"


def build_manifest() -> dict[str, list[str]]:
    """Enumerate every object under realsense_overhead/ via the JSON API."""
    files: dict[str, dict[str, int]] = {}
    token = None
    while True:
        url = f"{LIST_API}?prefix={PREFIX}&maxResults=1000&fields=items(name,size),nextPageToken"
        if token:
            url += f"&pageToken={token}"
        with urllib.request.urlopen(url, timeout=60) as r:
            page = json.load(r)
        for obj in page.get("items", []):
            parts = obj["name"].split("/")
            if len(parts) >= 5:
                files.setdefault(parts[3], {})[parts[4]] = int(obj["size"])
        token = page.get("nextPageToken")
        if not token:
            break
        print(f"  ...{len(files)} dishes listed", end="\r", file=sys.stderr)
    manifest = {k: sorted(v) for k, v in files.items()}
    n5k.DATA.mkdir(parents=True, exist_ok=True)
    with open(n5k.DATA / "n5k_overhead_manifest.json", "w") as fh:
        json.dump(manifest, fh)
    with open(n5k.DATA / "n5k_overhead_sizes.json", "w") as fh:
        json.dump(files, fh)
    total = sum(v.get("rgb.png", 0) for v in files.values())
    print(f"manifest: {len(files)} dishes, rgb.png total {total / 1e9:.2f} GB")
    return manifest


def _expected_sizes() -> dict[str, int]:
    path = n5k.DATA / "n5k_overhead_sizes.json"
    if not path.exists():
        return {}
    with open(path) as fh:
        return {k: v.get("rgb.png", 0) for k, v in json.load(fh).items()}


def fetch_one(dish_id: str, expected: int = 0, retries: int = 3) -> tuple[str, str, int]:
    """Return (dish_id, status, bytes). Writes atomically via a .part file."""
    dish = n5k.IMAGES / dish_id
    dest = dish / "rgb.png"
    if dest.exists() and (expected == 0 or dest.stat().st_size == expected):
        return dish_id, "skip", dest.stat().st_size
    dish.mkdir(parents=True, exist_ok=True)
    url = f"{n5k.GCS}/imagery/realsense_overhead/{dish_id}/rgb.png"
    tmp = dish / "rgb.png.part"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
                blob = r.read()
                fh.write(blob)
            if expected and len(blob) != expected:
                raise OSError(f"size mismatch {len(blob)} != {expected}")
            tmp.replace(dest)
            return dish_id, "ok", len(blob)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            if attempt == retries - 1:
                tmp.unlink(missing_ok=True)
                return dish_id, f"FAIL {exc}", 0
            time.sleep(2**attempt)
    return dish_id, "FAIL", 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true", help="rebuild the bucket listing and exit")
    ap.add_argument("--split", choices=["train", "test"], help="download one official split")
    ap.add_argument("--all", action="store_true", help="download every dish with imagery")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of dishes (smoke test)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if args.manifest:
        build_manifest()
        return 0

    if args.all:
        ids = sorted(n5k.available_image_ids())
    elif args.split:
        ids = n5k.usable_split(args.split)
    else:
        ap.error("pass --manifest, --split {train,test}, or --all")
    if args.limit:
        ids = ids[: args.limit]

    sizes = _expected_sizes()
    todo_bytes = sum(sizes.get(i, 0) for i in ids)
    print(f"{len(ids)} dishes queued (~{todo_bytes / 1e9:.2f} GB before skips)")

    done = failed = skipped = 0
    got = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, i, sizes.get(i, 0)): i for i in ids}
        for fut in as_completed(futures):
            _, status, nbytes = fut.result()
            if status == "ok":
                done += 1
                got += nbytes
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
                print(f"\n  {futures[fut]}: {status}", file=sys.stderr)
            n = done + skipped + failed
            if n % 25 == 0 or n == len(ids):
                rate = got / max(time.time() - t0, 1e-6) / 1e6
                print(
                    f"  {n}/{len(ids)}  ok={done} skip={skipped} fail={failed}  "
                    f"{got / 1e9:.2f} GB  {rate:.1f} MB/s",
                    end="\r",
                )
    print(f"\ndone in {time.time() - t0:.0f}s — {done} downloaded, {skipped} already present, "
          f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
