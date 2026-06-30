"""Process all tile-grid jeans and dump hip_mm per garment to CSV.

Filter (mirrors the validation slice criteria but at scale):
  type == "Jeans"
  pattern in {None, "None"}
  brand is set and not a sentinel ("Not in the list", "Missing")
  size is a 2-digit numeric EU size
  category in {"Ladies", "Men"}
  month >= mar 2023 (tile-grid era)

For each image: run measure_jeans --rembg --manual-px-per-mm 0.66, capture
the hip circumference, apply a plausibility filter (HIP_MIN_MM <= hip <=
HIP_MAX_MM), record the outcome.

Output: dataset_eval/hip_survey.csv with one row per garment:
    ts, station, brand, size, category, cut, hip_mm, status
where status is one of {ok, failed_pipeline, failed_plausibility}.

Usage:
    CIRCULAR_FASHION_ROOT=/path/to/circular_fashion_v2 \
        python dataset_eval/hip_survey.py [--limit N] [--workers K]
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = Path(__file__).resolve().parent / "hip_survey.csv"

HIP_MIN_MM = 600.0
HIP_MAX_MM = 1500.0

BRAND_SENTINELS = {"", "not in the list", "missing", "none"}
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]


def parse_month(name):
    m = re.match(r"^([a-z]{3})(\d{4})$", name.strip().lower())
    if not m or m.group(1) not in MONTHS:
        return None
    return int(m.group(2)), MONTHS.index(m.group(1))


def is_tile_grid(year, month_idx):
    # First reliably tile-grid month is mar 2023 (verified empirically).
    return (year, month_idx) >= (2023, 2)


def walk_jeans(root):
    """Yield (jp, front, labels_dict, station) for each candidate jeans label."""
    for station_dir in sorted(root.glob("station*")):
        for month_dir in sorted(station_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            ym = parse_month(month_dir.name)
            if ym is None or not is_tile_grid(*ym):
                continue
            for jp in sorted(month_dir.glob("labels_*.json")):
                try:
                    d = json.loads(jp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if (d.get("type") or "").strip() != "Jeans":
                    continue
                if (d.get("pattern") or "None") not in (None, "None"):
                    continue
                if d.get("category") not in ("Ladies", "Men"):
                    continue
                brand = (d.get("brand") or "").strip()
                if brand.lower() in BRAND_SENTINELS:
                    continue
                size = str(d.get("size", ""))
                if not (size.isdigit() and len(size) == 2):
                    continue
                ts = jp.stem.replace("labels_", "")
                front = month_dir / f"front_{ts}.jpg"
                if not front.exists():
                    continue
                yield (ts, station_dir.name, brand, size, d.get("category"),
                       d.get("cut"), front)


def measure_one(front):
    """Run measure_jeans on one image. Return hip_mm or None, plus status string."""
    with tempfile.TemporaryDirectory() as tmp:
        json_out = Path(tmp) / "r.json"
        r = subprocess.run(
            [sys.executable, str(REPO_ROOT / "measure_jeans.py"), str(front),
             "--no-click", "--manual-px-per-mm", "0.66", "--rembg",
             "--json", str(json_out)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return None, "failed_pipeline"
        try:
            d = json.loads(json_out.read_text())
        except Exception:
            return None, "failed_pipeline"
    hip = d.get("measurements", {}).get("hip")
    if hip is None:
        return None, "failed_pipeline"
    hip_mm = hip.get("circumference_mm")
    if hip_mm is None:
        return None, "failed_pipeline"
    if not (HIP_MIN_MM <= hip_mm <= HIP_MAX_MM):
        return hip_mm, "failed_plausibility"
    return hip_mm, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=os.environ.get("CIRCULAR_FASHION_ROOT"))
    ap.add_argument("--limit", type=int, default=None, help="Cap on garments processed.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent subprocesses. rembg uses ~1 CPU each.")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()
    if not args.dataset_root:
        ap.error("--dataset-root or CIRCULAR_FASHION_ROOT env var is required")
    root = Path(args.dataset_root)

    candidates = list(walk_jeans(root))
    if args.limit:
        candidates = candidates[:args.limit]
    print(f"Candidates: {len(candidates)}", file=sys.stderr)

    cut_str = lambda c: ",".join(c) if isinstance(c, list) else (c or "")
    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(measure_one, front): (ts, station, brand, size, category, cut)
                   for (ts, station, brand, size, category, cut, front) in candidates}
        for i, fut in enumerate(as_completed(futures), 1):
            ts, station, brand, size, category, cut = futures[fut]
            try:
                hip_mm, status = fut.result()
            except Exception as e:
                hip_mm, status = None, f"exception:{e}"
            rows.append((ts, station, brand, size, category, cut_str(cut),
                         hip_mm if hip_mm is not None else "",
                         status))
            if i % 25 == 0 or i == len(candidates):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(candidates) - i) / rate if rate > 0 else 0
                print(f"[{i}/{len(candidates)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s",
                      file=sys.stderr)

    rows.sort(key=lambda r: r[0])  # sort by timestamp for determinism
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "station", "brand", "size", "category", "cut",
                    "hip_mm", "status"])
        w.writerows(rows)
    print(f"\nWrote {args.out} ({len(rows)} rows)", file=sys.stderr)

    by_status = {}
    for r in rows:
        by_status[r[-1]] = by_status.get(r[-1], 0) + 1
    print("Status breakdown:", file=sys.stderr)
    for s, c in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"  {s:<22} {c}", file=sys.stderr)


if __name__ == "__main__":
    main()
