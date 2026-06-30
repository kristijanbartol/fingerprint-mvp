"""Front-vs-back hip consistency: empirical floor on pipeline noise.

For each garment with a successful front-image hip measurement
(status='ok' in hip_survey.csv), run the same pipeline on the matching
back_<ts>.jpg. If the two views measured the same physical garment, any
difference between hip_front and hip_back is *not* garment variation —
it's pipeline + lay-out noise. So:

    sigma_pipeline = std(hip_front - hip_back) / sqrt(2)

is an empirical estimate of the noise of a single hip measurement. The
"real" within-cell garment-to-garment std is then approximately

    sigma_real = sqrt(observed_sigma^2 - sigma_pipeline^2)

CSV out: dataset_eval/front_back_pairs.csv with one row per garment.
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

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
SURVEY_CSV = Path(__file__).resolve().parent / "hip_survey.csv"
OUT_CSV = Path(__file__).resolve().parent / "front_back_pairs.csv"

MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]

HIP_MIN_MM = 600.0
HIP_MAX_MM = 1500.0


def back_path_for(root, station, ts):
    """Reconstruct .../<station>/<mon><yyyy>/back_<ts>.jpg from ts and station."""
    m = re.match(r"^(\d{4})_(\d{2})_(\d{2})_", ts)
    if not m:
        return None
    year = int(m.group(1))
    month_idx = int(m.group(2)) - 1
    if not (0 <= month_idx < 12):
        return None
    folder = f"{MONTH_NAMES[month_idx]}{year}"
    return root / station / folder / f"back_{ts}.jpg"


def measure_hip(image_path):
    with tempfile.TemporaryDirectory() as tmp:
        json_out = Path(tmp) / "r.json"
        r = subprocess.run(
            [sys.executable, str(REPO_ROOT / "measure_jeans.py"), str(image_path),
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
    if hip is None or hip.get("circumference_mm") is None:
        return None, "failed_pipeline"
    v = hip["circumference_mm"]
    if not (HIP_MIN_MM <= v <= HIP_MAX_MM):
        return v, "failed_plausibility"
    return v, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=os.environ.get("CIRCULAR_FASHION_ROOT"))
    ap.add_argument("--survey-csv", type=Path, default=SURVEY_CSV)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not args.dataset_root:
        ap.error("--dataset-root or CIRCULAR_FASHION_ROOT is required")
    root = Path(args.dataset_root)

    front_rows = []
    with args.survey_csv.open() as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            try:
                r["hip_mm"] = float(r["hip_mm"])
            except (TypeError, ValueError):
                continue
            front_rows.append(r)
    if args.limit:
        front_rows = front_rows[:args.limit]
    print(f"Front rows to pair: {len(front_rows)}", file=sys.stderr)

    def job(row):
        bp = back_path_for(root, row["station"], row["ts"])
        if bp is None or not bp.exists():
            return row, None, "missing_back"
        hip_back, status = measure_hip(bp)
        return row, hip_back, status

    pairs = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, r) for r in front_rows]
        for i, fut in enumerate(as_completed(futures), 1):
            row, hip_back, status = fut.result()
            pairs.append((row["ts"], row["station"], row["brand"], row["size"],
                          row["category"], row["cut"], row["hip_mm"],
                          hip_back if hip_back is not None else "",
                          status))
            if i % 25 == 0 or i == len(futures):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed else 0
                eta = (len(futures) - i) / rate if rate else 0
                print(f"[{i}/{len(futures)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s",
                      file=sys.stderr)

    pairs.sort(key=lambda r: r[0])
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "station", "brand", "size", "category", "cut",
                    "hip_front_mm", "hip_back_mm", "back_status"])
        w.writerows(pairs)
    print(f"\nWrote {args.out} ({len(pairs)} rows)", file=sys.stderr)

    diffs = [float(p[6]) - float(p[7]) for p in pairs
             if p[7] != "" and p[8] == "ok"]
    print(f"\nUsable pairs (both views ok): {len(diffs)} / {len(pairs)}",
          file=sys.stderr)
    if not diffs:
        return
    diffs = np.array(diffs)
    abs_diffs = np.abs(diffs)
    print(f"Mean front-back  : {diffs.mean():+.1f} mm  (a non-zero mean would "
          f"indicate a systematic front/back bias)", file=sys.stderr)
    print(f"Std (front-back) : {diffs.std(ddof=1):.1f} mm", file=sys.stderr)
    print(f"Mean |front-back|: {abs_diffs.mean():.1f} mm", file=sys.stderr)
    print(f"Median |front-back|: {np.median(abs_diffs):.1f} mm", file=sys.stderr)
    sigma_pipe = diffs.std(ddof=1) / np.sqrt(2)
    print(f"\nImplied per-image pipeline noise (sigma_pipeline): {sigma_pipe:.1f} mm",
          file=sys.stderr)
    print(f"At the brand+size+category+cut control level, observed within-cell "
          f"std for H&M was 116 mm; implied real-garment std is "
          f"sqrt(116^2 - {sigma_pipe:.0f}^2) = "
          f"{np.sqrt(max(0, 116**2 - sigma_pipe**2)):.0f} mm.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
