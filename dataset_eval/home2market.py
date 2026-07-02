"""Match a home-photo garment against the dataset "marketplace".

Pipeline:
    1. Run measure_jeans on the home photo (rembg + A4 by default).
    2. Load the dataset survey CSV (5 circumferences per garment).
    3. For each dataset garment marked 'ok', compute L2 distance to the
       query in z-score space (each measurement standardised by the
       dataset's own mean / std, so hip ~1000 doesn't dominate ankle ~200).
    4. Print the top-K rows and save a side-by-side comparison image
       (home photo on the left, K nearest dataset front photos to the right).

Usage:
    CIRCULAR_FASHION_ROOT=/path/to/circular_fashion_v2 \
        python dataset_eval/home2market.py path/to/home.jpg [--k 3] [--out out.jpg]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = Path(__file__).resolve().parent / "full_survey.csv"

MEASUREMENTS = ["waist", "hip", "thigh", "calf", "ankle"]


def measure_home_photo(image_path, manual_px_per_mm=None):
    """Return {"waist": mm, ...} for the home photo (or raise)."""
    with tempfile.TemporaryDirectory() as tmp:
        json_out = Path(tmp) / "r.json"
        cmd = [sys.executable, str(REPO_ROOT / "measure_jeans.py"),
               str(image_path), "--no-click", "--rembg",
               "--json", str(json_out)]
        if manual_px_per_mm is not None:
            cmd += ["--manual-px-per-mm", str(manual_px_per_mm)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"measure_jeans failed on the home photo:\n{r.stderr.strip()}"
            )
        d = json.loads(json_out.read_text())
    ms = d.get("measurements", {})
    out = {}
    for m in MEASUREMENTS:
        v = (ms.get(m) or {}).get("circumference_mm")
        if v is None:
            raise RuntimeError(f"home photo missing {m} measurement")
        out[m] = float(v)
    return out


def load_dataset(csv_path, dataset_root):
    """Return list of dict rows with parsed 5-vec + resolved front path."""
    root = Path(dataset_root)
    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            if r["status"] != "ok":
                continue
            try:
                vec = np.array([float(r[f"{m}_mm"]) for m in MEASUREMENTS])
            except (ValueError, TypeError):
                continue
            # Reconstruct front image path from ts + station.
            ts = r["ts"]
            year, mo = ts[:4], int(ts[5:7])
            months = ["jan", "feb", "mar", "apr", "may", "jun",
                     "jul", "aug", "sep", "oct", "nov", "dec"]
            folder = f"{months[mo-1]}{year}"
            front = root / r["station"] / folder / f"front_{ts}.jpg"
            rows.append({
                "ts": ts, "brand": r["brand"], "size": r["size"],
                "category": r["category"], "cut": r["cut"],
                "vec": vec, "front": front,
            })
    return rows


def rank_matches(query_vec, dataset_rows):
    """L2 in per-measurement z-score space. Returns sorted list of (dist, row)."""
    all_vecs = np.stack([r["vec"] for r in dataset_rows])
    mu = all_vecs.mean(axis=0)
    sd = all_vecs.std(axis=0, ddof=1)
    sd = np.where(sd < 1e-6, 1.0, sd)
    zs_data = (all_vecs - mu) / sd
    zs_query = (query_vec - mu) / sd
    dists = np.linalg.norm(zs_data - zs_query, axis=1)
    order = np.argsort(dists)
    return [(float(dists[i]), dataset_rows[i]) for i in order]


def build_comparison_image(home_path, home_vec, top_rows, out_path):
    """Save a 1 x (K+1) grid: home photo | top-K dataset photos, with captions."""
    k = len(top_rows)
    fig, axes = plt.subplots(1, k + 1, figsize=(3.2 * (k + 1), 5.8))
    if k == 0:
        axes = [axes]
    home_img = Image.open(home_path).convert("RGB")
    axes[0].imshow(home_img)
    axes[0].axis("off")
    home_caption = (
        "your home photo\n"
        f"W {home_vec[0]:.0f}  H {home_vec[1]:.0f}\n"
        f"T {home_vec[2]:.0f}  C {home_vec[3]:.0f}  A {home_vec[4]:.0f}  (mm)"
    )
    axes[0].set_title(home_caption, fontsize=9)

    for i, (dist, row) in enumerate(top_rows, 1):
        img = Image.open(row["front"]).convert("RGB")
        axes[i].imshow(img)
        axes[i].axis("off")
        caption = (
            f"#{i}: {row['brand']} · EU {row['size']} ({row['category']})\n"
            f"{row['cut'] or '—'}\n"
            f"W {row['vec'][0]:.0f}  H {row['vec'][1]:.0f}\n"
            f"T {row['vec'][2]:.0f}  C {row['vec'][3]:.0f}  A {row['vec'][4]:.0f}\n"
            f"distance = {dist:.2f}"
        )
        axes[i].set_title(caption, fontsize=8)

    fig.suptitle("home garment → nearest matches in the dataset marketplace",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path, help="Home photo of a garment.")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                    help="Dataset survey CSV (default: full_survey.csv).")
    ap.add_argument("--dataset-root", default=os.environ.get("CIRCULAR_FASHION_ROOT"),
                    help="Path to circular_fashion_v2 (for the front photos).")
    ap.add_argument("--manual-px-per-mm", type=float, default=None,
                    help="Skip A4 detection; use this scale directly.")
    ap.add_argument("--k", type=int, default=3, help="Number of matches.")
    ap.add_argument("--out", type=Path, default=Path("home2market_result.jpg"))
    args = ap.parse_args()
    if not args.dataset_root:
        ap.error("--dataset-root or CIRCULAR_FASHION_ROOT env var is required")

    print(f"Measuring {args.image} ...")
    home_ms = measure_home_photo(args.image, manual_px_per_mm=args.manual_px_per_mm)
    query = np.array([home_ms[m] for m in MEASUREMENTS])
    print("Home measurements (mm):")
    for m, v in home_ms.items():
        print(f"  {m:<6} {v:7.1f}")
    print()

    print(f"Loading dataset from {args.csv} ...")
    rows = load_dataset(args.csv, args.dataset_root)
    print(f"  {len(rows)} usable dataset garments\n")
    if not rows:
        print("No usable dataset rows — run dataset_eval/full_survey.py first.")
        return

    ranked = rank_matches(query, rows)
    top = ranked[:args.k]

    print(f"Top {args.k} matches:")
    print(f"{'#':<3} {'dist':>5} {'brand':<20} {'size':>5} {'cat':<7} {'cut':<20} "
          f"{'W':>5} {'H':>5} {'T':>5} {'C':>5} {'A':>5}")
    print("-" * 100)
    for i, (dist, r) in enumerate(top, 1):
        v = r["vec"]
        print(f"{i:<3} {dist:>5.2f} {r['brand'][:20]:<20} {r['size']:>5} "
              f"{r['category']:<7} {(r['cut'] or '')[:20]:<20} "
              f"{v[0]:>5.0f} {v[1]:>5.0f} {v[2]:>5.0f} {v[3]:>5.0f} {v[4]:>5.0f}")

    build_comparison_image(args.image, query, top, args.out)


if __name__ == "__main__":
    main()
