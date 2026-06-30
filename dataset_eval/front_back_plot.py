"""Summarise front-back hip pairs and plot the noise distribution.

Reads dataset_eval/front_back_pairs.csv, prints noise statistics, and
saves a two-panel plot:
  (left)  scatter hip_back vs hip_front with y=x diagonal
  (right) histogram of (hip_front - hip_back) with a normal fit and
          the implied sigma_pipeline annotation.

Also reports the corrected within-cell std implied by subtracting the
empirical pipeline noise.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = Path(__file__).resolve().parent / "front_back_pairs.csv"
OUT_DIR = REPO_ROOT / "dataset_analysis"

OBSERVED_BY_BRAND = {
    "H&M":         (172, 158, 116),
    "Kappahl":     (121, 123, 116),
    "Lindex":      (114, 108,  91),
    "Zara":        (128, 128, 110),
    "Gina Tricot": ( 98,  98, 109),
    "Isolde":      (113, 113, 103),
}


def load_pairs(csv_path):
    diffs = []
    fronts, backs = [], []
    n_total = n_ok = 0
    for row in csv.DictReader(csv_path.open()):
        n_total += 1
        if row["back_status"] != "ok":
            continue
        try:
            f = float(row["hip_front_mm"])
            b = float(row["hip_back_mm"])
        except (TypeError, ValueError):
            continue
        n_ok += 1
        fronts.append(f)
        backs.append(b)
        diffs.append(f - b)
    return np.array(fronts), np.array(backs), np.array(diffs), n_total, n_ok


def corrected(observed_std, sigma_pipe):
    return float(np.sqrt(max(0.0, observed_std**2 - sigma_pipe**2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    f, b, d, n_total, n_ok = load_pairs(args.csv)
    print(f"Pairs: {n_ok} usable / {n_total} attempted")
    if n_ok == 0:
        return

    mean = d.mean()
    std = d.std(ddof=1)
    mad = float(np.median(np.abs(d - np.median(d))))
    sigma_pipe = std / np.sqrt(2)
    sigma_pipe_robust = (1.4826 * mad) / np.sqrt(2)

    print(f"Mean front-back  : {mean:+.1f} mm  (systematic bias if non-zero)")
    print(f"Std (front-back) : {std:.1f} mm")
    print(f"Median |diff|    : {np.median(np.abs(d)):.1f} mm")
    print(f"Robust std (1.4826·MAD): {1.4826 * mad:.1f} mm")
    print()
    print(f"Implied per-image pipeline noise (sigma_pipeline) : {sigma_pipe:.1f} mm")
    print(f"Robust per-image pipeline noise (MAD-based)        : {sigma_pipe_robust:.1f} mm")
    print()
    print("Corrected within-cell garment std (sqrt(observed^2 - sigma_pipe^2)):")
    print(f"{'brand':<14} {'size only':>11} {'+ cat':>11} {'+ cut':>11}")
    print("-" * 50)
    for brand, (s_only, s_cat, s_cut) in OBSERVED_BY_BRAND.items():
        print(f"{brand:<14} "
              f"{corrected(s_only, sigma_pipe):>7.0f} mm  "
              f"{corrected(s_cat,  sigma_pipe):>7.0f} mm  "
              f"{corrected(s_cut,  sigma_pipe):>7.0f} mm")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Scatter
    lo, hi = min(f.min(), b.min()) - 30, max(f.max(), b.max()) + 30
    ax1.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.4)
    ax1.scatter(f, b, s=14, alpha=0.55, color="#3182bd", edgecolor="none")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_aspect("equal")
    ax1.set_xlabel("hip from front photo (mm)")
    ax1.set_ylabel("hip from back photo (mm)")
    ax1.set_title(f"Same garment, two views  (n={n_ok})")
    ax1.grid(alpha=0.3)

    # Histogram
    bins = np.linspace(-300, 300, 41)
    ax2.hist(d, bins=bins, color="#9ecae1", edgecolor="#3182bd")
    ax2.axvline(0, color="k", linewidth=0.8, alpha=0.6)
    ax2.axvline(mean, color="#cb6f57", linewidth=1.5,
                label=f"mean = {mean:+.1f} mm")
    ax2.set_xlabel("hip_front − hip_back (mm)")
    ax2.set_ylabel("count")
    ax2.set_title(f"Difference distribution\n"
                  f"std = {std:.0f} mm  →  σ_pipeline ≈ {sigma_pipe:.0f} mm")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out = args.out_dir / "26_front_back_noise.png"
    fig.savefig(out, dpi=140)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
