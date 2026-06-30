"""Within-brand hip consistency, controlling for labelled size.

For each garment, residual = hip_mm - median(hip_mm at same brand+size).
The std of residuals within a brand is the brand's *internal* sizing
consistency: how tightly the actual hip clusters around the brand's
typical value at any given labelled size. Lower std = more consistent
brand. This decouples consistency from any cross-brand bias.

Only (brand, size) cells with at least 2 garments contribute (a cell of
1 has zero residual by construction and would bias the std downward).
Brands need at least MIN_RESIDUALS qualifying residuals to be plotted.

The plot shows a 95% bootstrap CI on each brand's std so the reader can
tell signal from sample-size noise.

Usage:
    python dataset_eval/hip_consistency.py
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = Path(__file__).resolve().parent / "hip_survey.csv"
OUT_DIR = REPO_ROOT / "dataset_analysis"

MIN_CELL_N = 2          # min garments in a (brand, size) cell to include
MIN_RESIDUALS = 8       # min qualifying residuals to plot a brand
N_BOOTSTRAP = 2000


def load_ok(csv_path):
    out = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok":
                continue
            try:
                row["hip_mm"] = float(row["hip_mm"])
            except (TypeError, ValueError):
                continue
            out.append(row)
    return out


def residuals_per_brand(rows, extra_keys=()):
    """Compute residuals = hip - median(brand+size+extras).

    extra_keys: tuple of additional column names to include in the cell key.
    Empty tuple → (brand, size); ("category",) → (brand, size, category);
    ("category", "cut") → (brand, size, category, cut); etc.
    """
    cells = defaultdict(list)
    for r in rows:
        key = (r["brand"], r["size"]) + tuple(r.get(k, "") for k in extra_keys)
        cells[key].append(r["hip_mm"])
    out = defaultdict(list)
    cell_descriptions = []
    for key, vals in cells.items():
        if len(vals) < MIN_CELL_N:
            continue
        m = median(vals)
        for v in vals:
            out[key[0]].append(v - m)
        cell_descriptions.append((key, len(vals)))
    return out, cell_descriptions


def bootstrap_std_ci(values, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    stds = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stds[i] = vals[idx].std(ddof=1) if n > 1 else 0.0
    return float(np.percentile(stds, 2.5)), float(np.percentile(stds, 97.5))


def compute_summary(rows, extra_keys):
    per_brand, _ = residuals_per_brand(rows, extra_keys)
    qualifying = [(b, vals) for b, vals in per_brand.items()
                  if len(vals) >= MIN_RESIDUALS]
    out = []
    for brand, vals in qualifying:
        s = float(np.std(vals, ddof=1))
        lo, hi = bootstrap_std_ci(vals)
        out.append((brand, len(vals), s, lo, hi))
    out.sort(key=lambda r: r[2])
    return out


def print_summary(title, summary):
    print(f"\n{title}")
    print(f"{'brand':<24} {'n_res':>5} {'std (mm)':>9} {'95% CI (mm)':>17}")
    print("-" * 60)
    for brand, n, s, lo, hi in summary:
        print(f"{brand:<24} {n:>5} {s:>9.1f} {f'[{lo:.0f}, {hi:.0f}]':>17}")


def plot_grouped(summaries, labels, out_path, colors=None):
    """Side-by-side bars per brand, one bar per control level."""
    brands = sorted({b for s in summaries for b, *_ in s},
                    key=lambda b: -next((row[1] for s in summaries
                                         for row in s if row[0] == b), 0))
    if not brands:
        return
    by_brand = [{row[0]: row for row in s} for s in summaries]

    n_groups = len(brands)
    n_bars = len(summaries)
    width = 0.8 / n_bars
    y = np.arange(n_groups)
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.55 * n_groups + 2)))
    colors = colors or ["#74c476", "#fd8d3c", "#6baed6"]
    for i, (s, label, color) in enumerate(zip(summaries, labels, colors)):
        ys = []
        xs = []
        errs_lo, errs_hi = [], []
        ns = []
        for j, b in enumerate(brands):
            row = by_brand[i].get(b)
            if row is None:
                xs.append(0)
                errs_lo.append(0)
                errs_hi.append(0)
                ns.append(0)
            else:
                xs.append(row[2])
                errs_lo.append(row[2] - row[3])
                errs_hi.append(row[4] - row[2])
                ns.append(row[1])
            ys.append(j - 0.4 + i * width + width / 2)
        ax.barh(ys, xs, xerr=[errs_lo, errs_hi], height=width * 0.9,
                color=color, edgecolor="#444", capsize=2, label=label)
        for yi, xi, ni in zip(ys, xs, ns):
            if xi > 0:
                ax.text(xi + 4, yi, f"{xi:.0f}  (n={ni})", va="center",
                        fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(brands)
    ax.invert_yaxis()
    ax.set_xlabel("std of hip residuals (mm)")
    ax.set_title("Within-brand hip consistency at increasing control levels\n"
                 "lower = more consistent; error bars are 95% bootstrap CI")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nSaved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_ok(args.csv)

    summaries = []
    labels = [
        "control: brand+size",
        "control: brand+size+category",
        "control: brand+size+category+cut",
    ]
    extra_keys_list = [(), ("category",), ("category", "cut")]
    for extras, label in zip(extra_keys_list, labels):
        summary = compute_summary(rows, extras)
        print_summary(label, summary)
        summaries.append(summary)

    plot_grouped(summaries, labels,
                 args.out_dir / "24_hip_consistency_by_brand.png")


if __name__ == "__main__":
    main()
