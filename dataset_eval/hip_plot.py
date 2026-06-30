"""Plot hip distributions from the hip_survey.csv.

Usage:
    python dataset_eval/hip_plot.py
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median, mean, stdev

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = Path(__file__).resolve().parent / "hip_survey.csv"
OUT_DIR = REPO_ROOT / "dataset_analysis"


def load(csv_path):
    rows = []
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row["hip_mm"] = float(row["hip_mm"]) if row["hip_mm"] else None
            except ValueError:
                row["hip_mm"] = None
            rows.append(row)
    return rows


def summarize(rows):
    total = len(rows)
    status_counts = defaultdict(int)
    for r in rows:
        status_counts[r["status"]] += 1
    print(f"Total processed       : {total}")
    print("Status breakdown:")
    for s, c in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {s:<22} {c:>4} ({c/total:.1%})")
    print()

    by_brand_all = defaultdict(int)
    by_brand_ok = defaultdict(int)
    for r in rows:
        by_brand_all[r["brand"]] += 1
        if r["status"] == "ok":
            by_brand_ok[r["brand"]] += 1
    print(f"{'brand':<24} {'all':>4} {'ok':>4} {'keep%':>6}")
    print("-" * 42)
    for b, n in sorted(by_brand_all.items(), key=lambda kv: -kv[1]):
        ok = by_brand_ok[b]
        print(f"{b[:23]:<24} {n:>4} {ok:>4} {ok/n:>6.0%}")


def plot_brand_boxes(rows, out_path, min_n=8):
    """Boxplot of hip_mm per brand, restricted to brands with >= min_n ok rows."""
    by_brand = defaultdict(list)
    for r in rows:
        if r["status"] != "ok":
            continue
        by_brand[r["brand"]].append(r["hip_mm"])
    brands = sorted([b for b, vs in by_brand.items() if len(vs) >= min_n],
                    key=lambda b: -len(by_brand[b]))
    if not brands:
        print(f"(no brand has >= {min_n} ok rows)")
        return
    data = [by_brand[b] for b in brands]
    labels = [f"{b}\n(n={len(by_brand[b])})" for b in brands]

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(brands) + 2), 6))
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True, widths=0.6,
                    patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_edgecolor("#3182bd")
    for i, vs in enumerate(data, 1):
        ax.scatter([i + 0.0] * len(vs), vs, color="#08519c", s=10, alpha=0.4,
                   zorder=3)
    ax.set_ylabel("hip circumference (mm)")
    ax.set_title("Hip circumference distribution per brand (jeans, all sizes)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")


def plot_hip_by_size(rows, out_path, top_n=5, min_n=8):
    """Hip vs labelled size, points per garment, lines through brand medians."""
    by_brand_size = defaultdict(lambda: defaultdict(list))
    by_brand_n = defaultdict(int)
    for r in rows:
        if r["status"] != "ok":
            continue
        by_brand_size[r["brand"]][r["size"]].append(r["hip_mm"])
        by_brand_n[r["brand"]] += 1
    top = sorted([b for b in by_brand_n if by_brand_n[b] >= min_n],
                 key=lambda b: -by_brand_n[b])[:top_n]
    if not top:
        print(f"(no brand has >= {min_n} ok rows)")
        return

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, brand in enumerate(top):
        sizes = sorted(by_brand_size[brand].keys(), key=lambda s: int(s))
        all_pts_x, all_pts_y = [], []
        med_x, med_y = [], []
        for s in sizes:
            vs = by_brand_size[brand][s]
            for v in vs:
                all_pts_x.append(int(s))
                all_pts_y.append(v)
            if len(vs) >= 2:
                med_x.append(int(s))
                med_y.append(median(vs))
        c = cmap(i)
        ax.scatter(all_pts_x, all_pts_y, color=c, s=18, alpha=0.5,
                   label=f"{brand} (n={by_brand_n[brand]})")
        if med_x:
            ax.plot(med_x, med_y, color=c, linewidth=2, alpha=0.9)
    ax.set_xlabel("labelled EU size")
    ax.set_ylabel("measured hip (mm)")
    ax.set_title(f"Measured hip vs labelled size (top {len(top)} brands by sample)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--min-n", type=int, default=8,
                    help="Min ok-rows to include a brand in plots.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load(args.csv)
    summarize(rows)
    plot_brand_boxes(rows, args.out_dir / "22_hip_by_brand.png",
                     min_n=args.min_n)
    plot_hip_by_size(rows, args.out_dir / "23_hip_by_size.png",
                     min_n=args.min_n)


if __name__ == "__main__":
    main()
