"""Count tile-grid-era Jeans by named brand.

A 'named brand' = brand field is set, non-empty, and not one of the
sentinel values ('Not in the list', 'Missing').
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/Users/kristijanbartol/Downloads/circular_fashion_v2")
SENTINEL = {"", "not in the list", "missing", "none"}


def is_tile_grid(month_folder):
    name = month_folder.name.strip().lower()
    if len(name) < 7:
        return False
    months = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"]
    mon, yr = name[:3], name[3:]
    if mon not in months or not yr.isdigit():
        return False
    return int(yr) >= 2023


def main():
    counts = Counter()
    total_jeans = 0
    named = 0
    unnamed = Counter()

    for station_dir in sorted(ROOT.glob("station*")):
        for month_dir in station_dir.iterdir():
            if not month_dir.is_dir() or not is_tile_grid(month_dir):
                continue
            for jp in month_dir.glob("labels_*.json"):
                try:
                    data = json.loads(jp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if (data.get("type") or "").strip() != "Jeans":
                    continue
                total_jeans += 1
                brand = (data.get("brand") or "").strip()
                if brand.lower() in SENTINEL:
                    unnamed[brand or "(empty)"] += 1
                    continue
                counts[brand] += 1
                named += 1

    print(f"Tile-grid Jeans total            : {total_jeans}")
    print(f"With a named brand               : {named} ({named/max(1,total_jeans):.1%})")
    print(f"Without a named brand            : {total_jeans - named}")
    print(f"  Sentinel breakdown             : {dict(unnamed)}")
    print(f"Distinct named brands            : {len(counts)}")
    print()
    print(f"{'Brand':<35} count")
    print("-" * 45)
    for brand, c in counts.most_common():
        print(f"{brand:<35} {c}")

    # Plot top 30
    top = counts.most_common(30)
    labels = [b for b, _ in top][::-1]
    vals = [c for _, c in top][::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(labels) + 1)))
    ax.barh(labels, vals)
    ax.set_xlabel("number of jeans")
    ax.set_title(f"Jeans by brand — top 30 (of {len(counts)} named brands)")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    fig.tight_layout()
    out = Path("dataset_analysis/21_jeans_by_brand_top30.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nPlot: {out.resolve()}")


if __name__ == "__main__":
    main()
