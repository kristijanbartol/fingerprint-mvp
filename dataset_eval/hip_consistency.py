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


def plot_single_brand_progression(summaries, level_labels, focus_brand,
                                  out_path):
    """3 vertical bars for one brand across control levels.

    A horizontal band shows the range of the other qualifying brands at
    the strictest control level, so the viewer sees the focus brand's
    last bar dropping into the mid-pack range.
    """
    bars = []
    for s, lvl in zip(summaries, level_labels):
        row = next((r for r in s if r[0] == focus_brand), None)
        if row is None:
            continue
        _, n, std, lo, hi = row
        bars.append({"level": lvl, "n": n, "std": std, "lo": lo, "hi": hi})
    if not bars:
        print(f"(no data for {focus_brand!r})")
        return

    others = [r for r in summaries[-1] if r[0] != focus_brand]
    if others:
        ref_lo = min(r[2] for r in others)
        ref_hi = max(r[2] for r in others)
        ref_n = len(others)
    else:
        ref_lo = ref_hi = ref_n = None

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x = np.arange(len(bars))
    stds = [b["std"] for b in bars]
    errs_lo = [b["std"] - b["lo"] for b in bars]
    errs_hi = [b["hi"] - b["std"] for b in bars]
    colors = ["#cb6f57", "#e8a07c", "#74c476"]
    ax.bar(x, stds, yerr=[errs_lo, errs_hi], color=colors[:len(bars)],
           edgecolor="#333", width=0.6, capsize=6, zorder=3)
    for xi, b in zip(x, bars):
        ax.text(xi, b["std"] + 6, f"{b['std']:.0f} mm",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
                zorder=4)

    if ref_lo is not None:
        ax.axhspan(ref_lo, ref_hi, color="#9ecae1", alpha=0.3, zorder=1)
        # Put the label outside the bar area on the right.
        ax.set_xlim(-0.5, len(bars) - 0.5 + 0.6)
        ax.annotate(
            f"mid-pack range\n({ref_n} other brands,\nfully controlled)",
            xy=(len(bars) - 1 + 0.32, (ref_lo + ref_hi) / 2),
            xytext=(len(bars) - 0.5 + 0.05, (ref_lo + ref_hi) / 2),
            ha="left", va="center", fontsize=8.5, color="#08519c",
            fontstyle="italic",
            arrowprops=dict(arrowstyle="->", color="#08519c", lw=0.8),
            zorder=2,
        )

    short_labels = [
        "labelled size only",
        "+ category\n(Ladies/Men)",
        "+ cut\n(Regular/Tight/…)",
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels[:len(bars)])
    ax.set_ylabel("Hip residual std (mm)")
    ax.set_title(f"{focus_brand}: most of the apparent sizing inconsistency\n"
                 f"comes from mixing cut variants under each EU size",
                 fontsize=12)
    ax.set_ylim(0, max(b["hi"] for b in bars) * 1.1)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"\nSaved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--focus-brand", default="H&M",
                    help="Brand to feature in the single-brand zoom plot.")
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
    plot_single_brand_progression(summaries, labels, args.focus_brand,
                                  args.out_dir / "25_hip_consistency_focus.png")


if __name__ == "__main__":
    main()
