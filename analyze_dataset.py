"""Aggregate label statistics over the circular_fashion_v2 dataset and produce
plots + a printable summary.

Usage:
    python analyze_dataset.py --dataset-root PATH [--out-dir PATH]

The dataset root can also be supplied via the CIRCULAR_FASHION_ROOT env var.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_OUT = "dataset_analysis"

PANTS_TYPES = {"Jeans", "Trousers", "Shorts", "Tights", "Rain trousers", "Winter trousers"}

MONTH_ORDER = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]


def parse_month_folder(name):
    """Return (year, month_index) from a folder like 'dec2023', else None."""
    m = re.match(r"^([a-z]{3})(\d{4})$", name.strip().lower())
    if not m:
        return None
    mon = m.group(1)
    if mon not in MONTH_ORDER:
        return None
    return int(m.group(2)), MONTH_ORDER.index(mon)


def era_for(year, month_idx):
    """Pre-Jan 2023 = measuring-tape era; Jan 2023 onward = tile-grid era."""
    if year < 2023:
        return "tape"
    return "tile_grid"


def normalize_material(raw):
    """Best-effort normalization of free-text material strings."""
    if not raw or not raw.strip():
        return "(missing)"
    s = raw.strip().lower()
    s = s.replace("bomull", "cotton")      # swedish
    s = s.replace("ull", "wool") if "wool" not in s and s.endswith(" ull") else s
    s = re.sub(r"\s+", " ", s)
    s = s.replace("%cotton", "% cotton")
    s = s.replace("cott ", "cotton ")
    s = re.sub(r"\bcott\b", "cotton", s)
    s = s.replace("nylone", "nylon")
    s = s.replace("polyster", "polyester")
    s = s.replace("polyster", "polyester")
    if s in {"?", "??", "???", "unknown", "none", "n/a", "na", "-", "--"}:
        return "(unknown)"
    return s.strip(" .,;")


KNIT_HINTS = ("elastane", "spandex", "lycra", "viscose", "jersey", "rayon")
WOVEN_HINTS = ("denim", "cotton")  # cotton+nothing-else = often woven, but ambiguous


def fabric_class(material_norm):
    """Coarse woven/knit/stretchy classification from normalized material text."""
    if material_norm in {"(missing)", "(unknown)"}:
        return "unknown"
    m = material_norm
    if any(h in m for h in KNIT_HINTS):
        return "stretchy"
    if "polyester" in m and "cotton" not in m:
        return "synthetic"
    if "cotton" in m or "linen" in m or "wool" in m:
        return "woven_or_knit"
    return "other"


def walk_jsons(root):
    """Yield (station, year, month_idx, era, json_path) for each label JSON."""
    root = Path(root)
    for station_dir in sorted(root.glob("station*")):
        if not station_dir.is_dir():
            continue
        station = station_dir.name
        for month_dir in sorted(station_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            ym = parse_month_folder(month_dir.name)
            if ym is None:
                # e.g. test100 — include but tag year=0
                year, month_idx = 0, 0
                era = "other"
            else:
                year, month_idx = ym
                era = era_for(year, month_idx)
            for jp in month_dir.glob("labels_*.json"):
                yield station, year, month_idx, era, jp


def safe_load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def topn_bar(counter, n, title, out_path, xlabel="count"):
    items = counter.most_common(n)
    if not items:
        return
    labels = [str(k) for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(labels) + 1)))
    ax.barh(labels, vals)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def bar(counter, title, out_path, key_order=None, xlabel="count"):
    if key_order is not None:
        items = [(k, counter.get(k, 0)) for k in key_order]
    else:
        items = sorted(counter.items(), key=lambda kv: -kv[1])
    if not items:
        return
    labels = [str(k) for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(labels) + 2), 4))
    ax.bar(labels, vals)
    ax.set_ylabel(xlabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def stacked_time_series(by_period, title, out_path, ylabel="count"):
    """by_period: dict[(year, month_idx)] -> Counter over series labels."""
    if not by_period:
        return
    periods = sorted(by_period.keys())
    series = sorted({k for c in by_period.values() for k in c.keys()})
    bottoms = [0] * len(periods)
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(periods) + 2), 5))
    for s in series:
        vals = [by_period[p].get(s, 0) for p in periods]
        ax.bar(range(len(periods)), vals, bottom=bottoms, label=s)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    xt = [f"{MONTH_ORDER[m]}{y % 100:02d}" for (y, m) in periods]
    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels(xt, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def heatmap(matrix, row_labels, col_labels, title, out_path):
    if not matrix or not matrix[0]:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(col_labels) + 2),
                                     max(4, 0.4 * len(row_labels) + 1)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            if v == 0:
                continue
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v < max(map(max, matrix)) * 0.5 else "black",
                    fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=os.environ.get("CIRCULAR_FASHION_ROOT"),
                    help="Path to the circular_fashion_v2 dataset (or set CIRCULAR_FASHION_ROOT).")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()
    if not args.dataset_root:
        ap.error("--dataset-root or CIRCULAR_FASHION_ROOT env var is required")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    parse_errors = 0
    by_station = Counter()
    by_era = Counter()
    by_station_era = Counter()
    by_period_all = defaultdict(Counter)         # (y,m) -> Counter[station]
    by_period_pants = defaultdict(Counter)       # (y,m) -> Counter[station]

    type_all = Counter()
    type_pants_only = Counter()

    # Pants-only breakdowns
    p_category = Counter()
    p_size = Counter()
    p_material_raw = Counter()
    p_material_norm = Counter()
    p_fabric_class = Counter()
    p_pattern = Counter()
    p_colors = Counter()
    p_cut = Counter()
    p_brand = Counter()
    p_condition = Counter()
    p_usage = Counter()
    p_pilling = Counter()
    p_stains = Counter()
    p_holes = Counter()
    p_smell = Counter()
    p_season = Counter()
    p_type = Counter()
    p_type_x_pattern = defaultdict(Counter)
    p_type_x_category = defaultdict(Counter)
    p_size_x_category = defaultdict(Counter)

    for station, year, month_idx, era, jp in walk_jsons(args.dataset_root):
        total_files += 1
        data = safe_load(jp)
        if data is None:
            parse_errors += 1
            continue
        by_station[station] += 1
        by_era[era] += 1
        by_station_era[(station, era)] += 1
        if year > 0:
            by_period_all[(year, month_idx)][station] += 1

        gtype = (data.get("type") or "").strip()
        type_all[gtype] += 1

        if gtype not in PANTS_TYPES:
            continue
        if era != "tile_grid":
            # Track but don't include in main pants stats — tape era is unusable
            continue

        p_type[gtype] += 1
        if year > 0:
            by_period_pants[(year, month_idx)][station] += 1
        type_pants_only[gtype] += 1

        cat = (data.get("category") or "").strip() or "(missing)"
        p_category[cat] += 1

        sz = (data.get("size") or "").strip()
        sz_key = sz if sz else "(missing)"
        p_size[sz_key] += 1
        p_size_x_category[sz_key][cat] += 1

        mat = data.get("material") or ""
        if mat.strip():
            p_material_raw[mat.strip()] += 1
        norm = normalize_material(mat)
        p_material_norm[norm] += 1
        p_fabric_class[fabric_class(norm)] += 1

        pat = (data.get("pattern") or "").strip() or "(missing)"
        p_pattern[pat] += 1
        p_type_x_pattern[gtype][pat] += 1
        p_type_x_category[gtype][cat] += 1

        cols = data.get("colors") or []
        if isinstance(cols, list):
            for c in cols:
                if isinstance(c, str) and c.strip():
                    p_colors[c.strip()] += 1
        cuts = data.get("cut") or []
        if isinstance(cuts, list):
            for c in cuts:
                if isinstance(c, str) and c.strip():
                    p_cut[c.strip()] += 1

        br = (data.get("brand") or "").strip() or "(missing)"
        p_brand[br] += 1

        cond = data.get("condition")
        if cond is not None:
            p_condition[cond] += 1
        usage = (data.get("usage") or "").strip() or "(missing)"
        p_usage[usage] += 1
        pill = data.get("pilling")
        if pill is not None:
            p_pilling[pill] += 1
        for k, target in (("stains", p_stains), ("holes", p_holes), ("smell", p_smell)):
            v = (data.get(k) or "").strip() or "(missing)"
            target[v] += 1
        season = (data.get("season") or "").strip() or "(missing)"
        p_season[season] += 1

    pants_total = sum(p_type.values())

    # -------- Print summary --------
    def header(t):
        print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")

    header("DATASET SUMMARY")
    print(f"Total JSON files       : {total_files}")
    print(f"Parse errors           : {parse_errors}")
    print(f"By station             : {dict(by_station.most_common())}")
    print(f"By era                 : {dict(by_era.most_common())}")
    print(f"\nPants subset (tile-grid era): {pants_total}")
    print("By type:")
    for t, c in p_type.most_common():
        print(f"  {t:<18} {c}")

    # -------- Plots --------
    print(f"\nWriting plots to {out_dir.resolve()}")

    # 1. All-garment type distribution
    topn_bar(type_all, n=30,
             title="All garments — type distribution",
             out_path=out_dir / "01_type_all.png")

    # 2. Pants-only type distribution
    bar(p_type, "Pants subset — type distribution",
        out_path=out_dir / "02_type_pants.png",
        key_order=[t for t, _ in p_type.most_common()])

    # 3. Time series: pants intake per month, stacked by station
    stacked_time_series(by_period_pants,
                        title="Pants intake per month (stacked by station)",
                        out_path=out_dir / "03_pants_timeseries.png")

    # 4. Pants by category
    bar(p_category, "Pants — category", out_path=out_dir / "04_category.png")

    # 5. Pants by size — split numeric vs letter
    num_sizes = Counter({k: v for k, v in p_size.items() if k.strip().isdigit()})
    letter_sizes = Counter({
        k: v for k, v in p_size.items()
        if not k.strip().isdigit() and k != "(missing)" and k != "None"
    })
    missing_sizes = p_size.get("None", 0) + p_size.get("(missing)", 0)
    if num_sizes:
        ordered_keys = sorted(num_sizes.keys(), key=lambda k: int(k))
        bar(num_sizes, "Pants — numeric size",
            out_path=out_dir / "05a_size_numeric.png",
            key_order=ordered_keys)
    if letter_sizes:
        order = ["XXS", "XS", "S ", "S", "M ", "M", "L", "XL", "XXL", "XXXL", "onesize"]
        order = [k for k in order if k in letter_sizes]
        order += [k for k in letter_sizes if k not in order]
        bar(letter_sizes, "Pants — letter size",
            out_path=out_dir / "05b_size_letter.png",
            key_order=order)
    print(f"Pants with missing size: {missing_sizes} ({missing_sizes / max(1, pants_total):.1%})")

    # 6. Material (raw vs normalized)
    topn_bar(p_material_raw, n=25, title="Pants — material (raw, top 25)",
             out_path=out_dir / "06a_material_raw.png")
    topn_bar(p_material_norm, n=25, title="Pants — material (normalized, top 25)",
             out_path=out_dir / "06b_material_norm.png")
    bar(p_fabric_class, "Pants — coarse fabric class",
        out_path=out_dir / "06c_fabric_class.png")

    # 7. Pattern, colors, cut
    bar(p_pattern, "Pants — pattern", out_path=out_dir / "07_pattern.png")
    bar(p_colors, "Pants — colors (multi-valued)",
        out_path=out_dir / "08_colors.png")
    bar(p_cut, "Pants — cut (multi-valued)", out_path=out_dir / "09_cut.png")

    # 8. Brand
    topn_bar(p_brand, n=25, title="Pants — brand (top 25)",
             out_path=out_dir / "10_brand.png")

    # 9. Condition / usage / pilling / damage
    bar(p_condition, "Pants — condition (1=worst, 5=best)",
        out_path=out_dir / "11_condition.png",
        key_order=sorted(p_condition.keys()))
    bar(p_usage, "Pants — usage pathway", out_path=out_dir / "12_usage.png")
    bar(p_pilling, "Pants — pilling (1=worst, 5=best)",
        out_path=out_dir / "13_pilling.png",
        key_order=sorted(p_pilling.keys()))
    bar(p_stains, "Pants — stains", out_path=out_dir / "14_stains.png")
    bar(p_holes, "Pants — holes", out_path=out_dir / "15_holes.png")
    bar(p_smell, "Pants — smell", out_path=out_dir / "16_smell.png")
    bar(p_season, "Pants — season", out_path=out_dir / "17_season.png")

    # 10. Cross-tabs as heatmaps
    types = list(p_type.keys())
    pats = [p for p, _ in p_pattern.most_common()]
    mat = [[p_type_x_pattern[t].get(p, 0) for p in pats] for t in types]
    heatmap(mat, types, pats, "Type × Pattern (pants)",
            out_dir / "18_type_x_pattern.png")

    cats = [c for c, _ in p_category.most_common()]
    mat = [[p_type_x_category[t].get(c, 0) for c in cats] for t in types]
    heatmap(mat, types, cats, "Type × Category (pants)",
            out_dir / "19_type_x_category.png")

    # 11. "Usable for measurement" funnel
    # Define: tile_grid era ∩ pattern ∈ {None, missing} ∩ category ∈ {Ladies, Men} ∩
    #         size is numeric in 30..60 ∩ type ∈ {Jeans, Trousers}
    fn_total = pants_total
    fn_pattern_ok = sum(c for t in types for p, c in p_type_x_pattern[t].items()
                        if p in {"None", "(missing)"})
    # Re-walk for a clean funnel count — cheaper to do separately on small numbers.
    funnel = {
        "all_pants_tile_grid": pants_total,
        "pattern_None_or_missing": fn_pattern_ok,
    }

    # Count the size+category+type intersection cleanly
    cat_target = {"Ladies", "Men"}
    type_target = {"Jeans", "Trousers"}
    n_adult_jeans_trousers = 0
    n_clean_validation = 0
    for station, year, month_idx, era, jp in walk_jsons(args.dataset_root):
        if era != "tile_grid":
            continue
        data = safe_load(jp)
        if data is None:
            continue
        gtype = (data.get("type") or "").strip()
        if gtype not in type_target:
            continue
        cat = (data.get("category") or "").strip()
        if cat not in cat_target:
            continue
        sz = (data.get("size") or "").strip()
        if not sz.isdigit():
            continue
        szn = int(sz)
        if not (30 <= szn <= 60):
            continue
        n_adult_jeans_trousers += 1
        pat = (data.get("pattern") or "").strip()
        if pat in {"None", ""}:
            n_clean_validation += 1
    funnel["adult_jeans_or_trousers_with_numeric_size"] = n_adult_jeans_trousers
    funnel["+_pattern_None"] = n_clean_validation

    bar(Counter(funnel),
        "Validation funnel (pants → adult sized → unpatterned)",
        out_path=out_dir / "20_funnel.png",
        key_order=list(funnel.keys()))

    header("VALIDATION FUNNEL")
    for k, v in funnel.items():
        print(f"  {k:<45} {v}")

    print(f"\nDone. Plots in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
