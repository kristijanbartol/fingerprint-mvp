"""Render the Garment Marketplace mockup: a tool-style single image that
takes a home photo of a garment and presents:

  - the home photo (annotated with A4 + labelled measurement lines)
  - the recommended pick with brand / size / cut / material / price
  - per-measurement fit gauge + overall fit score
  - rule-based fit interpretation ("close through waist/thigh; ankle wider…")
  - rule-based material interpretation ("stretch poly-cotton, softer than…")
  - two alternate candidates as a bottom strip

The mock is a proof-of-concept of the product surface. Everything is
computed live from the dataset labels; no numbers are hand-crafted.

Usage:
    CIRCULAR_FASHION_ROOT=/path/to/circular_fashion_v2 \
        python dataset_eval/marketplace_mock.py path/to/home.jpg \
            [--out docs/marketplace_mock.jpg]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

# Reuse the measurement + matching plumbing already in home2market.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from home2market import (  # noqa: E402
    MEASUREMENTS,
    load_dataset,
    measure_home_photo,
    rank_matches,
    render_home_panel,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = Path(__file__).resolve().parent / "full_survey.csv"

# ---------------------------------------------------------------------------
# Metadata + interpretation
# ---------------------------------------------------------------------------

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]

# Gaussian tolerance kernel. Sigma combines our empirical pipeline noise
# (~37 mm) with a rough garment-manufacturing tolerance (~20 mm) in quadrature.
# Differences much smaller than sigma are essentially indistinguishable.
FIT_SIGMA_MM = 42.0


def load_full_labels(dataset_root, station, ts):
    """Load the raw labels JSON for a garment. Returns None if missing."""
    m = re.match(r"^(\d{4})_(\d{2})", ts)
    if not m:
        return None
    year, mo = m.group(1), int(m.group(2)) - 1
    if not (0 <= mo < 12):
        return None
    folder = f"{MONTHS[mo]}{year}"
    p = Path(dataset_root) / station / folder / f"labels_{ts}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def fit_score(diff_mm, sigma=FIT_SIGMA_MM):
    """0-100 fit score. exp(-diff^2 / 2 sigma^2) mapped to [0, 100]."""
    return 100.0 * float(np.exp(-(diff_mm ** 2) / (2.0 * sigma * sigma)))


def overall_fit_score(per_meas_scores):
    return float(np.mean(list(per_meas_scores.values())))


def fit_interpretation(diffs):
    """Short natural-language sentence describing the diff pattern."""
    named = {
        "waist": "waist", "hip": "hip", "thigh": "thigh",
        "calf": "calf", "ankle": "ankle",
    }
    close = [k for k, v in diffs.items() if abs(v) <= 20]
    off = [(k, v) for k, v in diffs.items() if abs(v) > 40]
    off.sort(key=lambda kv: -abs(kv[1]))

    if not off:
        return ("Essentially the same fit — every landmark is within about "
                "a half-size of yours.")
    parts = []
    if close:
        close_names = ", ".join(named[k] for k in close)
        parts.append(f"Close through {close_names}.")
    for k, v in off[:2]:
        # diff = home - match. v > 0 → match smaller (tighter);
        # v < 0 → match larger (wider). Phrase from the match's POV.
        direction = "wider" if v < 0 else "tighter"
        parts.append(
            f"Match is {abs(v):.0f} mm {direction} at the {named[k]}."
        )
    return " ".join(parts)


def material_interpretation(material_raw):
    """Very rough rule-based reading of the free-text material field."""
    if not material_raw:
        return "Material not on the label — feel is unknown."
    m = material_raw.lower()

    def pct(word):
        pat = re.compile(rf"(\d{{1,3}})\s*%?\s*{word}")
        match = pat.search(m)
        return int(match.group(1)) if match else None

    cotton = pct("cotton")
    poly = pct("polyester")
    elast = pct("elastane") or pct("spandex") or pct("elastan")
    visc = pct("viscose") or pct("rayon")
    wool = pct("wool")

    bits = []
    if elast and elast >= 1:
        bits.append(f"{elast}% elastane — noticeable stretch")
    if poly and poly >= 10:
        bits.append("polyester blend — softer, less structured than pure denim")
    elif cotton and cotton == 100 and not elast:
        bits.append("100% cotton — rigid, no stretch, will feel stiffer")
    elif cotton and cotton >= 95 and not elast:
        bits.append("mostly cotton — traditional denim feel")
    if visc:
        bits.append("viscose — drapey, less structured than denim")
    if wool:
        bits.append("wool content — warmer, more structured")

    if not bits:
        return f"Material: {material_raw}."
    return "  •  ".join(b.capitalize() for b in bits) + "."


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

BG = "#f5f4ef"
CARD = "#ffffff"
BORDER = "#d7d3c4"
INK = "#1c1c1c"
INK_SOFT = "#5a5a5a"
ACCENT = "#3f8f2c"
WARN = "#b26a1a"
BAD = "#a83232"


def score_color(score):
    if score >= 85:
        return ACCENT
    if score >= 60:
        return WARN
    return BAD


def add_card(ax, x, y, w, h, facecolor=CARD, edgecolor=BORDER, lw=1.0):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        transform=ax.transAxes,
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
        zorder=1,
    )
    ax.add_patch(box)


def render_measurement_gauge(fig_ax, x, y, w, h, name, diff, score,
                             home_val, match_val):
    """Draw a per-measurement diff gauge. Uses axes-fraction coords.

    `diff` is home - match. Positive → match is smaller (tighter);
    negative → match is larger (wider). We phrase from the match's
    perspective for the user.
    """
    fig_ax.text(x, y + h * 0.65, name.upper(), fontsize=10,
                fontweight="bold", color=INK, transform=fig_ax.transAxes,
                va="center")
    if abs(diff) < 5:
        rel_text = f"~ same  ({diff:+.0f} mm)"
    elif diff > 0:
        rel_text = f"{diff:.0f} mm tighter"
    else:
        rel_text = f"{-diff:.0f} mm wider"
    fig_ax.text(x + w * 0.22, y + h * 0.65, rel_text, fontsize=10,
                color=INK_SOFT, transform=fig_ax.transAxes, va="center")
    # Bar background
    bar_x0 = x + w * 0.55
    bar_x1 = x + w * 0.92
    bar_y = y + h * 0.55
    bar_h = h * 0.24
    fig_ax.add_patch(Rectangle(
        (bar_x0, bar_y - bar_h / 2), bar_x1 - bar_x0, bar_h,
        transform=fig_ax.transAxes, facecolor="#eee7d6",
        edgecolor="none", zorder=2))
    fill_w = (bar_x1 - bar_x0) * (score / 100.0)
    fig_ax.add_patch(Rectangle(
        (bar_x0, bar_y - bar_h / 2), fill_w, bar_h,
        transform=fig_ax.transAxes, facecolor=score_color(score),
        edgecolor="none", zorder=3))
    fig_ax.text(x + w * 0.96, y + h * 0.65, f"{score:.0f}",
                fontsize=9, color=INK_SOFT, transform=fig_ax.transAxes,
                va="center", ha="right")
    # Sub-line: home vs match value
    fig_ax.text(x, y + h * 0.22,
                f"you {home_val:.0f} mm   ·   match {match_val:.0f} mm",
                fontsize=8, color=INK_SOFT,
                transform=fig_ax.transAxes, va="center")


def wrap_text(text, max_chars):
    """Basic greedy word wrap. matplotlib's wrap=True is unreliable at
    axes-relative coords."""
    words = text.split()
    lines = []
    line = ""
    for w in words:
        candidate = (line + " " + w).strip()
        if len(candidate) <= max_chars:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    return "\n".join(lines)


def format_price(raw):
    return raw if raw else "Price n/a"


def render_thumbnail(fig, gs_cell, image_path, chip_lines):
    ax = fig.add_subplot(gs_cell)
    img = Image.open(image_path).convert("RGB")
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(1)
    ax.set_title("\n".join(chip_lines), fontsize=9, color=INK, pad=6)
    return ax


def build_mockup(image_path, out_path, dataset_root, csv_path,
                 brand_title="Garment Marketplace"):
    # 1. Run the pipeline on the home photo, persist debug output.
    debug_dir = Path(out_path).with_suffix("") / "debug"
    home_ms, result_json = measure_home_photo(image_path, debug_dir)
    query = np.array([home_ms[m] for m in MEASUREMENTS])

    home_panel_path = debug_dir / "07_home_panel.jpg"
    render_home_panel(debug_dir, result_json, home_panel_path)

    # 2. Rank matches.
    rows = load_dataset(csv_path, dataset_root)
    ranked = rank_matches(query, rows)
    if len(ranked) < 1:
        raise RuntimeError("no dataset matches available")

    top = ranked[0]
    dist_top, row_top = top
    labels_top = None
    for st in ("station1", "station2", "station3"):
        labels_top = load_full_labels(dataset_root, st, row_top["ts"])
        if labels_top is not None:
            break
    labels_top = labels_top or {}

    diffs = {m: home_ms[m] - row_top["vec"][i]
             for i, m in enumerate(MEASUREMENTS)}
    per_scores = {m: fit_score(diffs[m]) for m in MEASUREMENTS}
    overall = overall_fit_score(per_scores)
    fit_note = fit_interpretation(diffs)
    material_note = material_interpretation((labels_top or {}).get("material"))

    # 3. Render.
    fig = plt.figure(figsize=(14.5, 10.8), facecolor=BG)
    outer = fig.add_gridspec(
        nrows=3, ncols=3,
        height_ratios=[0.08, 0.7, 0.22],
        # Home photo is portrait, match photo is landscape. Make the match
        # column wider so its landscape image fills the panel rather than
        # sitting small in the middle.
        width_ratios=[0.85, 1.35, 1.2],
        left=0.03, right=0.97, top=0.97, bottom=0.03,
        hspace=0.05, wspace=0.04,
    )

    # Header row (spans full width).
    hax = fig.add_subplot(outer[0, :])
    hax.axis("off")
    hax.set_facecolor(BG)
    hax.text(0.01, 0.5, brand_title,
             fontsize=22, fontweight="bold", color=INK,
             transform=hax.transAxes, va="center")
    hax.text(0.99, 0.5, "match report",
             fontsize=12, color=INK_SOFT,
             transform=hax.transAxes, va="center", ha="right")
    hax.plot([0.01, 0.99], [0.05, 0.05], color=BORDER, lw=1,
             transform=hax.transAxes)

    # Home panel.
    ax_home = fig.add_subplot(outer[1, 0])
    ax_home.imshow(Image.open(home_panel_path).convert("RGB"))
    ax_home.axis("off")
    ax_home.set_title("home garment", fontsize=12, color=INK, pad=8,
                      loc="left", fontweight="bold")

    # Top match photo.
    ax_match = fig.add_subplot(outer[1, 1])
    ax_match.imshow(Image.open(row_top["front"]).convert("RGB"))
    ax_match.axis("off")
    ax_match.set_title("recommended pick", fontsize=12, color=INK, pad=8,
                       loc="left", fontweight="bold")
    # Meta strip under the photo.
    meta_parts = [
        f"{row_top['brand']}  ·  EU {row_top['size']} ({row_top['category']})",
        (row_top['cut'] or "—") + "  ·  "
        + (labels_top or {}).get("material", "material n/a"),
        format_price((labels_top or {}).get("price"))
        + "  ·  condition "
        + str((labels_top or {}).get("condition", "?")),
    ]
    # Use x-axis text below the image (Matplotlib will clip if outside axes).
    ax_match.text(0.5, -0.02, "\n".join(meta_parts),
                  transform=ax_match.transAxes, ha="center", va="top",
                  fontsize=10, color=INK)

    # Fit analysis card.
    ax_fit = fig.add_subplot(outer[1, 2])
    ax_fit.set_facecolor(CARD)
    ax_fit.set_xticks([])
    ax_fit.set_yticks([])
    for spine in ax_fit.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(1)
    ax_fit.set_title("fit analysis", fontsize=12, color=INK, pad=8,
                     loc="left", fontweight="bold")

    # Big score
    ax_fit.text(0.06, 0.92, "Fit similarity",
                fontsize=11, color=INK_SOFT, transform=ax_fit.transAxes,
                va="top")
    ax_fit.text(0.06, 0.88, f"{overall:.0f}",
                fontsize=48, color=score_color(overall),
                fontweight="bold",
                transform=ax_fit.transAxes, va="top")
    ax_fit.text(0.27, 0.83, "/ 100",
                fontsize=14, color=INK_SOFT, transform=ax_fit.transAxes,
                va="top")

    # Per-measurement gauges (5 rows).
    top_y = 0.70
    row_h = 0.09
    for i, m in enumerate(MEASUREMENTS):
        render_measurement_gauge(
            ax_fit,
            x=0.06, y=top_y - (i + 1) * row_h,
            w=0.88, h=row_h,
            name=m,
            diff=diffs[m],
            score=per_scores[m],
            home_val=home_ms[m],
            match_val=row_top["vec"][MEASUREMENTS.index(m)],
        )

    # Interpretation blurb. Manual wrap at ~46 characters so text stays
    # inside the card.
    ax_fit.text(0.06, 0.22, "FIT NOTES",
                fontsize=9, color=INK_SOFT, transform=ax_fit.transAxes,
                fontweight="bold")
    ax_fit.text(0.06, 0.19, wrap_text(fit_note, 46),
                fontsize=10, color=INK, transform=ax_fit.transAxes,
                va="top", linespacing=1.35)
    ax_fit.text(0.06, 0.10, "MATERIAL NOTES",
                fontsize=9, color=INK_SOFT, transform=ax_fit.transAxes,
                fontweight="bold")
    ax_fit.text(0.06, 0.07, wrap_text(material_note, 46),
                fontsize=10, color=INK, transform=ax_fit.transAxes,
                va="top", linespacing=1.35)

    # Bottom strip — alternates.
    bax = fig.add_subplot(outer[2, :])
    bax.axis("off")
    bax.text(0.01, 0.93, "other candidates",
             fontsize=11, color=INK, transform=bax.transAxes,
             fontweight="bold")

    alt_gs = outer[2, :].subgridspec(
        nrows=1, ncols=6, wspace=0.06,
        width_ratios=[0.5, 0.5, 1.5, 0.5, 0.5, 1.5],
    )
    for slot, (dist, row) in enumerate(ranked[1:3]):
        labels = None
        for st in ("station1", "station2", "station3"):
            labels = load_full_labels(dataset_root, st, row["ts"])
            if labels is not None:
                break
        row_diffs = {m: home_ms[m] - row["vec"][i]
                     for i, m in enumerate(MEASUREMENTS)}
        row_scores = {m: fit_score(row_diffs[m]) for m in MEASUREMENTS}
        row_overall = overall_fit_score(row_scores)
        chip_lines = [
            f"{row['brand']}  ·  EU {row['size']} ({row['category']})",
            (labels or {}).get("material", "material n/a"),
            f"fit  {row_overall:.0f} / 100",
        ]
        base_col = slot * 3
        ax_thumb = fig.add_subplot(alt_gs[0, base_col + 1])
        ax_thumb.imshow(Image.open(row["front"]).convert("RGB"))
        ax_thumb.set_xticks([])
        ax_thumb.set_yticks([])
        for spine in ax_thumb.spines.values():
            spine.set_edgecolor(BORDER)
            spine.set_linewidth(1)
        ax_text = fig.add_subplot(alt_gs[0, base_col + 2])
        ax_text.axis("off")
        ax_text.text(0.02, 0.85, chip_lines[0], fontsize=10, color=INK,
                     transform=ax_text.transAxes, fontweight="bold")
        ax_text.text(0.02, 0.55, chip_lines[1], fontsize=9, color=INK_SOFT,
                     transform=ax_text.transAxes)
        ax_text.text(0.02, 0.25, chip_lines[2], fontsize=10,
                     color=score_color(row_overall),
                     transform=ax_text.transAxes, fontweight="bold")

    fig.savefig(out_path, dpi=140, facecolor=BG)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--dataset-root", default=os.environ.get("CIRCULAR_FASHION_ROOT"))
    ap.add_argument("--out", type=Path,
                    default=Path("docs") / "marketplace_mock.jpg")
    args = ap.parse_args()
    if not args.dataset_root:
        ap.error("--dataset-root or CIRCULAR_FASHION_ROOT is required")

    build_mockup(args.image, args.out, args.dataset_root, args.csv)


if __name__ == "__main__":
    main()
