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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

# System sans-serif chain — Helvetica Neue is the primary, Avenir Next
# and Helvetica as fallbacks (all common on macOS), then DejaVu Sans as
# the ultimate matplotlib default.
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Helvetica Neue", "Avenir Next", "Helvetica", "Arial", "DejaVu Sans"
]
mpl.rcParams["axes.unicode_minus"] = False

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


# Wearability rules — each significant per-measurement difference maps to
# a practical fit consequence rather than a restated number. `wide` = the
# match is wider than home (diff < 0); `tight` = the match is tighter
# (diff > 0). The first form is the full sentence used in FIT NOTES; the
# second is a short chip used on alternate cards.
_WEAR_RULES = {
    ("waist", "wide"): (
        "The waist sits looser — a belt may be needed to keep the pants up.",
        "waist sits looser — belt may help",
    ),
    ("waist", "tight"): (
        "The waist runs snugger — may feel restrictive around the middle.",
        "waist runs snugger",
    ),
    ("hip", "wide"): (
        "Roomier through the seat — less form-fitting than yours.",
        "roomier through the seat",
    ),
    ("hip", "tight"): (
        "Closer through the seat — more form-fitting than yours.",
        "closer through the seat",
    ),
    ("thigh", "wide"): (
        "More room through the thigh — reads less shaped.",
        "more room through the thigh",
    ),
    ("thigh", "tight"): (
        "Closer through the thigh — may feel snug when sitting.",
        "closer through the thigh",
    ),
    ("calf", "wide"): (
        "Straighter through the calf — less shaped than yours.",
        "straighter through the calf",
    ),
    ("calf", "tight"): (
        "Closer through the calf — more shaped than yours.",
        "closer through the calf",
    ),
    ("ankle", "wide"): (
        "Less tapered at the ankle — reads more like a straight-leg or "
        "relaxed fit than a skinny.",
        "less tapered ankle — reads relaxed",
    ),
    ("ankle", "tight"): (
        "More tapered at the ankle — reads skinnier than yours.",
        "more tapered ankle — skinnier",
    ),
}


def _significant_diffs(diffs, threshold_mm=30):
    """Return diffs ordered by magnitude, wider ones filtered to be a real
    signal (above `threshold_mm`)."""
    return sorted(
        [(k, v) for k, v in diffs.items() if abs(v) >= threshold_mm],
        key=lambda kv: -abs(kv[1]),
    )


def wearability_note(diffs, threshold_mm=30, max_lines=2):
    """Full-sentence wearability interpretation for FIT NOTES.

    Skips measurements that are close (those are already visible in the
    gauges) and translates the notable ones into practical fit language.
    """
    significant = _significant_diffs(diffs, threshold_mm)
    if not significant:
        return ("Very similar fit overall — should wear much like your "
                "own pair.")
    lines = []
    for name, diff in significant[:max_lines]:
        key = (name, "wide" if diff < 0 else "tight")
        rule = _WEAR_RULES.get(key)
        if rule:
            lines.append(rule[0])
    return " ".join(lines) if lines else "Very similar fit overall."


def wearability_short(diffs, threshold_mm=30):
    """One-line wearability note for the alternate cards."""
    significant = _significant_diffs(diffs, threshold_mm)
    if not significant:
        return "very similar fit overall"
    name, diff = significant[0]
    key = (name, "wide" if diff < 0 else "tight")
    rule = _WEAR_RULES.get(key)
    return rule[1] if rule else "very similar fit overall"


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


def _draw_note_panel(ax, x, y, w, h, title, body, accent):
    """Render a highlighted "notes" panel inside an axes.

    Layout: soft-tinted background rectangle spanning the panel, a 4-unit
    accent bar on the left, a bold uppercase title, then body text at a
    larger size than the card's caption scale so the interpretation
    stands out.
    """
    # Background tint (very subtle so it doesn't overpower)
    ax.add_patch(Rectangle(
        (x, y), w, h,
        transform=ax.transAxes,
        facecolor="#faf7ec", edgecolor="#e5e0cd", linewidth=0.8,
        zorder=2,
    ))
    # Accent bar on the left
    bar_w = w * 0.018
    ax.add_patch(Rectangle(
        (x, y), bar_w, h,
        transform=ax.transAxes,
        facecolor=accent, edgecolor="none",
        zorder=3,
    ))
    # Title
    inner_x = x + w * 0.055
    ax.text(inner_x, y + h * 0.86, title,
            fontsize=9.5, color=accent, fontweight="bold",
            transform=ax.transAxes, va="top")
    # Body
    ax.text(inner_x, y + h * 0.62, body,
            fontsize=11.5, color=INK, transform=ax.transAxes,
            va="top", linespacing=1.5)


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


def tidy_material(raw):
    """Normalise the free-text material label (e.g. "80%cotton, 18%polyester"
    → "80% cotton, 18% polyester"). Also capitalise fibre names."""
    if not raw:
        return ""
    s = str(raw).strip()
    # "80%cotton" → "80% cotton"
    s = re.sub(r"(\d+)\s*%\s*", r"\1% ", s)
    # collapse repeated whitespace
    s = re.sub(r"\s+", " ", s)
    # normalise separator around commas
    s = re.sub(r"\s*,\s*", ", ", s)
    # capitalise known fibre words
    for fibre in ("cotton", "polyester", "elastane", "spandex", "viscose",
                  "rayon", "wool", "linen", "silk", "nylon"):
        s = re.sub(rf"\b{fibre}\b", fibre.capitalize(), s, flags=re.I)
    return s


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
    fit_note = wearability_note(diffs)
    material_note = material_interpretation((labels_top or {}).get("material"))

    # 3. Render.
    fig = plt.figure(figsize=(15.0, 11.5), facecolor=BG)
    outer = fig.add_gridspec(
        nrows=3, ncols=3,
        height_ratios=[0.075, 0.62, 0.305],
        # Home photo is portrait, match photo is landscape. Make the match
        # column wider so its landscape image fills the panel rather than
        # sitting small in the middle.
        width_ratios=[0.85, 1.35, 1.2],
        left=0.03, right=0.97, top=0.97, bottom=0.03,
        hspace=0.06, wspace=0.04,
    )

    # Header row (spans full width).
    hax = fig.add_subplot(outer[0, :])
    hax.axis("off")
    hax.set_facecolor(BG)
    hax.text(0.005, 0.62, brand_title,
             fontsize=26, fontweight="bold", color=INK,
             transform=hax.transAxes, va="center")
    hax.text(0.005, 0.22, "single-photo garment matching",
             fontsize=11, color=INK_SOFT,
             transform=hax.transAxes, va="center")
    hax.text(0.995, 0.5, "MATCH REPORT",
             fontsize=10, color=INK_SOFT,
             transform=hax.transAxes, va="center", ha="right",
             fontweight="bold")
    hax.plot([0.005, 0.995], [0.02, 0.02], color=BORDER, lw=1,
             transform=hax.transAxes)

    # Home panel.
    ax_home = fig.add_subplot(outer[1, 0])
    ax_home.imshow(Image.open(home_panel_path).convert("RGB"))
    ax_home.axis("off")
    ax_home.set_title("HOME GARMENT", fontsize=10, color=INK_SOFT, pad=8,
                      loc="left", fontweight="bold")

    # Top match — split middle column into 4 rows: photo, brand meta,
    # fit-notes panel, material-feel panel. The two interpretation panels
    # sit *with the product* (mirroring how the alternates present
    # image + info together); the score + gauges live in the fit-analysis
    # column to the right.
    match_inner = outer[1, 1].subgridspec(
        nrows=4, ncols=1,
        height_ratios=[3.1, 1.05, 1.05, 1.05],
        hspace=0.05,
    )
    ax_match = fig.add_subplot(match_inner[0, 0])
    ax_match.imshow(Image.open(row_top["front"]).convert("RGB"))
    ax_match.axis("off")
    ax_match.set_title("RECOMMENDED PICK", fontsize=10, color=INK_SOFT,
                       pad=8, loc="left", fontweight="bold")
    ax_meta = fig.add_subplot(match_inner[1, 0])
    ax_meta.axis("off")
    material_top = tidy_material((labels_top or {}).get("material")) or "material n/a"
    price_top = (labels_top or {}).get("price") or "price n/a"
    cond_top = (labels_top or {}).get("condition", "?")
    ax_meta.text(0.0, 0.95, row_top["brand"], fontsize=22, color=INK,
                 transform=ax_meta.transAxes, fontweight="bold", va="top")
    ax_meta.text(0.0, 0.55,
                 f"EU {row_top['size']} · {row_top['category']}"
                 + (f" · {row_top['cut']}" if row_top["cut"] else ""),
                 fontsize=12, color=INK_SOFT,
                 transform=ax_meta.transAxes, va="top")
    ax_meta.text(0.0, 0.30, material_top, fontsize=11, color=INK,
                 transform=ax_meta.transAxes, va="top")
    ax_meta.text(0.0, 0.05, f"{price_top} €  ·  condition {cond_top}",
                 fontsize=10, color=INK_SOFT,
                 transform=ax_meta.transAxes, va="top")

    # Interpretation panels below the meta block (accent-styled).
    ax_notes = fig.add_subplot(match_inner[2, 0])
    ax_notes.axis("off")
    _draw_note_panel(
        ax_notes, x=0.0, y=0.0, w=1.0, h=1.0,
        title="FIT NOTES", body=wrap_text(fit_note, 74),
        accent=score_color(overall),
    )
    ax_matnote = fig.add_subplot(match_inner[3, 0])
    ax_matnote.axis("off")
    _draw_note_panel(
        ax_matnote, x=0.0, y=0.0, w=1.0, h=1.0,
        title="MATERIAL FEEL", body=wrap_text(material_note, 74),
        accent="#4c6a86",
    )

    # Fit analysis card.
    ax_fit = fig.add_subplot(outer[1, 2])
    ax_fit.set_facecolor(CARD)
    ax_fit.set_xticks([])
    ax_fit.set_yticks([])
    for spine in ax_fit.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(1)
    ax_fit.set_title("FIT ANALYSIS", fontsize=10, color=INK_SOFT, pad=8,
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

    # Divider between the headline score and the per-measurement rows.
    ax_fit.plot([0.06, 0.94], [0.56, 0.56], color=BORDER, lw=0.8,
                transform=ax_fit.transAxes, solid_capstyle="butt")

    # Per-measurement gauges (5 rows). With the note panels moved to the
    # recommended-pick column, the right column has more room — grow the
    # gauges and add a small subtitle so it doesn't feel empty.
    ax_fit.text(0.06, 0.51, "PER MEASUREMENT",
                fontsize=9, color=INK_SOFT, fontweight="bold",
                transform=ax_fit.transAxes, va="top")
    top_y = 0.47
    row_h = 0.084
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

    # Bottom strip — alternates.
    bax = fig.add_subplot(outer[2, :])
    bax.axis("off")
    bax.text(0.005, 0.98, "OTHER CANDIDATES", fontsize=10, color=INK_SOFT,
             transform=bax.transAxes, fontweight="bold", va="top")

    # Two side-by-side alternate cards. Each card is a nested 2-column
    # subplot: image (left) + rich info (right).
    alt_gs = outer[2, :].subgridspec(nrows=1, ncols=2, wspace=0.035)
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
        material = tidy_material((labels or {}).get("material")) or "material n/a"
        price = (labels or {}).get("price") or "price n/a"
        cond = (labels or {}).get("condition", "?")

        inner = alt_gs[0, slot].subgridspec(
            nrows=1, ncols=2, width_ratios=[1.05, 1.05], wspace=0.02,
        )
        ax_thumb = fig.add_subplot(inner[0, 0])
        ax_thumb.imshow(Image.open(row["front"]).convert("RGB"))
        ax_thumb.set_xticks([])
        ax_thumb.set_yticks([])
        for spine in ax_thumb.spines.values():
            spine.set_edgecolor(BORDER)
            spine.set_linewidth(1)

        ax_text = fig.add_subplot(inner[0, 1])
        ax_text.set_facecolor(CARD)
        ax_text.set_xticks([])
        ax_text.set_yticks([])
        for spine in ax_text.spines.values():
            spine.set_edgecolor(BORDER)
            spine.set_linewidth(1)

        # Typography stack inside the info card.
        pad_x = 0.08
        ax_text.text(pad_x, 0.90, row["brand"], fontsize=17, color=INK,
                     transform=ax_text.transAxes, fontweight="bold",
                     va="top")
        ax_text.text(pad_x, 0.79,
                     f"EU {row['size']} · {row['category']}"
                     + (f" · {row['cut']}" if row['cut'] else ""),
                     fontsize=11, color=INK_SOFT,
                     transform=ax_text.transAxes, va="top")
        ax_text.text(pad_x, 0.66,
                     wrap_text(material, 34),
                     fontsize=10.5, color=INK,
                     transform=ax_text.transAxes, va="top", linespacing=1.35)
        ax_text.text(pad_x, 0.46,
                     f"{price} €  ·  condition {cond}",
                     fontsize=10, color=INK_SOFT,
                     transform=ax_text.transAxes, va="top")

        # 1-line fit note so users can decide between alternates without
        # clicking through. Coloured with the fit-score palette so it ties
        # visually to the score below.
        row_note = wearability_short(row_diffs)
        ax_text.text(pad_x, 0.34,
                     wrap_text(row_note, 46),
                     fontsize=10, color=score_color(row_overall),
                     transform=ax_text.transAxes, va="top",
                     linespacing=1.35, fontstyle="italic")

        # Fit similarity number, prominent
        ax_text.text(pad_x, 0.19, "FIT SIMILARITY", fontsize=8.5,
                     color=INK_SOFT, transform=ax_text.transAxes,
                     fontweight="bold", va="top")
        ax_text.text(pad_x, 0.13, f"{row_overall:.0f}", fontsize=32,
                     color=score_color(row_overall),
                     transform=ax_text.transAxes, fontweight="bold",
                     va="top")
        ax_text.text(pad_x + 0.15, 0.02, "/ 100", fontsize=12,
                     color=INK_SOFT, transform=ax_text.transAxes,
                     va="top")

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
