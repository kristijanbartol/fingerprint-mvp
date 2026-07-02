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

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = Path(__file__).resolve().parent / "full_survey.csv"

MEASUREMENTS = ["waist", "hip", "thigh", "calf", "ankle"]


def measure_home_photo(image_path, debug_dir, manual_px_per_mm=None):
    """Run the measurement pipeline on `image_path`, persist debug output
    under `debug_dir`, and return (measurements_dict, result_json_path).
    """
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    json_out = debug_dir / "r.json"
    cmd = [sys.executable, str(REPO_ROOT / "measure_jeans.py"),
           str(image_path), "--no-click", "--rembg",
           "--debug-dir", str(debug_dir),
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
    return out, json_out


def _find_a4_in_rotated(img_bgr):
    """Bounding box of the A4 sheet in the rotated color image (bright,
    low-saturation, A4-ish aspect). Returns (x, y, w, h) or None.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    cand = ((hsv[..., 1] < 40) & (hsv[..., 2] > 200)).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    num, _, stats, _ = cv2.connectedComponentsWithStats(cand)
    if num < 2:
        return None
    best = None
    best_score = 0.0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 5000:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        aspect_score = max(0.0, 1.0 - abs(aspect - 297.0 / 210.0) / 0.4)
        score = area * aspect_score
        if score > best_score:
            best_score = score
            best = (int(stats[i, cv2.CC_STAT_LEFT]),
                    int(stats[i, cv2.CC_STAT_TOP]), w, h)
    return best


def render_home_panel(debug_dir, result_json_path, out_path):
    """Compose the annotated home panel on the *rectified* (un-rotated) image.

    The measurement rows come from the rotated frame, so we invert the
    rotation transform (and the 180° flip if applied) to draw the lines
    on the rectified frame. Labels stay horizontal for readability. A4 is
    detected as the largest bright-white blob in the rectified image.
    """
    debug_dir = Path(debug_dir)
    rect_src = debug_dir / "05a_rectified_color.jpg"
    meta_path = debug_dir / "transform_meta.json"
    if rect_src.exists() and meta_path.exists():
        img = cv2.imread(str(rect_src))
        meta = json.loads(meta_path.read_text())
        rot_M = np.array(meta["rot_M"], dtype=np.float64)
        flipped = bool(meta["flipped_180"])
        h_rot, w_rot = meta["img_rot_shape"]
        use_rectified = True
    else:
        # Fall back to the rotated canvas if the rectified extras aren't there.
        img = cv2.imread(str(debug_dir / "05b_rotated_color.jpg"))
        rot_M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        flipped = False
        h_rot, w_rot = img.shape[:2] if img is not None else (0, 0)
        use_rectified = False
    if img is None:
        raise RuntimeError("no home canvas found")
    h, w = img.shape[:2]
    d = json.loads(Path(result_json_path).read_text())
    ms = d.get("measurements", {})

    # Inverse of the 2x3 affine rot_M: given a point in the rotated frame,
    # return the corresponding point in the rectified frame.
    A = rot_M[:2, :2]
    t = rot_M[:2, 2]
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.eye(2)

    def rot_to_rect(x_rot, y_rot):
        p = A_inv @ (np.array([x_rot, y_rot]) - t)
        return int(round(p[0])), int(round(p[1]))

    # Measurement lines. Distinct BGR colours per landmark so lines and
    # labels are visually tied together.
    line_colors = {
        "waist": (60, 120, 220),   # red-orange
        "hip":   (60, 200, 240),   # yellow-gold
        "thigh": (80, 200, 100),   # green
        "calf":  (220, 140, 60),   # blue
        "ankle": (200, 60, 200),   # magenta
    }
    font = cv2.FONT_HERSHEY_SIMPLEX
    # The composited image scales the home panel down aggressively, so
    # scale text + line thickness to the source image so overlays survive
    # the downsample. Reference ~1500 px wide source → scale ~2.5.
    label_scale = max(1.5, min(3.5, w / 600.0))
    label_thick = max(3, int(round(w / 400.0)))
    line_thick = max(5, int(round(w / 250.0)))

    for name in MEASUREMENTS:
        m = ms.get(name)
        if m is None:
            continue
        y_final = int(m["y_px"])
        # Undo the 180° flip (if ensure_upright applied one) to get y in
        # the pre-flip rotated frame.
        y_rot = (h_rot - 1 - y_final) if flipped else y_final
        color = line_colors.get(name, (0, 255, 255))

        if use_rectified:
            # The measurement line in the rotated frame spans the full row
            # y_rot. In the rectified frame it becomes slanted (perpendicular
            # to the tilted garment axis). For the demo we want everything
            # straight, so map the *centre* of that line to the rectified
            # frame and draw a horizontal line at that y across the image.
            _, y_line = rot_to_rect(w_rot / 2.0, y_rot)
            y_line = int(max(0, min(h - 1, y_line)))
            cv2.line(img, (0, y_line), (w, y_line), color, line_thick)
        else:
            y_line = y_rot
            cv2.line(img, (0, y_line), (w, y_line), color, line_thick)

        label = f"{name.upper()}  {m['circumference_mm']:.0f} mm"
        (tw, th), _ = cv2.getTextSize(label, font, label_scale, label_thick)
        # Right-align the label; sits just above the line unless there's no
        # room, in which case fall back to just below.
        lx = max(6, w - tw - 16)
        ly = y_line - 12 if y_line > th + 20 else y_line + th + 20
        ly = max(th + 8, min(h - 8, ly))
        cv2.putText(img, label, (lx, ly), font, label_scale, (0, 0, 0),
                    label_thick + 6)
        cv2.putText(img, label, (lx, ly), font, label_scale, color,
                    label_thick)

    a4_box = _find_a4_in_rotated(img)
    if a4_box is not None:
        x, y, w_a4, h_a4 = a4_box
        a4_color = (100, 255, 100)  # bright green
        cv2.rectangle(img, (x, y), (x + w_a4, y + h_a4), a4_color, line_thick)
        label = "A4  210 x 297 mm"
        a4_scale = max(1.2, label_scale * 0.75)
        (tw, th), _ = cv2.getTextSize(label, font, a4_scale, label_thick)
        # Prefer placing the label below the A4; fall back to above.
        lx = max(6, min(w - tw - 6, x))
        ly = (y + h_a4 + th + 12) if y + h_a4 + th + 20 < h else max(th + 6, y - 12)
        cv2.putText(img, label, (lx, ly), font, a4_scale, (0, 0, 0),
                    label_thick + 6)
        cv2.putText(img, label, (lx, ly), font, a4_scale, a4_color,
                    label_thick)

    cv2.imwrite(str(out_path), img)
    return out_path


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


def build_comparison_image(home_display_path, top_rows, out_path):
    """Save a 1 x (K+1) grid: annotated home panel | top-K dataset photos.

    The home panel already carries A4 highlight + labelled measurement
    lines, so no extra caption is needed on it.
    """
    k = len(top_rows)
    fig, axes = plt.subplots(1, k + 1, figsize=(3.4 * (k + 1), 5.8))
    if k == 0:
        axes = [axes]
    home_img = Image.open(home_display_path).convert("RGB")
    axes[0].imshow(home_img)
    axes[0].axis("off")
    axes[0].set_title("your home photo", fontsize=10)

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
    debug_dir = args.out.with_suffix("") / "debug"
    home_ms, result_json = measure_home_photo(
        args.image, debug_dir, manual_px_per_mm=args.manual_px_per_mm,
    )
    query = np.array([home_ms[m] for m in MEASUREMENTS])
    print("Home measurements (mm):")
    for m, v in home_ms.items():
        print(f"  {m:<6} {v:7.1f}")
    print()

    home_panel_path = debug_dir / "07_home_panel.jpg"
    render_home_panel(debug_dir, result_json, home_panel_path)

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

    build_comparison_image(home_panel_path, top, args.out)


if __name__ == "__main__":
    main()
