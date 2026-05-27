#!/usr/bin/env python3
"""MVP: extract garment measurements from a top-down photo of jeans with an A4 reference.

Pipeline:
  1. Load image (honor EXIF orientation).
  2. Estimate background color from image borders (HSV median).
  3. Foreground mask via HSV distance from background.
  4. Detect A4 paper (bright + desaturated foreground blob) and its 4 corners.
  5. Compute scale uncertainty from variability of A4 side lengths.
  6. Rectify (homography) to top-down using A4 as reference -> known px/mm.
  7. Isolate jeans mask = foreground minus A4, keep largest connected component.
  8. PCA on jeans mask -> rotate so principal axis is vertical.
  9. Flip 180° if waistband ended up at the bottom (detected via leg-split rows).
 10. Detect crotch point geometrically (scan rows bottom->top, find 2-run -> 1-run boundary).
 11. Sample widths at predefined ratios (yoke and per-leg), with ±band averaging
     and ±ratio sensitivity, double to get circumference.
 12. Report each measurement with combined uncertainty (edge + ratio + scale).
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# Default scale: 4 pixels per mm in the rectified frame. Enough for sub-mm
# discretization without blowing up image size; cm-level accuracy is the target.
PX_PER_MM = 4.0

A4_SHORT_MM = 210.0
A4_LONG_MM = 297.0

# Measurement positions (ratios). Yoke: 0 = waistband, 1 = crotch.
# Leg: 0 = crotch, 1 = cuff.
YOKE_RATIOS = {"waist": 0.0, "hip": 0.65}
LEG_RATIOS = {"thigh": 0.05, "calf": 0.70, "ankle": 1.0}

# Width sampling: average over y ± WIDTH_BAND rows.
WIDTH_BAND = 3
# Ratio sensitivity: re-measure at ratio ± RATIO_DELTA.
RATIO_DELTA = 0.02
# Ignore foreground runs shorter than this (noise).
MIN_RUN_PX = 8

# HSV thresholds for foreground (distance from estimated background).
H_TOL = 14
S_TOL = 40
V_TOL = 50
BORDER_FRAC = 0.04

# A4 candidate pixels: bright + desaturated.
A4_S_MAX = 70
A4_V_MIN = 170


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def estimate_background_hsv(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    bh = max(1, int(h * BORDER_FRAC))
    bw = max(1, int(w * BORDER_FRAC))
    border = np.concatenate([
        hsv[:bh].reshape(-1, 3),
        hsv[-bh:].reshape(-1, 3),
        hsv[:, :bw].reshape(-1, 3),
        hsv[:, -bw:].reshape(-1, 3),
    ])
    return np.median(border, axis=0).astype(np.float32)


def foreground_mask(img_bgr, bg_hsv):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    dh = np.abs(hsv[..., 0] - bg_hsv[0])
    dh = np.minimum(dh, 180 - dh)
    ds = np.abs(hsv[..., 1] - bg_hsv[1])
    dv = np.abs(hsv[..., 2] - bg_hsv[2])
    fg = (dh > H_TOL) | (ds > S_TOL) | (dv > V_TOL)
    fg = fg.astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    return fg


def detect_a4(img_bgr, fg_mask):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bright = (hsv[..., 1] < A4_S_MAX) & (hsv[..., 2] > A4_V_MIN)
    cand = (bright & (fg_mask > 0)).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 1000:
        return None, None
    rect = cv2.minAreaRect(c)
    corners = cv2.boxPoints(rect).astype(np.float32)
    mask = np.zeros_like(fg_mask)
    cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)
    return corners, mask


def isolate_jeans(fg_mask, a4_mask):
    j = ((fg_mask > 0) & (a4_mask == 0)).astype(np.uint8) * 255
    j = cv2.morphologyEx(j, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    j = cv2.morphologyEx(j, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    num, lab, stats, _ = cv2.connectedComponentsWithStats(j, connectivity=8)
    if num < 2:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    return (lab == idx).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def order_corners(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def scale_uncertainty(corners):
    """Fractional disagreement among the four sides' implied px/mm.

    Captures perspective residual + corner-localization noise. If the photo
    were perfectly top-down with sub-pixel corner detection, all four sides
    would imply the same px/mm.
    """
    ordered = order_corners(corners)
    tl, tr, br, bl = ordered
    sides = sorted([
        float(np.linalg.norm(tr - tl)),
        float(np.linalg.norm(br - tr)),
        float(np.linalg.norm(bl - br)),
        float(np.linalg.norm(tl - bl)),
    ])
    shorts, longs = sides[:2], sides[2:]
    est = [s / A4_SHORT_MM for s in shorts] + [s / A4_LONG_MM for s in longs]
    return float(np.std(est) / np.mean(est))


def rectify(img_bgr, masks, a4_corners, px_per_mm=PX_PER_MM):
    ordered = order_corners(a4_corners)
    tl, tr, br, bl = ordered
    top_len = np.linalg.norm(tr - tl)
    left_len = np.linalg.norm(bl - tl)
    if top_len >= left_len:
        w_mm, h_mm = A4_LONG_MM, A4_SHORT_MM
    else:
        w_mm, h_mm = A4_SHORT_MM, A4_LONG_MM
    dst = np.array([
        [0, 0],
        [w_mm * px_per_mm, 0],
        [w_mm * px_per_mm, h_mm * px_per_mm],
        [0, h_mm * px_per_mm],
    ], dtype=np.float32)
    H = cv2.getPerspectiveTransform(ordered, dst)
    h, w = img_bgr.shape[:2]
    img_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(img_corners, H).reshape(-1, 2)
    minxy = warped.min(axis=0)
    maxxy = warped.max(axis=0)
    T = np.array([[1, 0, -minxy[0]], [0, 1, -minxy[1]], [0, 0, 1]], dtype=np.float32)
    H_full = T @ H
    out_w = int(np.ceil(maxxy[0] - minxy[0]))
    out_h = int(np.ceil(maxxy[1] - minxy[1]))
    img_r = cv2.warpPerspective(img_bgr, H_full, (out_w, out_h))
    masks_r = {
        k: cv2.warpPerspective(m, H_full, (out_w, out_h), flags=cv2.INTER_NEAREST)
        for k, m in masks.items()
    }
    return img_r, masks_r


def rotate_to_vertical(img, mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 50:
        return img, mask, 0.0
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    _, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, -1]  # principal axis
    theta_deg = np.degrees(np.arctan2(v[1], v[0]))
    rot_deg = 90.0 - theta_deg
    while rot_deg > 90:
        rot_deg -= 180
    while rot_deg < -90:
        rot_deg += 180

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((float(mean[0]), float(mean[1])), rot_deg, 1.0)
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
    new = M @ corners
    minxy = new.min(axis=1)
    maxxy = new.max(axis=1)
    M[0, 2] -= minxy[0]
    M[1, 2] -= minxy[1]
    out_w = int(np.ceil(maxxy[0] - minxy[0]))
    out_h = int(np.ceil(maxxy[1] - minxy[1]))
    img_r = cv2.warpAffine(img, M, (out_w, out_h))
    mask_r = cv2.warpAffine(mask, M, (out_w, out_h), flags=cv2.INTER_NEAREST)
    return img_r, mask_r, rot_deg


# ---------------------------------------------------------------------------
# Row analysis
# ---------------------------------------------------------------------------

def runs_in_row(row, min_run=MIN_RUN_PX):
    """Return (start, end) for non-zero runs of length >= min_run."""
    out = []
    in_run = False
    start = 0
    for i, v in enumerate(row):
        if v > 0 and not in_run:
            in_run = True
            start = i
        elif v == 0 and in_run:
            if i - start >= min_run:
                out.append((start, i))
            in_run = False
    if in_run and len(row) - start >= min_run:
        out.append((start, len(row)))
    return out


def count_runs(row):
    return len(runs_in_row(row))


def ensure_upright(img, mask):
    """If rows with 2+ runs cluster in the top half, jeans are upside-down → flip 180°."""
    h = mask.shape[0]
    splits = [y for y in range(h) if count_runs(mask[y]) >= 2]
    if not splits:
        return img, mask, False
    mid = h / 2.0
    above = sum(1 for y in splits if y < mid)
    below = sum(1 for y in splits if y >= mid)
    if above > below:
        return cv2.rotate(img, cv2.ROTATE_180), cv2.rotate(mask, cv2.ROTATE_180), True
    return img, mask, False


def vertical_extent(mask):
    ys = np.where(mask.sum(axis=1) > 0)[0]
    return int(ys[0]), int(ys[-1])


def find_crotch(mask, confirm=5):
    """Scan rows bottom→top. The crotch is where the two-leg split closes into one run.

    Returns the y just below the last consistently 2-run region.
    """
    h = mask.shape[0]
    streak = 0
    last_split_y = None
    for y in range(h - 1, -1, -1):
        row = mask[y]
        if row.sum() == 0:
            streak = 0
            continue
        n = count_runs(row)
        if n >= 2:
            streak = 0
            last_split_y = y
        else:
            streak += 1
            if last_split_y is not None and streak >= confirm:
                return y + confirm
    return None


def measure_width_at(mask, y, leg=None, half_band=WIDTH_BAND):
    h = mask.shape[0]
    y0 = max(0, y - half_band)
    y1 = min(h, y + half_band + 1)
    widths = []
    for yy in range(y0, y1):
        runs = runs_in_row(mask[yy])
        if not runs:
            continue
        if leg is None:
            widths.append(runs[-1][1] - runs[0][0])
        else:
            if len(runs) < 2:
                continue
            sel = runs[0] if leg == "left" else runs[-1]
            widths.append(sel[1] - sel[0])
    if not widths:
        return None
    return float(np.mean(widths)), float(np.std(widths))


# ---------------------------------------------------------------------------
# Measurement assembly
# ---------------------------------------------------------------------------

def build_measurement(mask, y0, region_h, ratio, leg, px_per_mm, scale_unc):
    y = int(round(y0 + ratio * region_h))
    base = measure_width_at(mask, y, leg=leg)
    if base is None:
        return None
    mean_px, std_px = base

    y_lo = int(round(y0 + max(0.0, ratio - RATIO_DELTA) * region_h))
    y_hi = int(round(y0 + min(1.0, ratio + RATIO_DELTA) * region_h))
    w_lo = measure_width_at(mask, y_lo, leg=leg)
    w_hi = measure_width_at(mask, y_hi, leg=leg)
    if w_lo is not None and w_hi is not None:
        ratio_unc_px = abs(w_hi[0] - w_lo[0]) / 2.0
    else:
        ratio_unc_px = 0.0

    width_mm = mean_px / px_per_mm
    edge_unc_mm = std_px / px_per_mm
    ratio_unc_mm = ratio_unc_px / px_per_mm
    circ_mm = 2.0 * width_mm
    circ_edge = 2.0 * edge_unc_mm
    circ_ratio = 2.0 * ratio_unc_mm
    circ_scale = circ_mm * scale_unc
    circ_total = float(np.sqrt(circ_edge ** 2 + circ_ratio ** 2 + circ_scale ** 2))

    return {
        "y_px": y,
        "ratio": ratio,
        "width_mm": round(width_mm, 1),
        "circumference_mm": round(circ_mm, 1),
        "uncertainty_mm": round(circ_total, 1),
        "components_mm": {
            "edge": round(circ_edge, 2),
            "ratio": round(circ_ratio, 2),
            "scale": round(circ_scale, 2),
        },
    }


def measure_jeans(image_path, px_per_mm=PX_PER_MM, debug_dir=None):
    img = load_image(image_path)
    bg = estimate_background_hsv(img)
    fg = foreground_mask(img, bg)
    a4_corners, a4_mask = detect_a4(img, fg)
    if a4_corners is None:
        raise RuntimeError("A4 paper not detected. Check background contrast and protocol.")
    jeans = isolate_jeans(fg, a4_mask)
    if jeans is None:
        raise RuntimeError("Jeans not detected in foreground.")

    scale_unc = scale_uncertainty(a4_corners)

    img_rect, masks_rect = rectify(img, {"jeans": jeans, "a4": a4_mask}, a4_corners, px_per_mm)
    jeans_rect = masks_rect["jeans"]

    img_rot, mask_rot, _ = rotate_to_vertical(img_rect, jeans_rect)
    img_rot, mask_rot, flipped = ensure_upright(img_rot, mask_rot)

    y_top, y_bottom = vertical_extent(mask_rot)
    y_crotch = find_crotch(mask_rot)
    if y_crotch is None or y_crotch <= y_top or y_crotch >= y_bottom:
        raise RuntimeError("Crotch point detection failed.")

    yoke_h = y_crotch - y_top
    leg_h = y_bottom - y_crotch

    results = {}
    for name, ratio in YOKE_RATIOS.items():
        results[name] = build_measurement(
            mask_rot, y_top, yoke_h, ratio, leg=None,
            px_per_mm=px_per_mm, scale_unc=scale_unc,
        )
    for name, ratio in LEG_RATIOS.items():
        results[name] = build_measurement(
            mask_rot, y_crotch, leg_h, ratio, leg="left",
            px_per_mm=px_per_mm, scale_unc=scale_unc,
        )

    if debug_dir is not None:
        save_debug(debug_dir, img, fg, a4_mask, jeans, img_rot, mask_rot, results, y_top, y_crotch, y_bottom)

    return {
        "image": str(image_path),
        "px_per_mm": px_per_mm,
        "scale_uncertainty_rel": scale_unc,
        "y_top": y_top,
        "y_crotch": y_crotch,
        "y_bottom": y_bottom,
        "flipped_180": flipped,
        "measurements": results,
    }


# ---------------------------------------------------------------------------
# Debug output
# ---------------------------------------------------------------------------

def save_debug(debug_dir, img_orig, fg_mask, a4_mask, jeans_mask,
               img_rot, mask_rot, results, y_top, y_crotch, y_bottom):
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(debug_dir / "01_input.jpg"), img_orig)
    cv2.imwrite(str(debug_dir / "02_foreground.png"), fg_mask)
    cv2.imwrite(str(debug_dir / "03_a4_mask.png"), a4_mask)
    cv2.imwrite(str(debug_dir / "04_jeans_mask.png"), jeans_mask)
    cv2.imwrite(str(debug_dir / "05_jeans_rotated.png"), mask_rot)

    out = img_rot.copy()
    edges = cv2.Canny(mask_rot, 50, 150)
    out[edges > 0] = (0, 255, 255)
    h, w = out.shape[:2]
    cv2.line(out, (0, y_top), (w, y_top), (0, 255, 0), 1)
    cv2.line(out, (0, y_crotch), (w, y_crotch), (0, 165, 255), 2)
    cv2.line(out, (0, y_bottom), (w, y_bottom), (0, 255, 0), 1)
    for name, m in results.items():
        if m is None:
            continue
        y = m["y_px"]
        cv2.line(out, (0, y), (w, y), (255, 0, 0), 1)
        label = f"{name}: {m['circumference_mm']:.0f}+-{m['uncertainty_mm']:.0f} mm"
        org = (10, max(15, y - 6))
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(str(debug_dir / "06_annotated.jpg"), out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Measure jeans from a top-down photo with an A4 reference.")
    p.add_argument("image", type=Path, help="Input image path (jpg/png/heic).")
    p.add_argument("--debug-dir", type=Path, default=None, help="Write intermediate images here.")
    p.add_argument("--px-per-mm", type=float, default=PX_PER_MM, help="Rectified resolution (default 4).")
    p.add_argument("--json", type=Path, default=None, help="Write the full result as JSON here.")
    args = p.parse_args()

    result = measure_jeans(args.image, px_per_mm=args.px_per_mm, debug_dir=args.debug_dir)

    print(f"Image: {result['image']}")
    print(f"Scale uncertainty: {result['scale_uncertainty_rel'] * 100:.2f}%")
    print(f"Flipped 180 deg: {result['flipped_180']}")
    print()
    print(f"{'Measurement':<10}  {'Circumference':<16}  {'± unc':<8}")
    print("-" * 40)
    for name in ["waist", "hip", "thigh", "calf", "ankle"]:
        m = result["measurements"].get(name)
        if m is None:
            print(f"{name:<10}  (failed)")
        else:
            print(f"{name:<10}  {m['circumference_mm']:>8.1f} mm     ± {m['uncertainty_mm']:>4.1f} mm")

    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
        print(f"\nWrote JSON: {args.json}")


if __name__ == "__main__":
    sys.exit(main() or 0)
