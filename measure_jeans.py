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
# Waist at 0.05 rather than 0.0: ratio 0.0 lands on the top edge of the
# waistband band where the mask narrows from corner stitching/curvature;
# 0.05 of yoke_h centres on the band proper.
YOKE_RATIOS = {"waist": 0.05, "hip": 0.65}
# Ankle at 0.95 rather than 1.0: ratio 1.0 lands on the cuff hem's bottom
# edge where the mask tapers to a narrow row; 0.95 of leg_h centres on the
# cuff opening proper. Symmetric to the waist=0.05 offset.
LEG_RATIOS = {"thigh": 0.05, "calf": 0.70, "ankle": 0.95}

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


def foreground_mask_tile_bg(img_bgr):
    """Segment a garment against the Circular-Fashion tile-grid background.

    Background = bright white tiles. The grout lines between tiles are thin
    (~5 px at 1280×720) and dark — we cannot mark "dark = background" because
    that eats black denim. Instead we mark only white-ish pixels as background
    and rely on a generous morphological open to erase the thin grout pattern.
    Everything else (jeans body + any non-grid clutter like cables, devices)
    becomes foreground; the largest connected blob is the garment.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    bg_white = (s < 50) & (v > 160)
    fg = (~bg_white).astype(np.uint8) * 255
    # 13×13 ellipse: wide enough to break the ~5 px grout lattice, narrow
    # enough to preserve narrow jeans cuffs (typically 30-50 px wide).
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num < 2:
        return fg
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == idx).astype(np.uint8) * 255


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
    """Pick the foreground blob that best matches an A4 sheet.

    Scoring combines:
      - solidity (contour area / min-area-rect area): a sheet fills its bbox.
      - aspect-ratio match to 297/210 = 1.414.
    Picking by size alone fails when a same-colored garment dominates the scene,
    so we explicitly reward rectangle-ness.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bright = (hsv[..., 1] < A4_S_MAX) & (hsv[..., 2] > A4_V_MIN)
    cand = (bright & (fg_mask > 0)).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    a4_aspect = A4_LONG_MM / A4_SHORT_MM  # 1.4143
    min_area = 0.001 * fg_mask.shape[0] * fg_mask.shape[1]
    best_score = -1.0
    best_c = None
    best_rect = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        if w <= 0 or h <= 0:
            continue
        rect_area = w * h
        solidity = area / rect_area
        aspect = max(w, h) / min(w, h)
        aspect_score = max(0.0, 1.0 - abs(aspect - a4_aspect) / 0.5)
        score = solidity * aspect_score
        if score > best_score:
            best_score = score
            best_c = c
            best_rect = rect

    if best_c is None or best_score < 0.4:
        return None, None
    corners = cv2.boxPoints(best_rect).astype(np.float32)
    mask = np.zeros_like(fg_mask)
    cv2.drawContours(mask, [best_c], -1, 255, cv2.FILLED)
    return corners, mask


def isolate_jeans(img_bgr, fg_mask, a4_mask, click_point=None, max_dim=900, iterations=4):
    """Segment the jeans via GrabCut, seeded from the rough color mask and A4.

    Color thresholding alone fails when the garment and the surface are close
    in HSV (e.g. beige jeans on a wood floor). GrabCut models foreground and
    background as Gaussian mixtures and uses spatial smoothness, so it can
    pull the jeans/floor boundary along weak local edges.

    Seeding:
      - Image border (3% band) + dilated A4 region: definite background.
      - If `click_point` is provided: the rough-foreground blob containing the
        click is used as the foreground seed (eroded); all OTHER rough-fg blobs
        become definite background. This is the most reliable seeding: one
        click disambiguates which blob is the garment.
      - Otherwise (auto): heavily eroded core of the largest (rough_fg − A4)
        blob is the foreground seed.
      - Everything else: probable foreground.

    Downsampled for speed; result upsampled back to the original resolution.
    """
    H, W = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(H, W))
    new_w = max(1, int(W * scale))
    new_h = max(1, int(H * scale))
    img_s = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    a4_s = cv2.resize(a4_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    fg_s = cv2.resize(fg_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    a4_dilated = cv2.dilate(a4_s, np.ones((11, 11), np.uint8))
    b = max(1, int(min(new_h, new_w) * 0.03))

    if click_point is not None:
        # Click mode: tight ROI around the click. Everything outside a generous
        # disk is definite background. This pins GrabCut to the garment region.
        gc = np.full((new_h, new_w), cv2.GC_BGD, dtype=np.uint8)
        cx_s = int(round(click_point[0] * scale))
        cy_s = int(round(click_point[1] * scale))
        search_r = int(min(new_h, new_w) * 0.55)
        cv2.circle(gc, (cx_s, cy_s), search_r, int(cv2.GC_PR_FGD), -1)
        # Re-apply background constraints (A4 + border) inside the ROI.
        gc[a4_dilated > 0] = cv2.GC_BGD
        gc[:b, :] = cv2.GC_BGD
        gc[-b:, :] = cv2.GC_BGD
        gc[:, :b] = cv2.GC_BGD
        gc[:, -b:] = cv2.GC_BGD
        # Definite foreground: small disk centered on the click.
        fg_r = max(8, int(min(new_h, new_w) * 0.04))
        cv2.circle(gc, (cx_s, cy_s), fg_r, int(cv2.GC_FGD), -1)
    else:
        # Auto mode: start with PR_FGD everywhere, use rough color mask for hints.
        gc = np.full((new_h, new_w), cv2.GC_PR_FGD, dtype=np.uint8)
        gc[:b, :] = cv2.GC_BGD
        gc[-b:, :] = cv2.GC_BGD
        gc[:, :b] = cv2.GC_BGD
        gc[:, -b:] = cv2.GC_BGD
        gc[a4_dilated > 0] = cv2.GC_BGD
        fg_minus_a4 = ((fg_s > 0) & (a4_dilated == 0)).astype(np.uint8) * 255
        num, lab, stats, _ = cv2.connectedComponentsWithStats(fg_minus_a4, connectivity=8)
        if num >= 2:
            chosen_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            big = (lab == chosen_idx).astype(np.uint8) * 255
            eroded = cv2.erode(big, np.ones((25, 25), np.uint8), iterations=2)
            gc[eroded > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_s, gc, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)

    mask_s = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    # Upsample first, then do morphology at full resolution. Doing OPEN at the
    # downsampled scale would close the gap between the two legs (only a few
    # pixels wide once downsampled). At full res, the leg gap is much wider
    # than the kernel, so the legs stay separate.
    mask_full = cv2.resize(mask_s, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_full = cv2.morphologyEx(mask_full, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask_full, connectivity=8)
    if num < 2:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
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


def vertical_extent(mask, width_frac=0.5):
    """First/last row whose width is at least `width_frac` of peak width.

    The naive first/last-non-zero-row picks tilted corners of the waistband
    and cuff — a single pixel poking up or down from a sub-pixel-tilted mask
    is enough to collapse waist/ankle to a few millimeters. Requiring at
    least half the peak width skips those corner pixels and lands on the
    actual flat waistband / cuff region.
    """
    widths = (mask > 0).sum(axis=1)
    if widths.max() == 0:
        return 0, mask.shape[0] - 1
    threshold = widths.max() * width_frac
    ys = np.where(widths >= threshold)[0]
    if len(ys) == 0:
        ys = np.where(widths > 0)[0]
    return int(ys[0]), int(ys[-1])


def leg_bottoms(mask, y_crotch):
    """Last row (per leg) where the leg's run has length >= MIN_RUN_PX.

    The anatomical y_bottom (50% of peak width) lands inside the spread-leg
    region for flared/bootcut jeans — peak width is the spread itself, the
    cuffs are much narrower than the peak. Per-leg scanning gives the actual
    cuff row even when the legs splay outward.
    """
    h = mask.shape[0]
    y_left = y_crotch
    y_right = y_crotch
    for y in range(y_crotch + 1, h):
        runs = runs_in_row(mask[y])
        if not runs:
            continue
        if len(runs) >= 2:
            left, right = runs[0], runs[-1]
            if left[1] - left[0] >= MIN_RUN_PX:
                y_left = y
            if right[1] - right[0] >= MIN_RUN_PX:
                y_right = y
        else:
            # Single-run row (touching legs near the cuff): credit both.
            if runs[0][1] - runs[0][0] >= MIN_RUN_PX:
                y_left = y
                y_right = y
    return y_left, y_right


def find_crotch(mask, y_top, y_bottom, confirm=5):
    """Locate the crotch row, restricted to [y_top, y_bottom].

    Primary: scan bottom→top within the anatomical extent and return the y
    just below the last consistently 2-run region. Works on jeans laid with
    a visible inter-leg gap.

    Fallback (touching-leg jeans, no gap in the mask): use the row-width
    profile. Width climbs from the waistband, peaks at the hip, then narrows
    at the inseam — that minimum is the crotch. Below the crotch the silhouette
    may stay similar or widen (bootcut/spread). Restrict the search to between
    the hip peak and 85% of the anatomical span so we don't pick the cuff.
    """
    streak = 0
    last_split_y = None
    for y in range(y_bottom, y_top - 1, -1):
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

    widths = (mask > 0).sum(axis=1)
    span = y_bottom - y_top
    if span < 50:
        return None
    upper = y_top + int(span * 0.6)
    peak_y = y_top + int(np.argmax(widths[y_top:upper + 1]))
    lower = y_top + int(span * 0.85)
    if lower - peak_y < 20:
        return None
    return peak_y + int(np.argmin(widths[peak_y:lower + 1]))


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
            if len(runs) >= 2:
                sel = runs[0] if leg == "left" else runs[-1]
                widths.append(sel[1] - sel[0])
            else:
                # Touching legs: no inter-leg gap visible in this row. Approximate
                # the single-leg width as half the full row span (symmetric jeans).
                widths.append((runs[-1][1] - runs[0][0]) / 2.0)
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


def _segment_with_rembg(img_bgr):
    """Run rembg (U²-Net) and return a binary mask (uint8 0/255) of the garment.

    Imported lazily so the rest of the pipeline doesn't require the heavyweight
    onnxruntime dependency.
    """
    from rembg import remove, new_session  # noqa: WPS433  (lazy import)
    session = _segment_with_rembg._session
    if session is None:
        session = new_session("u2net")
        _segment_with_rembg._session = session
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgba = remove(rgb, session=session)
    alpha = rgba[..., 3]
    mask = (alpha > 127).astype(np.uint8) * 255
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num < 2:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == idx).astype(np.uint8) * 255


_segment_with_rembg._session = None


def measure_jeans(image_path, px_per_mm=PX_PER_MM, debug_dir=None, click_point=None,
                  manual_px_per_mm=None, use_rembg=False):
    """Run the full pipeline.

    If `manual_px_per_mm` is given, skip A4 detection and homography rectification —
    use the input image as-is and measure widths against the supplied scale.
    If `use_rembg` is True, use the rembg U²-Net segmenter for the foreground
    mask (much cleaner than the tile-background heuristic on arbitrary photos).
    """
    img = load_image(image_path)

    if manual_px_per_mm is None:
        bg = estimate_background_hsv(img)
        fg = foreground_mask(img, bg)
        a4_corners, a4_mask = detect_a4(img, fg)
        if a4_corners is None:
            raise RuntimeError("A4 paper not detected. Check background contrast and protocol.")
        scale_unc = scale_uncertainty(a4_corners)
        if use_rembg:
            # rembg + A4: rembg gives the garment mask (much more robust than
            # GrabCut on low-contrast home photos), A4 supplies the scale +
            # perspective. rembg may include the paper as foreground, so
            # subtract a dilated A4 mask and keep the largest CC.
            rembg_mask = _segment_with_rembg(img)
            a4_dilated = cv2.dilate(a4_mask, np.ones((15, 15), np.uint8))
            gm = (rembg_mask > 0) & (a4_dilated == 0)
            gm = gm.astype(np.uint8) * 255
            num, lab, stats, _ = cv2.connectedComponentsWithStats(gm, connectivity=8)
            if num < 2:
                raise RuntimeError("rembg produced no garment blob after A4 removal.")
            idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            jeans = (lab == idx).astype(np.uint8) * 255
        else:
            jeans = isolate_jeans(img, fg, a4_mask, click_point=click_point)
    else:
        a4_corners = None
        if use_rembg:
            fg = _segment_with_rembg(img)
        else:
            fg = foreground_mask_tile_bg(img)
        a4_mask = np.zeros_like(fg)
        # Manual mode: no per-photo scale uncertainty estimate. Use a fixed floor
        # that reflects rough eyeballing precision (~3%).
        scale_unc = 0.03
        jeans = fg

    if jeans is None:
        raise RuntimeError("Jeans not detected in foreground.")

    if manual_px_per_mm is None:
        img_rect, masks_rect = rectify(img, {"jeans": jeans, "a4": a4_mask}, a4_corners, px_per_mm)
        jeans_rect = masks_rect["jeans"]
        effective_px_per_mm = px_per_mm
    else:
        img_rect = img
        jeans_rect = jeans
        effective_px_per_mm = manual_px_per_mm

    if manual_px_per_mm is None:
        img_rot, mask_rot, _ = rotate_to_vertical(img_rect, jeans_rect)
    else:
        # Dataset prior: pants are landscape with waistband to the right.
        # A 90° CW rotation makes the waistband sit at the top. This avoids
        # PCA's bias on Y-shaped silhouettes (principal axis can lock onto
        # the leg-spread diagonal instead of the body axis).
        img_rot = cv2.rotate(img_rect, cv2.ROTATE_90_CLOCKWISE)
        mask_rot = cv2.rotate(jeans_rect, cv2.ROTATE_90_CLOCKWISE)
    img_rot, mask_rot, flipped = ensure_upright(img_rot, mask_rot)

    y_top, y_bottom = vertical_extent(mask_rot)
    y_crotch = find_crotch(mask_rot, y_top, y_bottom)
    if y_crotch is None or y_crotch <= y_top or y_crotch >= y_bottom:
        raise RuntimeError("Crotch point detection failed.")

    yoke_h = y_crotch - y_top
    # Per-leg cuff (each leg ends at a different row when the jeans are spread
    # or asymmetric). Use the closer cuff as the leg-end for the linear ratio
    # interpolation; ratio 1.0 then lands at an actual cuff, not in the
    # spread-leg widest section.
    y_left, y_right = leg_bottoms(mask_rot, y_crotch)
    y_leg_end = min(y_left, y_right)
    leg_h = y_leg_end - y_crotch
    if leg_h <= 0:
        leg_h = y_bottom - y_crotch  # fallback if per-leg scan returned nothing

    results = {}
    for name, ratio in YOKE_RATIOS.items():
        results[name] = build_measurement(
            mask_rot, y_top, yoke_h, ratio, leg=None,
            px_per_mm=effective_px_per_mm, scale_unc=scale_unc,
        )
    for name, ratio in LEG_RATIOS.items():
        results[name] = build_measurement(
            mask_rot, y_crotch, leg_h, ratio, leg="left",
            px_per_mm=effective_px_per_mm, scale_unc=scale_unc,
        )

    if debug_dir is not None:
        save_debug(debug_dir, img, fg, a4_mask, jeans, img_rot, mask_rot, results, y_top, y_crotch, y_bottom)

    return {
        "image": str(image_path),
        "px_per_mm": effective_px_per_mm,
        "manual_scale": manual_px_per_mm is not None,
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

def interactive_click(image_path, max_display=1000):
    """Open a window, capture a single click, return (x, y) in original image coords.

    User interaction: click once on the jeans body, then press any key. Esc cancels.
    """
    img = load_image(image_path)
    h, w = img.shape[:2]
    scale = min(1.0, max_display / max(h, w))
    if scale < 1.0:
        disp_base = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        disp_base = img.copy()

    state = {"click": None}

    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    window = "Click on the jeans body, then press any key (Esc to cancel)"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, cb)
    try:
        while True:
            display = disp_base.copy()
            if state["click"] is not None:
                cv2.circle(display, state["click"], 10, (0, 0, 0), 3)
                cv2.circle(display, state["click"], 10, (0, 255, 255), -1)
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                raise RuntimeError("Click cancelled by user")
            if key != 255 and state["click"] is not None:
                break
    finally:
        cv2.destroyAllWindows()

    cx, cy = state["click"]
    return int(round(cx / scale)), int(round(cy / scale))


def parse_click(s):
    if s is None:
        return None
    parts = s.replace(" ", "").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected --click x,y")
    return int(parts[0]), int(parts[1])


def main():
    p = argparse.ArgumentParser(description="Measure jeans from a top-down photo with an A4 reference.")
    p.add_argument("image", type=Path, help="Input image path (jpg/png/heic).")
    p.add_argument("--debug-dir", type=Path, default=None, help="Write intermediate images here.")
    p.add_argument("--px-per-mm", type=float, default=PX_PER_MM, help="Rectified resolution (default 4).")
    p.add_argument("--json", type=Path, default=None, help="Write the full result as JSON here.")
    p.add_argument("--click", type=parse_click, default=None,
                   help="Seed coordinates 'x,y' on the jeans (skips interactive prompt).")
    p.add_argument("--no-click", action="store_true",
                   help="Disable both interactive click and --click seeding.")
    p.add_argument("--manual-px-per-mm", type=float, default=None,
                   help="Skip A4 detection + homography. Use the given px/mm of the raw image directly. "
                        "For the Circular Fashion dataset (10cm tiles, ~66 px) use ~0.66.")
    p.add_argument("--rembg", action="store_true",
                   help="Use the rembg U^2-Net segmenter for the foreground mask. "
                        "Works with either --manual-px-per-mm (tile-grid/dataset "
                        "mode) or the default A4-detection mode (home photos "
                        "with an A4 for scale).")
    args = p.parse_args()

    click_point = args.click
    if click_point is None and not args.no_click and not args.rembg:
        click_point = interactive_click(args.image)
        print(f"Click seed: {click_point}")

    result = measure_jeans(args.image, px_per_mm=args.px_per_mm, debug_dir=args.debug_dir,
                           click_point=click_point, manual_px_per_mm=args.manual_px_per_mm,
                           use_rembg=args.rembg)

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
