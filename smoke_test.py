"""Build a synthetic top-down 'jeans + A4' scene and run the pipeline end-to-end.

This validates only that the pipeline runs and lands near a known synthetic
ground truth. Real accuracy validation needs real photos with tape measurements.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path("smoke_out")
OUT_DIR.mkdir(exist_ok=True)
IMG_PATH = OUT_DIR / "synthetic.png"
DEBUG_DIR = OUT_DIR / "debug"
JSON_PATH = OUT_DIR / "result.json"

PX_PER_MM = 6.0
CANVAS_MM = (500, 480)
W = int(CANVAS_MM[0] * PX_PER_MM)
H = int(CANVAS_MM[1] * PX_PER_MM)

BG = (60, 110, 70)        # muted green
JEAN = (90, 60, 50)        # dark blue (BGR)
PAPER = (245, 245, 245)

# A4 paper, portrait, in top-right. Fully inside the 500 mm wide canvas.
A4_W_MM, A4_H_MM = 210.0, 297.0
A4_X_MM, A4_Y_MM = 270.0, 15.0  # top-left of A4 in mm

# Jeans geometry. "half" is half the rendered (flat-lay) width at each landmark.
# Expected garment circumference = 2 * (rendered width) = 4 * half.
#
# Constraint for a clean smoke test: LEG_OFFSET >= THIGH_HALF so the two legs
# do NOT overlap at the crotch. Real jeans typically do overlap (thigh wider
# than half the hip), and the pipeline handles that by detecting the crotch
# as the y where legs visually separate — but that confounds the smoke test
# of intrinsic measurement correctness.
WAIST_HALF = 42.0
HIP_HALF = 62.0
THIGH_HALF = 28.0
CALF_HALF = 22.0
ANKLE_HALF = 18.0

CX = 100.0            # jeans horizontal center (mm)
TOP_Y = 30.0
YOKE_H = 90.0
LEG_H = 260.0
CROTCH_Y = TOP_Y + YOKE_H
ANKLE_Y = CROTCH_Y + LEG_H

# 10 mm gap between leg inner edges at the thigh; ~realistic for jeans.
LEG_OFFSET = THIGH_HALF + 10.0
CROTCH_LEG_HALF_OUTER = LEG_OFFSET + THIGH_HALF  # outer extent at crotch row

HIP_RATIO = 0.65
THIGH_RATIO = 0.05
CALF_RATIO = 0.70
ANKLE_RATIO = 1.00


def lerp(a, b, t):
    return a * (1 - t) + b * t


def yoke_half_at(y_mm):
    """Half-width of the yoke at row y_mm. Piecewise linear: waist → hip → crotch."""
    r = (y_mm - TOP_Y) / YOKE_H
    if r <= HIP_RATIO:
        t = r / HIP_RATIO
        return lerp(WAIST_HALF, HIP_HALF, t)
    t = (r - HIP_RATIO) / (1.0 - HIP_RATIO)
    return lerp(HIP_HALF, CROTCH_LEG_HALF_OUTER, t)


def leg_half_at(y_mm):
    """Half-width of one leg at row y_mm. Piecewise linear: thigh → calf → ankle."""
    r = (y_mm - CROTCH_Y) / LEG_H
    if r <= THIGH_RATIO:
        # Smoothly approach thigh from the yoke join. Constant for simplicity.
        return THIGH_HALF
    if r <= CALF_RATIO:
        t = (r - THIGH_RATIO) / (CALF_RATIO - THIGH_RATIO)
        return lerp(THIGH_HALF, CALF_HALF, t)
    t = (r - CALF_RATIO) / (ANKLE_RATIO - CALF_RATIO)
    return lerp(CALF_HALF, ANKLE_HALF, t)


def render():
    img = np.full((H, W, 3), BG, dtype=np.uint8)

    # A4 paper
    x0 = int(A4_X_MM * PX_PER_MM)
    y0 = int(A4_Y_MM * PX_PER_MM)
    x1 = int((A4_X_MM + A4_W_MM) * PX_PER_MM)
    y1 = int((A4_Y_MM + A4_H_MM) * PX_PER_MM)
    cv2.rectangle(img, (x0, y0), (x1, y1), PAPER, -1)

    # Jeans, rendered row by row in the jeans column.
    for y_px in range(H):
        y_mm = y_px / PX_PER_MM
        intervals_mm = []
        if TOP_Y <= y_mm <= CROTCH_Y:
            h = yoke_half_at(y_mm)
            intervals_mm.append((CX - h, CX + h))
        elif CROTCH_Y < y_mm <= ANKLE_Y:
            h = leg_half_at(y_mm)
            intervals_mm.append((CX - LEG_OFFSET - h, CX - LEG_OFFSET + h))
            intervals_mm.append((CX + LEG_OFFSET - h, CX + LEG_OFFSET + h))
        for l_mm, r_mm in intervals_mm:
            l_px = max(0, int(l_mm * PX_PER_MM))
            r_px = min(W, int(r_mm * PX_PER_MM))
            if r_px > l_px:
                img[y_px, l_px:r_px] = JEAN

    return img


img = render()
cv2.imwrite(str(IMG_PATH), img)
print(f"Wrote synthetic image: {IMG_PATH} ({W}x{H})")

# Expected circumferences: 2 × rendered_width = 2 × (2 × half) = 4 × half.
expected = {
    "waist": 4 * WAIST_HALF,
    "hip": 4 * HIP_HALF,
    "thigh": 4 * THIGH_HALF,
    "calf": 4 * CALF_HALF,
    "ankle": 4 * ANKLE_HALF,
}
print("\nExpected circumferences (mm):")
for k, v in expected.items():
    print(f"  {k:<6} {v:6.1f}")

print("\n--- Running measure_jeans.py ---")
subprocess.run(
    [sys.executable, "measure_jeans.py", str(IMG_PATH),
     "--debug-dir", str(DEBUG_DIR), "--json", str(JSON_PATH),
     "--no-click"],
    check=True,
)

data = json.loads(JSON_PATH.read_text())
print("\nMeasured vs expected (mm):")
print(f"  {'name':<6} {'measured':>10}  {'expected':>10}  {'delta':>8}")
ok = True
for k, exp in expected.items():
    m = data["measurements"].get(k)
    if m is None:
        print(f"  {k:<6} FAILED")
        ok = False
        continue
    meas = m["circumference_mm"]
    delta = meas - exp
    flag = "" if abs(delta) <= 4 else "  <-- off"
    if abs(delta) > 4:
        ok = False
    print(f"  {k:<6} {meas:>8.1f}    {exp:>8.1f}    {delta:>+6.1f}{flag}")

print(f"\nOverall: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
