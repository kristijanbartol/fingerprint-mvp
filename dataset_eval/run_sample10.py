"""Run the measurement pipeline on the 10-jeans validation slice.

Expects CIRCULAR_FASHION_ROOT to point at a local copy of the Circular
Fashion v2 dataset (https://fnauman.github.io/second-hand-fashion/), or
pass --dataset-root explicitly.

Usage:
    CIRCULAR_FASHION_ROOT=/path/to/circular_fashion_v2 \
        python dataset_eval/run_sample10.py
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILE = Path(__file__).resolve().parent / "sample10.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=os.environ.get("CIRCULAR_FASHION_ROOT"),
                    help="Path to the circular_fashion_v2 dataset (or set CIRCULAR_FASHION_ROOT).")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "runs"))
    ap.add_argument("--px-per-mm", type=float, default=0.66,
                    help="Manual calibration; ~0.66 holds for the 1280x720 tile-grid era.")
    args = ap.parse_args()
    if not args.dataset_root:
        ap.error("--dataset-root or CIRCULAR_FASHION_ROOT env var is required")
    root = Path(args.dataset_root)

    samples = json.loads(SAMPLE_FILE.read_text())
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in samples:
        front = root / s["front"]
        if not front.exists():
            rows.append((s["brand"], s["size"], "MISSING", front, None, None, None, None, None))
            continue
        out_dir = out_root / s["ts"]
        json_out = out_dir / "result.json"
        cmd = [sys.executable, str(REPO_ROOT / "measure_jeans.py"), str(front),
               "--no-click", "--manual-px-per-mm", str(args.px_per_mm),
               "--rembg", "--debug-dir", str(out_dir), "--json", str(json_out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            err = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "(unknown error)"
            rows.append((s["brand"], s["size"], "FAIL", err, None, None, None, None, None))
            continue
        d = json.loads(json_out.read_text())
        m = d["measurements"]
        def g(k):
            return m[k]["circumference_mm"] if m.get(k) else None
        rows.append((s["brand"], s["size"], "OK", None,
                     g("waist"), g("hip"), g("thigh"), g("calf"), g("ankle")))

    header = f'{"brand":<22} {"sz":<3} {"ok":<4} {"waist":>7} {"hip":>7} {"thigh":>7} {"calf":>7} {"ankle":>7}'
    print(header)
    print("-" * len(header))
    for b, sz, ok, err, w, h, t, c, a in rows:
        if ok != "OK":
            print(f'{b:<22} {sz:<3} {ok:<4} {err}')
            continue
        def f(x):
            return f"{x:7.0f}" if x is not None else "   None"
        print(f'{b:<22} {sz:<3} {ok:<4} {f(w)} {f(h)} {f(t)} {f(c)} {f(a)}')


if __name__ == "__main__":
    main()
