"""Run the measurement pipeline over the tile-grid jeans subset and capture
all five circumferences per garment (not just hip).

Extends hip_survey.py — same filter, same plausibility gate, same worker
model, but the CSV keeps waist / hip / thigh / calf / ankle + their
reported uncertainties. This is the schema that home2market.py matches
against.

Usage:
    CIRCULAR_FASHION_ROOT=/path/to/circular_fashion_v2 \
        python dataset_eval/full_survey.py [--workers 4] [--limit N]
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = Path(__file__).resolve().parent / "full_survey.csv"

BRAND_SENTINELS = {"", "not in the list", "missing", "none"}
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep", "oct", "nov", "dec"]

# Plausibility windows per measurement (mm). Very loose — the point is to
# catch obvious geometry failures, not to enforce anatomy.
PLAUSIBILITY = {
    "waist": (400.0, 1500.0),
    "hip":   (600.0, 1500.0),
    "thigh": (200.0,  900.0),
    "calf":  (100.0,  700.0),
    "ankle": ( 50.0,  600.0),
}
MEASUREMENT_ORDER = ["waist", "hip", "thigh", "calf", "ankle"]


def parse_month(name):
    m = re.match(r"^([a-z]{3})(\d{4})$", name.strip().lower())
    if not m or m.group(1) not in MONTHS:
        return None
    return int(m.group(2)), MONTHS.index(m.group(1))


def is_tile_grid(year, month_idx):
    return (year, month_idx) >= (2023, 2)


def walk_jeans(root):
    for station_dir in sorted(root.glob("station*")):
        for month_dir in sorted(station_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            ym = parse_month(month_dir.name)
            if ym is None or not is_tile_grid(*ym):
                continue
            for jp in sorted(month_dir.glob("labels_*.json")):
                try:
                    d = json.loads(jp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if (d.get("type") or "").strip() != "Jeans":
                    continue
                if (d.get("pattern") or "None") not in (None, "None"):
                    continue
                if d.get("category") not in ("Ladies", "Men"):
                    continue
                brand = (d.get("brand") or "").strip()
                if brand.lower() in BRAND_SENTINELS:
                    continue
                size = str(d.get("size", ""))
                if not (size.isdigit() and len(size) == 2):
                    continue
                ts = jp.stem.replace("labels_", "")
                front = month_dir / f"front_{ts}.jpg"
                if not front.exists():
                    continue
                yield (ts, station_dir.name, brand, size, d.get("category"),
                       d.get("cut"), front)


def measure_all(front):
    """Return dict of measurement -> circumference_mm plus overall status.

    Overall status is 'ok' only if all five circumferences pass their
    plausibility windows. If any measurement is missing or out of range,
    we record it in the CSV but mark the row 'failed_plausibility' (or
    'failed_pipeline' if the whole pipeline crashed).
    """
    with tempfile.TemporaryDirectory() as tmp:
        json_out = Path(tmp) / "r.json"
        r = subprocess.run(
            [sys.executable, str(REPO_ROOT / "measure_jeans.py"), str(front),
             "--no-click", "--manual-px-per-mm", "0.66", "--rembg",
             "--json", str(json_out)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return {}, {}, "failed_pipeline"
        try:
            d = json.loads(json_out.read_text())
        except Exception:
            return {}, {}, "failed_pipeline"
    ms = d.get("measurements", {})
    values, uncs = {}, {}
    status = "ok"
    for name in MEASUREMENT_ORDER:
        m = ms.get(name)
        if m is None or m.get("circumference_mm") is None:
            status = "failed_pipeline"
            continue
        v = float(m["circumference_mm"])
        u = float(m.get("uncertainty_mm") or 0.0)
        values[name] = v
        uncs[name] = u
        lo, hi = PLAUSIBILITY[name]
        if not (lo <= v <= hi):
            status = "failed_plausibility"
    return values, uncs, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=os.environ.get("CIRCULAR_FASHION_ROOT"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()
    if not args.dataset_root:
        ap.error("--dataset-root or CIRCULAR_FASHION_ROOT env var is required")
    root = Path(args.dataset_root)

    candidates = list(walk_jeans(root))
    if args.limit:
        candidates = candidates[:args.limit]
    print(f"Candidates: {len(candidates)}", file=sys.stderr)

    def cut_str(c):
        return ",".join(c) if isinstance(c, list) else (c or "")

    def job(item):
        ts, station, brand, size, category, cut, front = item
        values, uncs, status = measure_all(front)
        return (ts, station, brand, size, category, cut, values, uncs, status)

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, c) for c in candidates]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
            except Exception as e:
                print(f"exception: {e}", file=sys.stderr)
                continue
            rows.append(res)
            if i % 25 == 0 or i == len(candidates):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed else 0
                eta = (len(candidates) - i) / rate if rate else 0
                print(f"[{i}/{len(candidates)}] {elapsed:.0f}s elapsed, "
                      f"ETA {eta:.0f}s", file=sys.stderr)

    rows.sort(key=lambda r: r[0])
    header = (["ts", "station", "brand", "size", "category", "cut"]
              + [f"{m}_mm"  for m in MEASUREMENT_ORDER]
              + [f"{m}_unc" for m in MEASUREMENT_ORDER]
              + ["status"])
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for ts, station, brand, size, category, cut, values, uncs, status in rows:
            row = [ts, station, brand, size, category, cut_str(cut)]
            for m in MEASUREMENT_ORDER:
                row.append(values.get(m, ""))
            for m in MEASUREMENT_ORDER:
                row.append(uncs.get(m, ""))
            row.append(status)
            w.writerow(row)
    print(f"\nWrote {args.out} ({len(rows)} rows)", file=sys.stderr)

    from collections import Counter
    by_status = Counter(r[-1] for r in rows)
    print("Status breakdown:", file=sys.stderr)
    for s, c in sorted(by_status.items(), key=lambda kv: -kv[1]):
        print(f"  {s:<22} {c}", file=sys.stderr)


if __name__ == "__main__":
    main()
