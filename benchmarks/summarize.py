#!/usr/bin/env python3
"""Turn benchmarks/results/*/raw.csv into a Markdown table (medians over repeats).

Usage: summarize.py raw.csv [hardware.txt]
"""
import csv
import statistics
import sys
from collections import defaultdict


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2

    runs = defaultdict(list)
    unstable = set()
    with open(sys.argv[1], newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["width"]), int(row["height"]), row["mode"])
            runs[key].append(row)
            if row["finite"] != "ok":
                unstable.add(key)

    def med(key, col):
        rows = runs.get(key)
        return statistics.median(float(r[col]) for r in rows) if rows else None

    def spread(key, col):
        rows = runs.get(key, [])
        vals = [float(r[col]) for r in rows]
        if len(vals) < 2:
            return ""
        m = statistics.median(vals)
        return f" (±{(max(vals) - min(vals)) / 2 / m * 100:.0f}%)" if m else ""

    def fmt(x, digits=1):
        return "—" if x is None else f"{x:,.{digits}f}"

    hw = {}
    if len(sys.argv) > 2:
        with open(sys.argv[2]) as f:
            for line in f:
                if ":" in line and not line.startswith("#"):
                    k, v = line.split(":", 1)
                    hw[k.strip()] = v.strip()

    if hw:
        print(f"GPU: {hw.get('gpu', '?')}  ")
        print(f"CPU: {hw.get('cpu', '?')}  ")
        print(f"OS: {hw.get('os', '?')}; {hw.get('nvcc', '')}  ")
        print(f"Commit {hw.get('commit', '?')}, {hw.get('settings', '')}")
        print()

    sizes = sorted({(w, h) for (w, h, _) in runs}, key=lambda s: s[0] * s[1])
    has_render = any(m == "render" for (_, _, m) in runs)
    has_cpu = any(m == "cpu" for (_, _, m) in runs)

    header = ["Grid", "Nodes", "GPU ms/step", "GPU MLUPS"]
    if has_render:
        header += ["GPU + render, FPS"]
    if has_cpu:
        header += ["CPU prototype ms/step", "Speed-up vs CPU prototype"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")

    for w, h in sizes:
        g, r, c = (w, h, "compute"), (w, h, "render"), (w, h, "cpu")
        cells = [f"{w}×{h}", f"{w * h / 1e6:.2f} M",
                 fmt(med(g, "ms_per_step"), 2) + spread(g, "ms_per_step"),
                 fmt(med(g, "mlups"), 0)]
        if has_render:
            cells.append(fmt(med(r, "steps_per_s"), 0))
        if has_cpu:
            cpu_ms, gpu_ms = med(c, "ms_per_step"), med(g, "ms_per_step")
            cells.append(fmt(cpu_ms, 1))
            cells.append(f"{cpu_ms / gpu_ms:,.0f}×" if cpu_ms and gpu_ms else "—")
        print("| " + " | ".join(cells) + " |")

    if unstable:
        print()
        print("**Warning:** NaN/Inf in the final field for: "
              + ", ".join(f"{w}×{h} {m}" for (w, h, m) in sorted(unstable)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
