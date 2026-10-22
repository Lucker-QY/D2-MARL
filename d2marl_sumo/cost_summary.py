# -*- coding: utf-8 -*-
"""Summarise the JSON files written by inference_cost / inference_cost_torch into the inference-cost table.

    python -m d2marl_sumo.cost_summary cost_json_v2

Why the median (p50) is the main column rather than the mean:
  learning-based methods have only a few dozen decision samples per run, and a
  single spike (e.g. TensorFlow retracing the graph, ~150 ms) can raise the mean by
  15-40 %. Latency is conventionally reported by its median; mean, p95 and max are
  still listed.
"""

import os
import sys
import csv
import glob
import json

NAME = {
    "D2MARL": "D2-MARL", "AdvancedColight": "Adv-CoLight",
    "EfficientColight": "Eff-CoLight", "CoLight": "CoLight",
    "PressLight": "PressLight", "MPLight": "MPLight", "Attend": "AttendLight",
    "MaxPressure": "MaxPressure", "Fixedtime": "Fixed-time",
    "Random": "Random", "Actuated": "Actuated",
    "CoordLight": "CoordLight", "LLMLight": "LLMLight-13B",
}
ACTION_INTERVAL_MS = 15000.0     # decision interval of 15 s (Table I)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "cost_json_v2"
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        tag = r.get("tag") or os.path.basename(f).split("_")[0]
        n = max(int(r["n_intersections"]), 1)
        # Number of timed calls per decision step:
        #   methods with one shared network (D2-MARL / CoLight family / CoordLight / LLMLight)
        #   call it once per step for all intersections;
        #   methods with one agent per intersection (PressLight / Fixed-time / MaxPressure /
        #   Actuated / Random) call it n times; the JSON holds the time of a single intersection.
        sim = float(r.get("sim_seconds") or 3600)
        steps = sim / (ACTION_INTERVAL_MS / 1000.0)
        k = max(1, int(round(r["n_decisions"] / steps)))
        p50c = r.get("latency_ms_p50", r["latency_ms_mean"])
        p50 = p50c * k              # approximation: identical per-intersection times, step time = single time x calls
        rows.append(dict(
            tag=tag, method=NAME.get(r["method"], r["method"]),
            n_inter=int(r["n_intersections"]), params=int(r["trainable_params"]),
            calls_per_step=k,
            p50=p50, mean=r["latency_ms_mean"] * k, p95=r["latency_ms_p95"] * k,
            max=r.get("latency_ms_max", float("nan")) * k,
            per_inter_p50=p50 / n, n_dec=int(r["n_decisions"]),
            device=r.get("device", "cpu"),
            share=100.0 * p50 / ACTION_INTERVAL_MS,
        ))
    if not rows:
        raise SystemExit("no JSON under %s" % d)

    for tag in sorted({r["tag"] for r in rows}):
        grp = sorted([r for r in rows if r["tag"] == tag], key=lambda r: r["p50"])
        ours = next((r for r in grp if r["method"] == "D2-MARL"), None)
        print("\n" + "=" * 118)
        print("scenario %s   (%d intersections)" % (tag, grp[0]["n_inter"]))
        print("=" * 118)
        print("%-14s %12s %5s %10s %10s %10s %11s %8s %10s %6s  %s" % (
            "method", "params", "calls/step", "step p50", "step mean", "step p95",
            "inter p50", "vs ours", "of 15s", "samples", "device"))
        print("-" * 118)
        for r in grp:
            ps = "LLM(13B)" if r["params"] < 0 else ("-" if r["params"] == 0 else "{:,}".format(r["params"]))
            ratio = ("%.2fx" % (r["p50"] / ours["p50"])) if ours and ours["p50"] > 0 else "-"
            print("%-14s %12s %5d %10.3f %10.3f %10.3f %11.4f %8s %9.3f%% %6d  %s" % (
                r["method"], ps, r["calls_per_step"], r["p50"], r["mean"], r["p95"],
                r["per_inter_p50"], ratio, r["share"], r["n_dec"], r["device"]))

    out = os.path.join(d, "cost_summary.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwritten to %s" % out)
    print("Note: LLMLight is measured on a GPU, all others serially on the same CPU node; absolute values are not directly comparable.")
    print("Note: for methods with calls/step > 1 (one agent per intersection) the step time is approximated as single time x calls.")


if __name__ == "__main__":
    main()
