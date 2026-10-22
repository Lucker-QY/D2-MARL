"""
collect_results.py
==================
Summarises the per-round test metrics of all runs under records/ into one table:
  1. to monitor running jobs (progress, trends);
  2. to produce the multi-seed mean +- standard deviation of the paper tables.

Metric definitions (identical to LLMTSCS):
  ATT : sum(leave - enter) merged over intersections; unfinished trips use RUN_COUNTS
  Mwt : longest single waiting episode of any vehicle (also from delay_events_over_90s.csv)
  Cd  : number of waiting episodes longer than 90 s
  Awt / AQL / Gini : read from reward_log.txt

Usage:
    python -m d2marl_sumo.collect_results --records records --last-n 5
    python -m d2marl_sumo.collect_results --records records --csv summary.csv
"""

import os
import re
import csv
import glob
import json
import time
import argparse
from collections import defaultdict


def parse_reward_log(path):
    out = {}
    pats = {
        "reward": r"Episode average reward:\s*([-\d.]+)",
        "Awt": r"AWT[^:]*:\s*([\d.]+)",
        "AQL": r"AQL[^:]*:\s*([\d.]+)",
        "Gini": r"Gini coefficient[^:]*:\s*([\d.]+)",
        "Cd": r"Severe delay events[^:]*:\s*(\d+)",
    }
    try:
        txt = open(path, encoding="utf-8", errors="ignore").read()
    except IOError:
        return out
    for k, p in pats.items():
        m = re.search(p, txt)
        if m:
            out[k] = float(m.group(1))
    return out


def round_metrics(round_dir, horizon=3600):
    """ATT and Mwt are computed from the per-vehicle files, as in LLMTSCS/oneline.py."""
    tt, seen, done = {}, set(), set()
    for f in glob.glob(os.path.join(round_dir, "vehicle_inter_*.csv")):
        try:
            fh = open(f)
        except IOError:
            continue
        with fh:
            for r in csv.DictReader(fh):
                v = r.get("") or r.get("Unnamed: 0")
                if not v or "shadow" in v:
                    continue
                try:
                    e = float(r["enter_time"])
                except (KeyError, ValueError, TypeError):
                    continue
                if e != e:                      # nan
                    continue
                seen.add(v)
                l = r.get("leave_time", "")
                try:
                    lv = float(l)
                    if lv != lv:
                        raise ValueError
                    done.add(v)
                except (ValueError, TypeError):
                    lv = horizon
                tt[v] = tt.get(v, 0.0) + (lv - e)

    # An empty round is invalid as a whole; its metrics must not become 0.
    # A job often leaves a round directory that was created but never written.
    # Counting Mwt / finish_rate as 0.0 there would pull the last-N mean down.
    # ATT / Awt / AQL / Gini return None when missing and are filtered correctly.
    if not seen:
        return None

    # Censoring of Mwt: the event file only holds waits above 90 s, so the value
    # computed here is the maximum over waits above 90 s. For Cd > 0 this equals
    # the true maximum; for a method that never exceeds 90 s the file is missing
    # or empty, and Mwt is "< 90", not 0.
    # mwt_censored marks this case; tables print "<90".
    mwt = None
    censored = True
    p = os.path.join(round_dir, "delay_events_over_90s.csv")
    if os.path.exists(p):
        vals = []
        with open(p) as fh:
            for r in csv.DictReader(fh):
                try:
                    vals.append(float(r["wait_time"]))
                except (KeyError, ValueError, TypeError):
                    pass
        if vals:
            mwt = max(vals)
            censored = False

    # ---- Mwt_cum: maximum over vehicles of the **cumulative** waiting time ----
    # Mwt is the longest single continuous wait. It can be lowered by splitting a
    # long wait into several pieces (the timer restarts whenever the vehicle moves).
    #
    # all_vehicles_wait_time.csv holds env.vehicle_total_wait_time, the waiting
    # seconds accumulated by every vehicle over its trip (also used for Gini). Its
    # maximum is the total wait of the worst-off traveller and cannot be split.
    # Both definitions are reported.
    # Mwt_exact: exact network-wide longest single continuous wait, no threshold, no censoring.
    # Older outputs without the max_single_wait column return None (left empty),
    # so the two definitions are never mixed in one column.
    mwt_cum = None
    mwt_exact = None
    pw = os.path.join(round_dir, "all_vehicles_wait_time.csv")
    if os.path.exists(pw):
        vals, exact = [], []
        with open(pw) as fh:
            for r in csv.DictReader(fh):
                try:
                    vals.append(float(r["total_wait_time"]))
                except (KeyError, ValueError, TypeError):
                    pass
                try:
                    exact.append(float(r["max_single_wait"]))
                except (KeyError, ValueError, TypeError):
                    pass
        if vals:
            mwt_cum = max(vals)
        if exact:
            mwt_exact = max(exact)

    d = parse_reward_log(os.path.join(round_dir, "reward_log.txt"))
    d["Mwt_cum"] = mwt_cum
    d["Mwt_exact"] = mwt_exact
    d["ATT"] = (sum(tt.values()) / len(tt)) if tt else None
    d["Mwt"] = mwt
    d["mwt_censored"] = censored
    d["n_vehicles"] = len(seen)
    d["n_finished"] = len(done)
    d["finish_rate"] = len(done) / len(seen)
    return d


# ---- cache of round metrics -----------------------------------------
# A full collection parses every vehicle_inter_*.csv of every round, which can
# take an hour for a large records/ tree. A finished round directory never
# changes, so its metrics are cached by (round directory -> mtime + metrics).
#
# Only directories older than CACHE_MIN_AGE seconds are cached: a round that is
# still being written may be incomplete.
CACHE_MIN_AGE = 600.0

# Cache version. **Increase it whenever the fields returned by round_metrics or
# the computation of any metric change**; otherwise stale entries would be used
# and new metrics would silently be empty.
CACHE_VERSION = 3


def _load_cache(records_dir):
    path = os.path.join(records_dir, ".collect_cache.json")
    try:
        with open(path) as f:
            blob = json.load(f)
    except (IOError, ValueError):
        return {}, path
    # a different version invalidates the whole cache
    if not isinstance(blob, dict) or blob.get("_v") != CACHE_VERSION:
        print("[cache] version changed from %s to %s; cache discarded, recomputing everything"
              % (blob.get("_v") if isinstance(blob, dict) else "none",
                 CACHE_VERSION))
        return {}, path
    return blob.get("rounds", {}), path


def _save_cache(cache, path):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"_v": CACHE_VERSION, "rounds": cache}, f)
        os.replace(tmp, path)
    except (IOError, OSError):
        pass          # a cache write failure must not break the collection


def scan(records_dir, include=None, exclude=None, use_cache=True):
    """Return {memo: {traffic: {round: metrics}}}.

    include/exclude are applied here and not after the scan: round_metrics
    parses every vehicle_inter_*.csv under every test_round, which is slow on a
    large records/ tree.
    """
    exclude = [x for x in (exclude or []) if x]
    out = defaultdict(lambda: defaultdict(dict))
    cache, cache_path = _load_cache(records_dir) if use_cache else ({}, None)
    now = time.time()
    hit = miss = 0
    for memo in sorted(os.listdir(records_dir)):
        mdir = os.path.join(records_dir, memo)
        if not os.path.isdir(mdir):
            continue
        if include and include not in memo:
            continue
        if any(e in memo for e in exclude):
            continue
        for run in sorted(os.listdir(mdir)):
            rdir = os.path.join(mdir, run, "test_round")
            if not os.path.isdir(rdir):
                continue
            traffic = run.split(".json")[0]
            for rd in os.listdir(rdir):
                m = re.match(r"round_(\d+)$", rd)
                if not m:
                    continue
                rpath = os.path.join(rdir, rd)
                try:
                    mt = os.path.getmtime(rpath)
                except OSError:
                    continue
                ent = cache.get(rpath)
                if use_cache and ent and ent[0] == mt:
                    met = ent[1]
                    hit += 1
                else:
                    met = round_metrics(rpath)
                    miss += 1
                    # rounds still being written are not cached
                    if use_cache and now - mt > CACHE_MIN_AGE:
                        cache[rpath] = [mt, met]
                if met is not None:            # skip empty rounds
                    out[memo][traffic][int(m.group(1))] = met
    if use_cache and miss:
        _save_cache(cache, cache_path)
    if use_cache:
        print("[cache] %d rounds from cache, %d parsed" % (hit, miss))
    return out


def _paired_t(d):
    """Two-sided p-value of the paired t-test, without scipy.

    Uses the exact CDF of the t distribution: for small samples (n <= 20) the
    normal approximation underestimates p.
    """
    import math
    n = len(d)
    if n < 2:
        return None
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    if var <= 0:
        return 0.0 if m != 0 else 1.0
    t = m / math.sqrt(var / n)
    df = n - 1
    # two-sided p = I_{df/(df+t^2)}(df/2, 1/2), regularised incomplete beta
    # function via continued fractions (Numerical Recipes 6.4).
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5

    def betacf(a, b, x):
        MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d, h = 1.0 / d, 1.0 / d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < EPS:
                break
        return h

    lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    if x < (a + 1.0) / (a + b + 2.0):
        ib = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) * betacf(a, b, x) / a
    else:
        ib = 1.0 - math.exp(b * math.log(1 - x) + a * math.log(x) - lbeta) * betacf(b, a, 1 - x) / b
    return max(0.0, min(1.0, ib))


def _wilcoxon(d):
    """Exact two-sided p of the paired Wilcoxon signed-rank test (all 2^n sign patterns, n <= 15).

    With only 6-8 scenarios the t-test relies on near-normal differences, while the
    differences of ATT/Cd across scenarios are skewed. Both tests are reported; if
    they disagree, the evidence is insufficient ("no significant difference").
    For n = 6 the smallest possible two-sided p is 0.031.
    """
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    if n > 15:
        return None            # enumeration too expensive (not needed here)
    order = sorted(range(n), key=lambda i: abs(d[i]))
    rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0          # average rank for ties
        for k in range(i, j + 1):
            rank[order[k]] = avg
        i = j + 1
    w_obs = sum(rank[i] for i in range(n) if d[i] > 0)
    total = sum(rank)
    cnt = 0
    for mask in range(1 << n):
        w = sum(rank[i] for i in range(n) if mask & (1 << i))
        if abs(w - total / 2.0) >= abs(w_obs - total / 2.0) - 1e-9:
            cnt += 1
    return cnt / float(1 << n)


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    n = len(vals)
    mu = sum(vals) / n
    if n < 2:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, var ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="records")
    ap.add_argument("--last-n", type=int, default=5,
                    help="aggregate the last N valid rounds of each run")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--include", default=None,
                    help="only memos containing this substring (substring match, "
                         "so choose prefixes that do not contain each other)")
    ap.add_argument("--exclude", default="smoke,pilot",
                    help="exclude memos containing any of these comma-separated "
                         "substrings (default: smoke,pilot)")
    ap.add_argument("--progress", action="store_true",
                    help="only show progress: the latest round of each run")
    ap.add_argument("--agg", choices=["mean", "median"], default="mean",
                    help="aggregation of the last N rounds (default mean). "
                         "median is robust to single-round spikes")
    ap.add_argument("--at-round", type=int, default=None, metavar="N",
                    help="use rounds N-last_n .. N-1 for every run instead of the "
                         "last N rounds of each run, so that all runs are compared "
                         "at the same training budget; 0 = the smallest round "
                         "reached by all runs")
    ap.add_argument("--sig", metavar="METHOD", default=None,
                    help="significance test: part of the reference method name "
                         "(e.g. D2MARL); every other method is compared "
                         "paired by scenario")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the round cache and parse everything again; "
                         "the cache is only used for round directories whose mtime is unchanged")
    ap.add_argument("--spikes", action="store_true",
                    help="diagnostics only: list runs whose last-N mean differs "
                         "strongly from the median (single-round spikes)")
    a = ap.parse_args()

    # smoke and pilot memos are excluded so that they do not enter the multi-seed
    # statistics as extra seeds.
    ex = [x for x in (a.exclude or "").split(",") if x]
    data = scan(a.records, include=a.include, exclude=ex,
                use_cache=not a.no_cache)
    if not data:
        print("no test_round records under {0}".format(a.records))
        return

    if a.spikes:
        print("%-44s %-26s %6s %10s %10s %7s  %s" % (
            "memo", "traffic", "metric", "mean", "median", "ratio", "last N rounds"))
        print("-" * 128)
        hit = 0
        for memo in sorted(data):
            for tr in sorted(data[memo]):
                rs = data[memo][tr]
                idx = sorted(rs)[-a.last_n:]
                for k in ("ATT", "Cd", "Mwt"):
                    v = [rs[i].get(k) for i in idx]
                    v = [x for x in v if x is not None]
                    if len(v) < 3:
                        continue
                    mu = sum(v) / len(v)
                    md = sorted(v)[len(v) // 2]
                    # mean more than 50% above the median: a single-round spike in the window
                    if md > 0 and mu > 1.5 * md:
                        hit += 1
                        print("%-44s %-26s %6s %10.1f %10.1f %6.1fx  %s" % (
                            memo[:44], tr[:26], k, mu, md, mu / md,
                            " ".join("%.0f" % x for x in v)))
        print("-" * 128)
        print("%d cases. These values are inflated by a single-round spike, "
              "not by divergence; use --agg median." % hit)
        return

    if a.progress:
        print("%-28s %-34s %8s %10s" % ("memo", "traffic", "rounds", "last ATT"))
        print("-" * 84)
        for memo in sorted(data):
            for tr in sorted(data[memo]):
                rs = data[memo][tr]
                last = max(rs)
                att = rs[last].get("ATT")
                print("%-28s %-34s %8d %10s" % (
                    memo[:28], tr[:34], len(rs),
                    "  N/A " if att is None else "%.1f" % att))
        return

    # ---- evaluation window ----
    # By default each run uses its own last last_n rounds. If runs have different
    # numbers of rounds this is not comparable; --at-round fixes one window for all.
    at = a.at_round
    if at == 0:
        at = min(len(data[m][t]) for m in data for t in data[m])
        print("[--at-round 0] smallest round reached by all runs = %d" % at)

    def window(rs):
        ordered = sorted(rs)
        if at is None:
            return ordered[-a.last_n:]
        # Select by round **number**, not by position: a run may miss a round
        # (a failed test episode leaves an empty round_N), and slicing by
        # position would silently shift its window by one round.
        w = [r for r in ordered if at - a.last_n <= r < at]
        # Runs with fewer than `at` rounds (non-learning methods have a single
        # test round) would get an empty window; the window does not apply to
        # them, so they use their own last rounds.
        return w if w else ordered[-a.last_n:]

    rows = []
    print("=" * 114)
    if at is None:
        print("Final result of every run (its own last %d valid rounds)" % a.last_n)
    else:
        print("Final result of every run (common window: rounds %d-%d)"
              % (max(0, at - a.last_n), at - 1))
    if ex:
        print("excluded memos containing: %s" % ", ".join(ex))
    if a.include:
        print("only memos containing: %s" % a.include)
    print("=" * 114)
    print("%-42s %-30s %5s %6s %8s %8s %8s %8s %8s %8s %7s %7s %7s" % (
        "memo", "traffic", "rounds", "used", "ATT", "Awt", "AQL", "Mwt",
        "Mwt_exact", "Mwt_cum", "Cd", "Gini", "finish"))
    print("-" * 124)
    thin = []          # runs with missing rounds inside the common window, reported after the table
    for memo in sorted(data):
        for tr in sorted(data[memo]):
            rs = data[memo][tr]
            idx = window(rs)   # valid rounds only; empty rounds were dropped in scan
            # Selecting by number prevents shifted windows, but a missing round
            # **inside** the window leaves this run with fewer samples; report it.
            if at is not None and len(idx) < a.last_n and len(rs) >= a.last_n:
                thin.append((memo, tr, len(idx),
                             sorted(set(range(max(0, at - a.last_n), at)) - set(rs))))
            agg = {}
            # n_vehicles is part of the table: it is the direct check that two
            # methods simulate the same vehicles; a finish rate far off the other
            # methods usually means a different number of loaded vehicles.
            for k in ("ATT", "Awt", "AQL", "Mwt", "Mwt_exact", "Mwt_cum",
                      "Cd", "Gini",
                      "finish_rate", "n_vehicles", "n_finished"):
                vals = [rs[i].get(k) for i in idx]
                if a.agg == "median":
                    vv = sorted(x for x in vals if x is not None)
                    agg[k] = vv[len(vv) // 2] if vv else None
                else:
                    agg[k] = mean_std(vals)[0]
            mwt_cell = ("   <90" if agg["Mwt"] is None
                        else "%6.0f" % agg["Mwt"])
            # older outputs lack Mwt_exact: leave it empty instead of 0, which
            # would read as "the longest wait is 0 s" (same error as with "<90").
            exact_cell = ("     -" if agg.get("Mwt_exact") is None
                          else "%6.0f" % agg["Mwt_exact"])
            print("%-42s %-30s %5d %6s %8.1f %8.1f %8.1f %8s %8s %8.0f %7.0f %7.3f %6.1f%%" % (
                memo[:42], tr[:30], len(rs),
                "%d-%d" % (idx[0], idx[-1]) if idx else "-",
                agg["ATT"] or 0, agg["Awt"] or 0, agg["AQL"] or 0,
                mwt_cell, exact_cell, agg.get("Mwt_cum") or 0, agg["Cd"] or 0,
                agg["Gini"] or 0, 100 * (agg["finish_rate"] or 0)))
            rows.append(dict(memo=memo, traffic=tr, n_rounds=len(rs), **agg))

    # Aggregate the seeds by (method, traffic).
    # Aggregating by traffic alone would average different methods (or ablation
    # arms) as if they were seeds of one method.
    # method name = memo without the trailing -seedNNN.
    if thin:
        print("")
        print("!! these runs miss rounds inside the common window; their sample "
              "size is smaller than that of the other runs:")
        for memo, tr, n, miss in thin:
            print("   %-42s %-26s only %d/%d rounds in the window, missing %s"
                  % (memo, tr, n, a.last_n,
                     ", ".join("round_%d" % r for r in miss)))
        print("   (rounds missing outside the window do not matter)")

    print("\n" + "=" * 114)
    print("Multi-seed statistics by (method, traffic) (mean +- std), as in the paper tables")
    print("=" * 114)
    by_group = defaultdict(list)
    for r in rows:
        method = re.sub(r"-seed\d+$", "", r["memo"])
        by_group[(method, r["traffic"])].append(r)
    print("%-38s %-26s %4s %15s %15s %15s %13s" % (
        "method", "traffic", "seeds", "ATT", "Mwt", "Cd", "Gini"))
    print("-" * 114)
    for method, tr in sorted(by_group):
        rs = by_group[(method, tr)]
        cells = []
        for k in ("ATT", "Mwt", "Cd", "Gini"):
            mu, sd = mean_std([r[k] for r in rs])
            if mu is None and k == "Mwt":
                cells.append("<90 (censored)")
                continue
            cells.append("  N/A" if mu is None else
                         ("%.3f±%.3f" % (mu, sd) if k == "Gini"
                          else "%.1f±%.1f" % (mu, sd)))
        print("%-38s %-26s %4d %15s %15s %15s %13s" % (
            method[:38], tr[:26], len(rs), cells[0], cells[1], cells[2], cells[3]))

    if a.sig:
        # Paired by scenario: the seeds of each scenario are averaged first, then
        # the per-scenario differences (reference - other) are tested.
        # Pooling all seeds into a two-sample test would let the differences
        # between scenarios (ATT ~180 on New York, ~345 on Hangzhou-2) swamp the
        # differences between methods.
        # The trailing scenario tag is also removed from the method name, so that the
        # runs of one method on different scenarios are paired
        # (e.g. sumo-base-CoLight-hz1 -> sumo-base-CoLight). Only the test uses this
        # merge; the aggregated table above keeps its groups.
        _TAGS = ("hz1", "hz2", "jn1", "jn2", "jn3", "ny163", "ing21")

        def _norm(m):
            for t in _TAGS:
                if m.endswith("-" + t):
                    return m[: -(len(t) + 1)]
            return m

        cell = {}
        for (method, tr), rs in by_group.items():
            nm = _norm(method)
            for k in ("ATT", "Mwt", "Cd", "Gini"):
                mu, _ = mean_std([r[k] for r in rs])
                if mu is not None:
                    cell[(nm, tr, k)] = mu
        methods = sorted(set(_norm(m) for m, _ in by_group))
        base = [m for m in methods if m == a.sig] or \
               [m for m in methods if a.sig in m]
        if len(base) != 1:
            print("\n!! --sig %s matches %d methods %s; exactly one is required"
                  % (a.sig, len(base), base))
        else:
            base = base[0]
            scens = sorted(set(t for m, t in by_group if _norm(m) == base))
            print("\n" + "=" * 114)
            print("Significance test: reference = %s, paired by scenario (%d scenarios)"
                  % (base, len(scens)))
            print("Both the paired t-test and the exact Wilcoxon signed-rank test are "
                  "reported; if they disagree, the evidence is insufficient.")
            print("=" * 114)
            print("%-30s %-6s %10s %10s %9s %9s %9s %s" % (
                "method", "metric", "reference", "other", "rel.diff",
                "t-test p", "W-test p", "verdict"))
            print("-" * 114)
            for m in methods:
                if m == base:
                    continue
                for k in ("ATT", "Mwt", "Cd", "Gini"):
                    pairs = [(cell[(base, t, k)], cell[(m, t, k)])
                             for t in scens
                             if (base, t, k) in cell and (m, t, k) in cell]
                    if len(pairs) < 3:
                        continue        # too few paired scenarios for a test
                    d = [x - y for x, y in pairs]      # reference - other
                    mb = sum(x for x, _ in pairs) / len(pairs)
                    mm = sum(y for _, y in pairs) / len(pairs)
                    pt, pw = _paired_t(d), _wilcoxon(d)
                    rel = (mb - mm) / mm * 100 if mm else float("nan")
                    sig_t = pt is not None and pt < 0.05
                    sig_w = pw is not None and pw < 0.05
                    if sig_t and sig_w:
                        # all these metrics are lower-is-better
                        note = "reference better" if mb < mm else "reference worse"
                    elif sig_t or sig_w:
                        note = "tests disagree"
                    else:
                        note = "no significant difference"
                    print("%-30s %-6s %10.1f %10.1f %8.1f%% %9s %9s %s" % (
                        m[:30], k, mb, mm, rel,
                        "N/A" if pt is None else "%.4f" % pt,
                        "N/A" if pw is None else "%.4f" % pw,
                        note))
            print("-" * 114)
            print("Note: %d pairs. The smallest two-sided Wilcoxon p is 0.031 for n=6 "
                  "and 0.25 for n=3; with few scenarios \"no significant difference\" "
                  "reflects the sample size, not evidence of equality." % len(scens))

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\nwritten to %s" % a.csv)


if __name__ == "__main__":
    main()
