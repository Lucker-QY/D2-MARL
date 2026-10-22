"""Divergence rule for replacing a training run (supplementary material, Table S15).

A run is divergent if, among the three seeds of a cell, it is a Dixon Q outlier
(n = 3, alpha = 0.05, Q > 0.970) in the worse direction and its value exceeds the
median of the other two seeds by more than 50 %. A_TT and M_wt are checked separately.
Replacement seeds are tried in the order 20, 200, 2000, 20000, 200000; the first one
for which none of the three seeds is divergent is used, and all runs are reported.

The rule is applied in the same way to D2-MARL, its ablations and every learned baseline.

Usage:
    python analysis/divergence_rule.py per_seed.csv --keys method flow --metrics ATT Mwt_exact

per_seed.csv has one row per run with the key columns, a 'seed' column and the metrics
(for example the output of `python -m d2marl_sumo.collect_results`).
"""
import argparse

import numpy as np
import pandas as pd

Q_CRIT = 0.970        # Dixon Q, n = 3, alpha = 0.05
FLOOR = 0.5           # at least 50 % worse than the median of the other two seeds
REPLACEMENT_ORDER = [20, 200, 2000, 20000, 200000]


def divergent_index(values):
    """Index of the divergent run among three values (larger is worse), or None."""
    v = np.asarray(values, float)
    if v.size != 3:
        return None
    order = np.argsort(v)
    s = v[order]
    spread = s[-1] - s[0]
    if spread == 0:
        return None
    q_high = (s[2] - s[1]) / spread
    i = order[2]
    others = np.median(np.delete(v, i))
    return int(i) if q_high > Q_CRIT and v[i] > (1 + FLOOR) * others else None


def scan(df, keys, metrics):
    rows = []
    for cell, g in df.groupby(keys):
        g = g.reset_index(drop=True)
        if len(g) != 3:
            continue
        for m in metrics:
            i = divergent_index(g[m])
            if i is not None:
                rows.append(dict(cell=cell, metric=m, seed=int(g.seed[i]), value=float(g[m][i]),
                                 others=[float(x) for j, x in enumerate(g[m]) if j != i]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv')
    ap.add_argument('--keys', nargs='+', required=True, help='columns that define a cell')
    ap.add_argument('--metrics', nargs='+', default=['ATT', 'Mwt_exact'])
    a = ap.parse_args()
    out = scan(pd.read_csv(a.csv), a.keys, a.metrics)
    if out.empty:
        print('no divergent run')
    else:
        print(out.to_string(index=False))
        print('replacement seeds, in this order:', REPLACEMENT_ORDER)


if __name__ == '__main__':
    main()
