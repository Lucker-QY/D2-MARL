"""Safety figure: (a) TTC and (b) PET cumulative encounter rates, (c) severe conflicts vs. spillback.

Input: the output directory of ``python -m d2marl_sumo.ssm_metrics``.
Panels (a) and (b) show, per 10^3 completed trips, the number of encounters whose
TTC / PET is <= x (completed trips = ``completed`` x ``n_runs`` from the method
summary). Normalising by trips rather than by each method's own encounter count keeps
the curves on the same scale as the rates in the summary table; a cumulative share
would be distorted when methods produce very different numbers of encounters.
Dashed vertical lines mark the severe-conflict thresholds (TTC 1.5 s, PET 1.0 s).
The six learned baselines are drawn as a grey min/max band; the rule-based
controllers and D2-MARL are drawn as individual lines.

Usage::

    python analysis/plot_safety.py --data safety_out --out figures
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OURS = 'D2MARL'
LEARNED = ['AdvancedColight', 'EfficientColight', 'CoLight', 'Attend', 'MPLight', 'PressLight']
CLASSIC = {  # label, colour, line style
    'Fixedtime': ('Fixed-time', '#eb6834', (0, (5, 2))),
    'Actuated': ('ASC', '#1baf7a', (0, (1, 1.5))),
    'MaxPressure': ('MaxPressure', '#4a3aa7', (0, (5, 1.5, 1, 1.5))),
}
RANDOM = ('Random', '#8a8985', (0, (1, 2.5)))
BLUE = '#2a78d6'
GREY_BAND = '#c4c3bd'
INK, INK2 = '#0b0b0b', '#52514e'
LABEL = {'D2MARL': 'D$^2$-MARL', 'AdvancedColight': 'Adv-CoLight', 'EfficientColight': 'Eff-CoLight',
         'CoLight': 'CoLight', 'Attend': 'AttendLight', 'MPLight': 'MPLight', 'PressLight': 'PressLight',
         'MaxPressure': 'MaxPressure', 'Fixedtime': 'Fixed-time', 'Actuated': 'ASC', 'Random': 'Random'}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'axes.edgecolor': '#9a9994', 'axes.linewidth': 0.6, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.labelcolor': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def cum_rate(values, grid, trips):
    """Encounters with value <= grid, per 10^3 completed trips."""
    v = np.sort(np.asarray(values, float))
    return np.searchsorted(v, grid, side='right') / trips * 1000


def ecdf_panel(ax, dists, trips, key, grid, xlabel, title, severe, logy=False, skip=()):
    curves = np.array([cum_rate(dists[m][key], grid, trips[m]) for m in LEARNED if m in dists and dists[m][key]])
    if len(curves):
        # the band is narrow, so outline it with a thin edge to keep it visible under the lines
        ax.fill_between(grid, curves.min(0), curves.max(0), color=GREY_BAND, lw=0.4, edgecolor='#9a9994',
                        label='Other learned (%d, range)' % len(curves))
    for m, (name, color, ls) in CLASSIC.items():
        if m in skip or m not in dists or not dists[m][key]:
            continue
        ax.plot(grid, cum_rate(dists[m][key], grid, trips[m]), color=color, lw=0.8, ls=ls, label=name)
    if 'Random' in dists and dists['Random'][key]:
        ax.plot(grid, cum_rate(dists['Random'][key], grid, trips['Random']), color=RANDOM[1], lw=0.8,
                ls=RANDOM[2], label='Random')
    ax.plot(grid, cum_rate(dists[OURS][key], grid, trips[OURS]), color=BLUE, lw=0.9, label=LABEL[OURS])
    ax.axvline(severe, color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=0)
    ax.set_xlim(grid[0], grid[-1])
    if logy:
        ax.set_yscale('log')
    else:
        ax.set_ylim(0, None)
    ax.text(severe, 1.0, ' severe', transform=ax.get_xaxis_transform(), fontsize=6, color=INK2,
            ha='right' if severe > grid[0] + 0.3 * (grid[-1] - grid[0]) else 'left', va='top')
    ax.set_xlabel(xlabel); ax.set_ylabel('Encounters $\\leq x$ / $10^3$ trips')
    ax.set_title(title, loc='left', color=INK)
    ax.grid(color='#ecebe7', lw=0.5); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, required=True, help='output directory of d2marl_sumo.ssm_metrics')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dists = json.loads((a.data / 'safety_distributions.json').read_text())
    summ = {r['method']: r for r in csv.DictReader((a.data / 'safety_method_summary.csv').open())}
    # total completed trips over all runs of a method (summary 'completed' is a per-run mean)
    trips = {m: float(r['completed']) * int(r['n_runs']) for m, r in summ.items()}
    defs_path = a.data / 'definitions.json'
    defs = json.loads(defs_path.read_text()) if defs_path.exists() else {}
    ttc_severe = float(defs.get('ttc_serious_s', 1.5))
    pet_severe = float(defs.get('pet_serious_s', 1.0))

    # single-column layout: 2 x 2 grid, legend in the fourth cell
    fig, ax4 = plt.subplots(2, 2, figsize=(3.5, 3.6), gridspec_kw=dict(wspace=0.55, hspace=0.62))
    axes = [ax4[0, 0], ax4[0, 1], ax4[1, 0]]
    ecdf_panel(axes[0], dists, trips, 'ttc', np.linspace(1.0, 3.0, 401), 'Minimum TTC $x$ (s)',
               '(a) TTC', severe=ttc_severe, logy=True)
    # Fixed-time is deterministic and yields only a handful of PET records, too few for a
    # reliable curve, so it is left out of the PET panel; its legend entry comes from the TTC panel.
    ecdf_panel(axes[1], dists, trips, 'pet', np.linspace(0.0, 12.0, 401), 'PET $x$ (s)',
               '(b) PET (crossing)', severe=pet_severe, skip=('Fixedtime',))
    h, l = axes[0].get_legend_handles_labels()
    # the legend shows the line style of (a)(b) together with the marker of (c)
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    mk = {'Other learned (6, range)': dict(color='#a9a8a2'), 'Fixed-time': dict(color=CLASSIC['Fixedtime'][1]),
          'ASC': dict(color=CLASSIC['Actuated'][1]), 'MaxPressure': dict(color=CLASSIC['MaxPressure'][1]),
          'Random': dict(markerfacecolor='white', markeredgecolor=RANDOM[1]), LABEL[OURS]: dict(color=BLUE)}
    h = [(hh, Line2D([], [], ls='none', marker='o', ms=4, **mk.get(ll, {}))) for hh, ll in zip(h, l)]
    ax4[1, 1].axis('off')
    ax4[1, 1].legend(h, l, loc='center', frameon=False, handlelength=3.2,
                     handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)})

    ax = axes[2]
    xy = lambda r: (float(r['spill_lane_seconds']), float(r['conflicts_per1k_trip']))
    for m in LEARNED:
        if m in summ:
            ax.scatter(*xy(summ[m]), s=22, color='#a9a8a2', edgecolor='white', lw=0.6, zorder=2)
    for m, (name, color, _) in CLASSIC.items():
        if m in summ:
            ax.scatter(*xy(summ[m]), s=30, color=color, edgecolor='white', lw=0.6, zorder=3)
    if 'Random' in summ:   # hollow marker, distinct from the grey learned baselines
        ax.scatter(*xy(summ['Random']), s=30, facecolor='white', edgecolor=RANDOM[1], lw=1.0, zorder=3)
    ax.scatter(*xy(summ[OURS]), s=40, color=BLUE, edgecolor='white', lw=0.8, zorder=4)
    # colours as in the legend; lower left is better
    ax.annotate('better', xy=(0.06, 0.05), xytext=(0.30, 0.30), xycoords='axes fraction',
                textcoords='axes fraction', fontsize=6.5, color=INK2, ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color=INK2, lw=0.7))
    ytop = 1.1 * max(xy(r)[1] for r in summ.values())
    ax.set_xscale('log')
    ax.set_ylim(0, ytop)
    ax.set_xlabel('Spillback (lane-s, log)')
    ax.set_ylabel('Severe / $10^3$ trips')
    ax.set_title('(c) Conflicts vs. spillback', loc='left', color=INK)
    ax.grid(color='#ecebe7', lw=0.5); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    for ext in ('pdf', 'png'):
        fig.savefig(a.out / ('safety.' + ext), bbox_inches='tight', dpi=300)
    # the plotted numbers as a table
    with (a.out / 'safety_figure_data.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'spill_lane_seconds', 'conflicts_per1k_trip', 'ttc_median', 'pet_median',
                    'ttc_n_pooled', 'pet_n_pooled'])
        for m in [OURS] + LEARNED + list(CLASSIC) + ['Random']:
            if m in summ:
                r = summ[m]
                w.writerow([m, r['spill_lane_seconds'], r['conflicts_per1k_trip'], r['ttc_median'],
                            r['pet_median'], len(dists[m]['ttc']), len(dists[m]['pet'])])
    print(a.out)


if __name__ == '__main__':
    main()
