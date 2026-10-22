"""
ssm_metrics.py
==============
Surrogate safety measures from the SUMO SSM device (Section V of the paper).

Two parts:
  * ``sumo_cmd_ssm_flags``: command-line switches that attach the SSM device to
    every vehicle; used by ``sumo_env.py`` when ``ENABLE_SSM`` is on
    (``run_eval.py -sumo`` enables it unless ``-no_ssm`` is given).
  * The metric computation used for the safety table and figure.

Definitions
-----------
1. With the SSM device on every vehicle, each encounter is logged by both
   vehicles (mirrored types 2<->3, 6<->7, 10<->11). Records of the same
   unordered vehicle pair whose time intervals overlap are merged into one
   encounter; its TTC (and PET) is the smaller of the two records.
2. The conflict class is the SUMO encounter type at the minimum TTC.
3. A severe conflict has minimum TTC < 1.5 s or PET < 1.0 s.
   RLC (rear-looking conflict) = severe rear-end + severe merging conflicts.
4. A hard-braking event is a maximal run of seconds with braking rate
   >= 3 m/s^2 in the per-second ``BRSpan`` of ``globalMeasures``.
5. Spillback: a lane whose queue back reaches 95 % of the lane length; we
   count onsets and the total duration (lane-seconds).
6. Exposure: demand (all SSM egos), inserted vehicles and completed trips;
   every count is also reported per 1000 of each exposure measure.

Input layout
------------
``--raw_dir`` must contain one directory per run::

    <raw_dir>/<method>/<flow>/seed<seed>/ssm.xml[.gz]
                                        /lane_queue.csv[.gz]
                                        /collisions.xml[.gz]   (optional)

These three files are written by ``run_eval.py -sumo`` into
``records/<out_memo>/<run>/test_round/round_<N>/``.

Usage::

    python -m d2marl_sumo.ssm_metrics --raw_dir safety_raw --out safety_out
"""
import argparse
import csv
import gzip
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path


# ==========================================================================
# SSM device configuration
# ==========================================================================
def sumo_cmd_ssm_flags(out_file, measure_range=50.0):
    """SUMO command-line switches that put an SSM device on every vehicle."""
    return [
        "--device.ssm.probability", "1.0",
        "--device.ssm.measures", "TTC DRAC PET BR SGAP TGAP",
        "--device.ssm.range", str(measure_range),
        "--device.ssm.file", out_file,
        "--device.ssm.trajectories", "false",
    ]


# ==========================================================================
# Metric definitions
# ==========================================================================
# SUMO encounter type codes, see
# https://sumo.dlr.de/docs/Simulation/Output/SSM_Device.html
REAR_END = {1, 2, 3, 18}
MERGING = {5, 6, 7, 8, 19}
CROSSING = {9, 10, 11, 12, 13, 14, 15, 16, 17}

TTC_SERIOUS = 1.5
TTC_LOOSE = 3.0
PET_SERIOUS = 1.0
BR_HARD = 3.0
SPILL_RATIO = 0.95
WAVE_MIN_GROWTH_M = 15.0   # about two vehicle lengths; ignores single-vehicle jumps
WAVE_MIN_SECONDS = 2.0
RUN_END = 3600.0


def _open(path, mode='rb'):
    """Open ``path`` or ``path.gz``, whichever exists."""
    path = Path(path)
    gz = path.with_name(path.name + '.gz')
    if gz.is_file():
        return gzip.open(str(gz), 'rt' if 't' in mode else 'rb')
    if path.is_file():
        return open(str(path), 'r' if 't' in mode else 'rb')
    raise FileNotFoundError(str(path) + '[.gz]')


def _exists(path):
    path = Path(path)
    return path.is_file() or path.with_name(path.name + '.gz').is_file()


def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _i(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def category(t):
    if t in REAR_END:
        return 'rear_end'
    if t in MERGING:
        return 'merging'
    if t in CROSSING:
        return 'crossing'
    if t == 111:
        return 'collision'
    return 'other'


def parse_ssm(path):
    directed = []
    vehicles = 0
    hard_events = 0
    hard_vehicles = 0
    inserted = completed = 0
    veh_seconds = 0.0
    with _open(path) as fh:
        for _, e in ET.iterparse(fh, events=('end',)):
            if e.tag == 'conflict':
                rec = dict(a=e.get('ego'), b=e.get('foe'), t0=_f(e.get('begin')), t1=_f(e.get('end')),
                           ttc=None, ttc_type=None, pet=None, pet_type=None)
                for c in e:
                    if c.tag == 'minTTC':
                        rec['ttc'], rec['ttc_type'] = _f(c.get('value')), _i(c.get('type'))
                    elif c.tag == 'PET':
                        rec['pet'], rec['pet_type'] = _f(c.get('value')), _i(c.get('type'))
                directed.append(rec)
                e.clear()
            elif e.tag == 'globalMeasures':
                vehicles += 1
                n_ev = 0
                for c in e:
                    if c.tag == 'timeSpan':
                        ts = c.get('values', '').split()
                        if ts:                                   # empty = never inserted
                            inserted += 1
                            t0, t1 = float(ts[0]), float(ts[-1])
                            veh_seconds += t1 - t0 + 1
                            completed += t1 < RUN_END - 2        # left the network before the end
                    elif c.tag == 'BRSpan':
                        prev = False
                        for v in c.get('values', '').split():
                            x = _f(v)
                            on = x is not None and x >= BR_HARD
                            if on and not prev:
                                n_ev += 1
                            prev = on
                hard_events += n_ev
                hard_vehicles += n_ev > 0
                e.clear()
    return directed, dict(demand=vehicles, inserted=inserted, completed=completed,
                          veh_hours=veh_seconds / 3600.0), hard_events, hard_vehicles


def merge_encounters(directed):
    """Merge mirrored records: same unordered vehicle pair with overlapping intervals."""
    by_pair = defaultdict(list)
    for r in directed:
        by_pair[tuple(sorted((r['a'], r['b'])))].append(r)
    events, unmatched = [], 0
    for recs in by_pair.values():
        recs.sort(key=lambda r: (r['t0'] or 0))
        groups = []
        for r in recs:
            g = next((g for g in groups if g['t0'] <= (r['t1'] or 0) and (r['t0'] or 0) <= g['t1']), None)
            if g is None:
                groups.append(dict(t0=r['t0'] or 0, t1=r['t1'] or 0, recs=[r]))
            else:
                g['t0'] = min(g['t0'], r['t0'] or 0); g['t1'] = max(g['t1'], r['t1'] or 0); g['recs'].append(r)
        for g in groups:
            unmatched += len(g['recs']) == 1
            ttcs = [(r['ttc'], r['ttc_type']) for r in g['recs'] if r['ttc'] is not None and r['ttc'] >= 0]
            pets = [(r['pet'], r['pet_type']) for r in g['recs'] if r['pet'] is not None and r['pet'] >= 0]
            ttc = min(ttcs) if ttcs else (None, None)
            pet = min(pets) if pets else (None, None)
            cats = {category(t) for _, t in ttcs} | {category(t) for _, t in pets}
            cat = ('collision' if 'collision' in cats else 'crossing' if 'crossing' in cats
                   else 'merging' if 'merging' in cats else 'rear_end' if 'rear_end' in cats else 'other')
            events.append(dict(ttc=ttc[0], pet=pet[0], cat=cat, n_records=len(g['recs'])))
    return events, unmatched


def parse_queue(path):
    series = defaultdict(list)
    length = {}
    with _open(path, 'rt') as fh:
        for r in csv.DictReader(fh):
            series[r['lane_id']].append((float(r['time']), float(r['queue_back_m'])))
            length[r['lane_id']] = float(r['lane_length_m'])
    spill_events = spill_seconds = 0
    speeds, max_ratio = [], 0.0
    for lane, pts in series.items():
        pts.sort()
        L = length[lane]
        prev = False
        for _, b in pts:
            on = L > 0 and b >= SPILL_RATIO * L
            spill_events += on and not prev
            spill_seconds += on
            prev = on
            if L > 0:
                max_ratio = max(max_ratio, b / L)
        # Non-decreasing stretches of the queue back: from the start of the
        # stretch to the first time it reaches its maximum.
        i = 0
        while i < len(pts) - 1:
            j = i
            while j + 1 < len(pts) and pts[j + 1][1] >= pts[j][1]:
                j += 1
            seg = pts[i:j + 1]
            top = max(b for _, b in seg)
            t_top = next(t for t, b in seg if b == top)
            growth, dur = top - seg[0][1], t_top - seg[0][0]
            if growth >= WAVE_MIN_GROWTH_M and dur >= WAVE_MIN_SECONDS:
                speeds.append(growth / dur)
            i = j + 1
    return dict(spill_events=spill_events, spill_lane_seconds=spill_seconds,
                wave_speed_median=statistics.median(speeds) if speeds else None,
                wave_episodes=len(speeds), max_queue_ratio=max_ratio)


def count_collisions(path):
    if not _exists(path):
        return 0
    n = 0
    with _open(path) as fh:
        for _, e in ET.iterparse(fh, events=('end',)):
            if e.tag == 'collision':
                n += 1
            e.clear()
    return n


def pct(v, p):
    if not v:
        return None
    s = sorted(v)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def one_run(args):
    method, flow, seed, d = args
    d = Path(d)
    directed, exposure, hard_ev, hard_veh = parse_ssm(d / 'ssm.xml')
    veh = exposure['demand']
    events, unmatched = merge_encounters(directed)
    ttc = [e['ttc'] for e in events if e['ttc'] is not None]
    pet = [e['pet'] for e in events if e['pet'] is not None]
    serious = [e for e in events if (e['ttc'] is not None and e['ttc'] < TTC_SERIOUS)
               or (e['pet'] is not None and e['pet'] < PET_SERIOUS)]
    loose = [e for e in events if e['ttc'] is not None and e['ttc'] < TTC_LOOSE]
    cat = lambda xs, c: sum(e['cat'] == c for e in xs)
    q = parse_queue(d / 'lane_queue.csv')
    k = 1000.0 / veh if veh else float('nan')
    kt = 1000.0 / exposure['completed'] if exposure['completed'] else float('nan')
    ki = 1000.0 / exposure['inserted'] if exposure['inserted'] else float('nan')
    row = dict(method=method, flow=flow, seed=seed, vehicles=veh, inserted=exposure['inserted'],
               completed=exposure['completed'], veh_hours=exposure['veh_hours'],
               ssm_records=len(directed), encounters=len(events), unmatched_records=unmatched,
               ttc_n=len(ttc), ttc_p5=pct(ttc, 5), ttc_p15=pct(ttc, 15), ttc_median=pct(ttc, 50),
               pet_n=len(pet), pet_p5=pct(pet, 5), pet_median=pct(pet, 50),
               conflicts=len(serious), conflicts_ttc3=len(loose),
               rear_end=cat(serious, 'rear_end'), crossing=cat(serious, 'crossing'),
               merging=cat(serious, 'merging'),
               rlc=cat(serious, 'rear_end') + cat(serious, 'merging'),
               hard_brake_events=hard_ev, hard_brake_vehicles=hard_veh,
               collisions=count_collisions(d / 'collisions.xml'), **q)
    for key in _BASE_RATES:
        row[key + '_per1k'] = row[key] * k            # per 1000 demand (incl. never inserted)
        row[key + '_per1k_ins'] = row[key] * ki       # per 1000 inserted vehicles
        row[key + '_per1k_trip'] = row[key] * kt      # per 1000 completed trips
    return row, dict(ttc=ttc, pet=pet)


_BASE_RATES = ['conflicts', 'conflicts_ttc3', 'rear_end', 'crossing', 'merging', 'rlc',
               'hard_brake_events', 'spill_events']
RATE_COLS = [k + suf for suf in ('_per1k', '_per1k_ins', '_per1k_trip') for k in _BASE_RATES]
LEVEL_COLS = ['inserted', 'completed', 'veh_hours', 'ttc_p5', 'ttc_p15', 'ttc_median', 'pet_p5', 'pet_median',
              'wave_speed_median', 'spill_lane_seconds', 'max_queue_ratio', 'collisions', 'vehicles',
              'encounters', 'conflicts', 'rear_end', 'crossing', 'merging', 'rlc', 'hard_brake_events',
              'spill_events']


def discover(raw_dir):
    """List (method, flow, seed, directory) for every run under ``raw_dir``."""
    tasks = []
    for d in sorted(Path(raw_dir).glob('*/*/seed*')):
        if d.is_dir() and _exists(d / 'ssm.xml') and _exists(d / 'lane_queue.csv'):
            tasks.append((d.parent.parent.name, d.parent.name, int(d.name[4:]), str(d)))
    if not tasks:
        raise FileNotFoundError('no <method>/<flow>/seed<N>/ssm.xml under ' + str(raw_dir))
    return tasks


def _mean_sd(vals):
    if not vals or any(v is None for v in vals):
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw_dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--jobs', type=int, default=6)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tasks = discover(a.raw_dir)
    with Pool(a.jobs) as pool:
        results = pool.map(one_run, tasks)
    rows = [r for r, _ in results]
    with (a.out / 'safety_per_seed.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)

    # method x flow: mean and sample standard deviation over seeds
    methods = sorted({r['method'] for r in rows})
    flows = sorted({r['flow'] for r in rows})
    agg = []
    for m in methods:
        for f in flows:
            rr = [r for r in rows if r['method'] == m and r['flow'] == f]
            if not rr:
                continue
            out = dict(method=m, flow=f, n_seeds=len(rr))
            for c in RATE_COLS + LEVEL_COLS:
                out[c + '_mean'], out[c + '_sd'] = _mean_sd([r[c] for r in rr])
            agg.append(out)
    with (a.out / 'safety_flow_seeds.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, list(agg[0])); w.writeheader(); w.writerows(agg)

    # method summary: equal-weight average over flows of the per-flow seed means
    summary = []
    for m in methods:
        rr = [x for x in agg if x['method'] == m]
        out = dict(method=m, n_flows=len(rr), n_runs=sum(x['n_seeds'] for x in rr))
        for c in RATE_COLS + LEVEL_COLS:
            vals = [x[c + '_mean'] for x in rr]
            out[c] = None if any(v is None for v in vals) else statistics.mean(vals)
        summary.append(out)
    with (a.out / 'safety_method_summary.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, list(summary[0])); w.writeheader(); w.writerows(summary)

    # pooled TTC / PET distributions per method (for the figure)
    dists = defaultdict(lambda: dict(ttc=[], pet=[]))
    for (m, _, _, _), (_, d) in zip(tasks, results):
        dists[m]['ttc'] += d['ttc']; dists[m]['pet'] += d['pet']
    (a.out / 'safety_distributions.json').write_text(json.dumps(dists))
    (a.out / 'definitions.json').write_text(json.dumps(dict(
        ttc_serious_s=TTC_SERIOUS, ttc_loose_s=TTC_LOOSE, pet_serious_s=PET_SERIOUS,
        hard_brake_mps2=BR_HARD, spill_ratio=SPILL_RATIO, wave_min_growth_m=WAVE_MIN_GROWTH_M,
        wave_min_seconds=WAVE_MIN_SECONDS, rear_end_types=sorted(REAR_END),
        merging_types=sorted(MERGING), crossing_types=sorted(CROSSING),
        rlc='serious conflicts approached from behind on a shared path: rear_end + merging',
        conflict='merged encounter with min TTC < 1.5 s or PET < 1.0 s',
        exposure=dict(demand='SSM globalMeasures egos, incl. vehicles never inserted (empty timeSpan)',
                      inserted='egos with non-empty timeSpan',
                      completed='last timeSpan < 3598 s (left the network)',
                      rates='_per1k = per 1000 demand; _per1k_ins = per 1000 inserted; '
                            '_per1k_trip = per 1000 completed trips')), indent=2))
    print('runs', len(rows), '->', a.out)


if __name__ == '__main__':
    main()
