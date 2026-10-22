# -*- coding: utf-8 -*-
"""Resume an interrupted run: infer the next round from model/<memo>/<traffic_file>_<stamp>/.

Shared by run_baseline.py and run_d2marl.py.
"""
import glob
import os
import re


def find_resume_point(memo, traffic_file):
    """Return (stamp, start_round), or (None, 0) if there is nothing to resume.

    A memo can hold several time-stamped run directories (one per submission).
    The one with the **most progress** is chosen, not the newest: a resubmission
    that was interrupted after a few minutes leaves an almost empty new directory,
    and choosing it would discard the progress of the earlier one.

    Round N loads the weights of round N-1, so the resume point is the last round
    for which the weights of all intersections were saved, plus one. A round that
    was saved only partly (e.g. the job was killed) is rolled back, otherwise
    intersections would load weights of different rounds.
    """

    def progress(mdir):
        per_round = {}
        try:
            names = os.listdir(mdir)
        except OSError:
            return 0
        for f in names:
            m = re.match(r"round_(\d+)_inter_(\d+)", f)
            if m:
                per_round.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
        if not per_round:
            return 0
        n_inter = max(len(v) for v in per_round.values())
        complete = [r for r, v in per_round.items() if len(v) == n_inter]
        return (max(complete) + 1) if complete else 0

    cands = []
    for mdir in sorted(glob.glob(os.path.join("model", memo, traffic_file + "_*"))):
        stamp = os.path.basename(mdir)[len(traffic_file) + 1:]
        if not os.path.isdir(os.path.join("records", memo, traffic_file + "_" + stamp)):
            continue
        cands.append((progress(mdir), stamp))
    if not cands:
        return None, 0

    cands.sort()
    best_round, best_stamp = cands[-1]
    if len(cands) > 1:
        print("==> -resume: %d run directories under %s, chose %s by progress (round %d)"
              % (memo, len(cands), best_stamp, best_round))
        for r, st in cands[:-1]:
            print("      ignoring %s (only %d rounds)" % (st, r))
    if best_round <= 0:
        return None, 0          # only empty directories: start a new run
    return best_stamp, best_round
