"""
robustness.py
=============
Robustness to noisy observations (Section VI of the paper).

Robustness is evaluated under
    (a) noisy observations (this module) and
    (b) shifted demand: zero-shot transfer, which needs no extra code (run_eval.py
        with a -memo trained on one traffic file and a different -traffic_file).


Method
----
Evaluation only: stored weights are loaded and the observation is perturbed before
it enters the network; no retraining. As in inference_cost.py, choose_action of the
agent class is wrapped and the state is modified there. The environment code is
untouched and all methods (including rule-based baselines) get the same perturbation.

Which features are perturbed
------------
Only detector quantities. Not perturbed:

    cur_phase          known to the controller, not sensed
    time_this_phase    internal timer of the controller
    adjacency_matrix   static network topology

Everything else (traffic_movement_pressure_* / lane_enter_running_part / ...) counts
as sensed. An exclusion list is used rather than an inclusion list because the
baselines use different feature names (pressure_queue / pressure_queue_efficient /
pressure_num ...); an inclusion list could miss a feature of some method and
perturb the methods unequally.

Two kinds of noise
--------
    sigma   relative Gaussian: x <- x * (1 + N(0, sigma)), detector calibration error
    dropout per-movement loss: a reading is set to 0 with probability p, detector failure

Dropout is closer to practice (detector failures are more common than drift) and
creates a **systematic loss of information**.

Queue and waiting quantities are non-negative and clipped at 0 after perturbation;
up/downstream pressure differences keep their sign. With zero noise the observation
is passed unchanged and no random numbers are drawn.

Usage
----
    python -m d2marl_sumo.robustness -mod D2MARL -memo sumo-hz1-seed10 \\
        -hangzhou -traffic_file anon_4_4_hangzhou_real.json \\
        -sigma 0.2 -dropout 0.0 -out_tag s020-d000
"""

import os
import sys
import argparse

import numpy as np


# internal controller state and static topology, not sensed: never perturbed
_NO_NOISE_KEYS = frozenset([
    "cur_phase",
    "time_this_phase",
    "adjacency_matrix",
    "adjacency_matrix_lane",
])

# in SUMO these features are up/downstream differences and may be negative; do not clip
_SIGNED_NOISE_KEYS = frozenset([
    "delta_lane_num_vehicle",
    "traffic_movement_pressure_queue",
    "traffic_movement_pressure_queue_efficient",
    "traffic_movement_pressure_num",
])

_RNG = None
_STATS = {"calls": 0, "fields": 0}


def _perturb_value(v, sigma, dropout, allow_negative=False):
    """Perturb the sensed features; signed features keep negative values; zero noise returns them unchanged."""
    if sigma == 0.0 and dropout == 0.0:
        return v
    try:
        arr = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return v
    if arr.size == 0:
        return v
    out = arr.copy()
    if sigma > 0:
        out = out * (1.0 + _RNG.normal(0.0, sigma, size=out.shape))
    if dropout > 0:
        # dropout per element (i.e. per movement), not for a whole feature:
        # in practice single detectors fail, not all detectors of an intersection at once
        mask = _RNG.random_sample(out.shape) < dropout
        out = np.where(mask, 0.0, out)
    if not allow_negative:
        out = np.maximum(out, 0.0)
    _STATS["fields"] += 1
    return out.tolist() if isinstance(v, list) else out


def _perturb_state(st, sigma, dropout):
    """state is a dict (rule-based agents) or a list of dicts (CoLight family)."""
    if isinstance(st, (list, tuple)):
        return type(st)(_perturb_state(x, sigma, dropout) for x in st)
    if not isinstance(st, dict):
        return st
    out = dict(st)
    for k, v in st.items():
        if k in _NO_NOISE_KEYS:
            continue
        out[k] = _perturb_value(v, sigma, dropout,
                                allow_negative=(k in _SIGNED_NOISE_KEYS))
    return out


def _install(sigma, dropout, seed):
    global _RNG
    _RNG = np.random.RandomState(seed + 991)   # separate from the environment seed
    from utils.config import DIC_AGENTS
    for cls in set(DIC_AGENTS.values()):
        if getattr(cls, "_noise_patched", False):
            continue
        orig = cls.choose_action

        def make(orig_fn):
            def noisy(self, count, states, *a, **kw):
                _STATS["calls"] += 1
                return orig_fn(self, count,
                               _perturb_state(states, sigma, dropout),
                               *a, **kw)
            return noisy

        cls.choose_action = make(orig)
        cls._noise_patched = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-mod", required=True)
    ap.add_argument("-memo", default=None)
    ap.add_argument("-round", type=int, default=None)
    ap.add_argument("-traffic_file", default=None)
    ap.add_argument("-seed", type=int, default=10)
    ap.add_argument("-sigma", type=float, default=0.0,
                    help="std of the relative Gaussian noise; 0 = none")
    ap.add_argument("-dropout", type=float, default=0.0,
                    help="per-movement dropout probability; 0 = none")
    ap.add_argument("-out_tag", default=None,
                    help="tag added to out_memo, to group the results by noise level")
    ap.add_argument("-out_prefix", default="rob",
                    help="out_memo prefix; use another prefix to avoid overwriting earlier results")
    from run_baseline import SCENARIOS
    for k in SCENARIOS:
        ap.add_argument("-" + k, action="store_true", default=False)
    a = ap.parse_args()

    if a.sigma < 0 or a.dropout < 0 or a.dropout > 1:
        raise ValueError("sigma must be >= 0 and dropout in [0, 1]")

    _install(a.sigma, a.dropout, a.seed)

    tag = a.out_tag or ("s%03d-d%03d" % (round(a.sigma * 100),
                                         round(a.dropout * 100)))
    out_memo = "%s-%s-%s-seed%d" % (a.out_prefix, a.mod, tag, a.seed)

    argv = ["run_eval.py", "-mod", a.mod, "-seed", str(a.seed),
            "-no_ssm", "-sumo", "-out_memo", out_memo]
    if a.memo:
        argv += ["-memo", a.memo]
    if a.round is not None:
        argv += ["-round", str(a.round)]
    if a.traffic_file:
        argv += ["-traffic_file", a.traffic_file]
    for k in SCENARIOS:
        if getattr(a, k):
            argv.append("-" + k)

    print("=" * 62)
    print("Observation noise:  %s   sigma=%.3f  dropout=%.3f" %
          (a.mod, a.sigma, a.dropout))
    print("  out_memo = %s" % out_memo)
    print("  features not perturbed: %s" % ", ".join(sorted(_NO_NOISE_KEYS)))
    print("=" * 62)

    import run_eval
    sys.argv = argv
    try:
        run_eval.main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise

    print("\nperturbation: choose_action called %d times, %d feature fields perturbed"
          % (_STATS["calls"], _STATS["fields"]))
    if _STATS["calls"] == 0:
        raise RuntimeError("choose_action was never called; the noise evaluation is invalid")
    elif (a.sigma > 0 or a.dropout > 0) and _STATS["fields"] == 0:
        raise RuntimeError("no feature was perturbed; check the state features of this method")


if __name__ == "__main__":
    main()
