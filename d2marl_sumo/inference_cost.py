"""
inference_cost.py
=================
Computational complexity and inference time.

Measured for D2-MARL and all baselines with the same procedure:

Measurement
--------
No synthetic states are built: they would have to reproduce every feature that
convert_state_to_input expects (cur_phase / traffic_movement_pressure_* /
lane_enter_running_part / adjacency_matrix ...), and any mismatch would distort the cost.

Instead choose_action of the agent class is wrapped with a timer and a **real SUMO
simulation** is run; only the wrapped time is counted. The states are therefore
exactly those of a real evaluation.

Three quantities:
  * wall-clock time per decision (mean / p50 / p95); one choose_action covers all intersections
  * time per intersection and decision, comparable across network sizes
  * number of trainable parameters, the static side of complexity

Usage
----
    python -m d2marl_sumo.inference_cost -mod D2MARL -memo d2marl-sumo-hz-seed10 \\
        -hangzhou -traffic_file anon_4_4_hangzhou_real.json -steps 600
    python -m d2marl_sumo.inference_cost -mod Fixedtime -hangzhou -steps 600

--steps is in simulation seconds, not decisions: one decision every MIN_ACTION_TIME
seconds, i.e. about 60 decisions in 600 s, enough for stable quantiles.
"""

import os
import sys
import json
import time
import argparse

import numpy as np


# ---- timer: patched onto the agent class, not the instance ----------------
_TIMINGS = []          # duration of every choose_action call (s)
_AGENT_REF = {}        # one agent instance, for counting parameters


def _install_timer():
    """Replace choose_action of every class in DIC_AGENTS with a timed version.

    The class is patched rather than the instance: agents are created inside
    Generator.__init__ and are not reachable from outside, while DIC_AGENTS maps
    names to classes; patching the class treats all models (including rule-based) alike.
    """
    from utils.config import DIC_AGENTS
    for name, cls in DIC_AGENTS.items():
        if getattr(cls, "_cost_patched", False):
            continue
        orig = cls.choose_action

        def make(orig_fn):
            def timed(self, *args, **kwargs):
                t0 = time.perf_counter()
                out = orig_fn(self, *args, **kwargs)
                _TIMINGS.append(time.perf_counter() - t0)
                _AGENT_REF.setdefault("agent", self)
                return out
            return timed

        cls.choose_action = make(orig)
        cls._cost_patched = True


def _count_params(agent):
    """Number of trainable parameters; 0 for rule-based agents without a network."""
    for attr in ("q_network", "model", "network"):
        net = getattr(agent, attr, None)
        if net is not None and hasattr(net, "count_params"):
            try:
                return int(np.sum([np.prod(v.shape) for v in net.trainable_weights]))
            except Exception:
                try:
                    return int(net.count_params())
                except Exception:
                    pass
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-mod", required=True)
    ap.add_argument("-memo", default=None, help="memo of the training run; not needed for rule-based methods")
    ap.add_argument("-round", type=int, default=None)
    ap.add_argument("-source_run", default=None, help="run directory under records/<memo> if there are several")
    ap.add_argument("-traffic_file", default=None)
    ap.add_argument("-seed", type=int, default=10)
    ap.add_argument("-steps", type=int, default=600,
                    help="simulation seconds (not decisions)")
    ap.add_argument("-warmup", type=int, default=5,
                    help="discard the first N decisions: the first call includes graph building "
                         "and memory allocation and is not steady-state inference")
    ap.add_argument("-out", default=None, help="write the result as JSON")
    # network switches as in run_baseline (-hangzhou / -jinan / -newyork163 ...)
    from run_baseline import SCENARIOS
    for k in SCENARIOS:
        ap.add_argument("-" + k, action="store_true", default=False)
    a = ap.parse_args()

    _install_timer()

    # reuse the whole evaluation path of run_eval (network, weights, SSM off)
    argv = ["run_eval.py", "-mod", a.mod, "-seed", str(a.seed), "-no_ssm", "-sumo"]
    if a.memo:
        argv += ["-memo", a.memo]
    if a.round is not None:
        argv += ["-round", str(a.round)]
    if a.traffic_file:
        argv += ["-traffic_file", a.traffic_file]
    for k in SCENARIOS:
        if getattr(a, k):
            argv.append("-" + k)
    if a.source_run:
        argv += ["-source_run", a.source_run]
    # output prefix (D2MARL_COST_PREFIX): run_eval refuses to overwrite an existing output memo
    argv += ["-out_memo", "%s-%s" % (os.environ.get("D2MARL_COST_PREFIX", "cost"), a.memo or a.mod)]

    # RUN_COUNTS sets the simulation length; run_eval uses 3600 by default, here
    # it is set from -steps instead of simulating a whole hour.
    os.environ["D2MARL_RUN_COUNTS"] = str(a.steps)

    import run_eval
    sys.argv = argv
    t_wall = time.perf_counter()
    try:
        run_eval.main()
    except SystemExit:
        pass
    wall = time.perf_counter() - t_wall

    if len(_TIMINGS) <= a.warmup:
        print("only %d decisions recorded, too few for statistics (warmup=%d); "
              "increase -steps." % (len(_TIMINGS), a.warmup))
        return

    t = np.array(_TIMINGS[a.warmup:]) * 1000.0      # milliseconds
    agent = _AGENT_REF.get("agent")
    params = _count_params(agent) if agent is not None else 0

    n_inter = getattr(agent, "num_agents", None) or \
        getattr(getattr(agent, "dic_traffic_env_conf", {}), "get", lambda *_: None)(
            "NUM_INTERSECTIONS") or 1
    try:
        n_inter = int(agent.dic_traffic_env_conf["NUM_INTERSECTIONS"])
    except Exception:
        pass

    res = dict(
        method=a.mod, memo=a.memo, traffic=a.traffic_file,
        n_decisions=int(t.size), n_intersections=int(n_inter),
        trainable_params=int(params),
        latency_ms_mean=float(t.mean()),
        latency_ms_p50=float(np.percentile(t, 50)),
        latency_ms_p95=float(np.percentile(t, 95)),
        latency_ms_max=float(t.max()),
        per_inter_ms_mean=float(t.mean() / max(n_inter, 1)),
        sim_seconds=a.steps,
        wall_seconds_total=float(wall),
    )

    print("\n" + "=" * 62)
    print("Inference cost:  %s" % a.mod)
    print("=" * 62)
    print("  intersections            : %d" % res["n_intersections"])
    print("  trainable parameters     : %d" % res["trainable_params"])
    print("  decisions (after warmup) : %d" % res["n_decisions"])
    print("  per decision (ms)        : mean %.3f  p50 %.3f  p95 %.3f  max %.3f"
          % (res["latency_ms_mean"], res["latency_ms_p50"],
             res["latency_ms_p95"], res["latency_ms_max"]))
    print("  per intersection (ms)    : %.4f" % res["per_inter_ms_mean"])
    print("  wall clock for %ds       : %.1fs" % (a.steps, res["wall_seconds_total"]))
    print("=" * 62)

    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print("written to %s" % a.out)


if __name__ == "__main__":
    main()
