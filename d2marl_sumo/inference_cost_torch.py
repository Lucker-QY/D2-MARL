# -*- coding: utf-8 -*-
"""Inference cost of the PyTorch baselines (CoordLight / LLMLight).

inference_cost.py only covers the TF agents: it patches choose_action of the classes
in DIC_AGENTS, and neither method goes through that path:

  CoordLight  calls self.network.forward(...) directly in its own Tester,
              without DIC_AGENTS.
  LLMLight    choose_action is a table lookup; the real cost is in the class method
              prepare_batch (all prompts of a step sent to vLLM as one batch).
              Timing choose_action would report almost zero latency.

The timer is therefore attached to the call that actually runs the inference; the
JSON fields are those of inference_cost.py, so cost_summary merges all three.

Usage:
  python -m d2marl_sumo.inference_cost_torch coordlight \\
      -model_path /path/cl_runs/SUMO_CoordLight_hz1_seed10 \\
      -traffic_file anon_4_4_hangzhou_real.json -tag hz1 \\
      -out cost_json_v2/hz1_CoordLight.json

  python -m d2marl_sumo.inference_cost_torch llmlight -tag hz1 \\
      -out cost_json_v2/hz1_LLMLight.json -- <arguments of run_baseline.py>
      (-single_process is required, otherwise the simulation runs in a child process
       and the timing patch has no effect)
"""

import os
import sys
import json
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

_T = []          # duration of every inference call (s)


def _summarize(method, tag, n_inter, params, warmup, extra):
    if len(_T) <= warmup:
        raise SystemExit("only %d inference calls recorded, too few for statistics (warmup=%d); "
                         "increase the simulation length." % (len(_T), warmup))
    t = np.array(_T[warmup:]) * 1000.0
    n = max(int(n_inter), 1)
    res = dict(
        method=method, tag=tag,
        n_decisions=int(t.size), n_intersections=int(n_inter),
        trainable_params=int(params),
        latency_ms_mean=float(t.mean()),
        latency_ms_p50=float(np.percentile(t, 50)),
        latency_ms_p95=float(np.percentile(t, 95)),
        latency_ms_max=float(t.max()),
        per_inter_ms_mean=float(t.mean() / n),
    )
    res.update(extra)
    return res


def _dump(res, out):
    print("\n" + "=" * 62)
    print("Inference cost:  %s  (%s)" % (res["method"], res["tag"]))
    print("=" * 62)
    print("  intersections            : %d" % res["n_intersections"])
    print("  decisions (after warmup) : %d" % res["n_decisions"])
    print("  per decision (ms)        : p50 %.3f  mean %.3f  p95 %.3f  max %.3f"
          % (res["latency_ms_p50"], res["latency_ms_mean"],
             res["latency_ms_p95"], res["latency_ms_max"]))
    print("  device                   : %s" % res.get("device", "?"))
    print("=" * 62)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print("written to %s" % out)


# ------------------------------------------------------------------ CoordLight
def run_coordlight(a):
    sys.path.insert(0, ROOT)
    from coordlight_sumo import run_coordlight as rc

    # Tester reads these environment variables through Parameters at import time, so
    # they are set before Tester is imported for patching; same values as rc.main().
    stem = rc.scenario_stem(a.sumo_dir, a.traffic_file)
    os.environ["CL_SIMULATOR"] = "sumo"
    os.environ["CL_SUMO_NET"] = os.path.abspath(stem + ".net.xml")
    os.environ["CL_SUMO_ROU"] = os.path.abspath(stem + ".repaired.rou.xml")
    os.environ["CL_HORIZON"] = str(a.horizon)
    os.environ["CL_ENABLE_SSM"] = "0"

    cwd = os.getcwd()
    sys.path.insert(0, rc.CL_DIR)
    os.chdir(rc.CL_DIR)
    try:
        import Tester as tester_mod       # the import in rc.main() gets the same module
    finally:
        os.chdir(cwd)

    info = {}
    orig_run = tester_mod.Tester.run_episode_single_threaded

    def patched(self, *args, **kwargs):
        net = self.network
        if not getattr(net, "_cost_wrapped", False):
            fwd = net.forward

            def timed(*x, **y):
                t0 = time.perf_counter()
                out = fwd(*x, **y)
                _T.append(time.perf_counter() - t0)
                return out

            net.forward = timed           # Tester calls self.network.forward; the instance attribute wins
            net._cost_wrapped = True
            info["params"] = sum(p.numel() for p in net.parameters() if p.requires_grad)
            info["n_inter"] = len(getattr(self.env, "rl_agent_id", []) or [])
            info["device"] = str(getattr(self, "device", "cpu"))
        return orig_run(self, *args, **kwargs)

    tester_mod.Tester.run_episode_single_threaded = patched

    sys.argv = ["run_coordlight.py",
                "-model_path", a.model_path,
                "-traffic_file", a.traffic_file,
                "-seed", str(a.seed),
                "-out_memo", "%s-CoordLight-%s-seed%d" % (os.environ.get("D2MARL_COST_PREFIX", "cost"), a.tag, a.seed),
                "-sumo_dir", a.sumo_dir,
                "-records", a.records,
                "-horizon", str(a.horizon),
                "-n_eval", "1"]
    try:
        rc.main()
    except SystemExit:
        pass

    res = _summarize("CoordLight", a.tag, info.get("n_inter", 0),
                     info.get("params", 0), a.warmup,
                     dict(device=info.get("device", "cpu"), framework="pytorch",
                          sim_seconds=a.horizon))
    _dump(res, a.out)


# ------------------------------------------------------------------ LLMLight
def run_llmlight(a, rest):
    if "-single_process" not in rest:
        raise SystemExit("LLMLight timing needs -single_process: with multiprocessing the "
                         "simulation runs in a child process and the class patch is not applied.")
    sys.path.insert(0, ROOT)
    os.environ["D2MARL_RUN_COUNTS"] = str(a.steps)

    from models.llmlight_agent import LLMLightAgent
    raw = LLMLightAgent.__dict__["prepare_batch"]
    fn = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw
    info = {}

    def timed(cls, count, state_list, conf):
        t0 = time.perf_counter()
        out = fn(cls, count, state_list, conf)
        _T.append(time.perf_counter() - t0)
        if "n_inter" not in info:
            try:
                info["n_inter"] = len(state_list)
            except TypeError:
                pass
        return out

    LLMLightAgent.prepare_batch = classmethod(timed)

    import run_baseline
    sys.argv = ["run_baseline.py"] + rest
    try:
        run_baseline.main(run_baseline.parse_args())
    except SystemExit:
        pass

    device = "gpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
    except Exception:
        pass
    # -1 = not the parameter count of a trainable network: LLMLight uses a 13B language model
    res = _summarize("LLMLight", a.tag, info.get("n_inter", 0), -1, a.warmup,
                     dict(device=device, framework="vllm", sim_seconds=a.steps))
    _dump(res, a.out)


def main():
    argv = sys.argv[1:]
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]

    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["coordlight", "llmlight"])
    ap.add_argument("-tag", required=True, help="scenario tag, e.g. hz1 / ny163")
    ap.add_argument("-out", default=None)
    ap.add_argument("-warmup", type=int, default=20)
    ap.add_argument("-seed", type=int, default=10)
    # CoordLight
    ap.add_argument("-model_path", default=None)
    ap.add_argument("-traffic_file", default=None)
    ap.add_argument("-sumo_dir", default=os.path.join(ROOT, "sumo_scenarios"))
    ap.add_argument("-records", default=os.path.join(ROOT, "records"))
    ap.add_argument("-horizon", type=int, default=3600)
    # LLMLight
    ap.add_argument("-steps", type=int, default=1800)
    a = ap.parse_args(argv)

    if a.which == "coordlight":
        if not (a.model_path and a.traffic_file):
            raise SystemExit("coordlight needs -model_path and -traffic_file")
        run_coordlight(a)
    else:
        run_llmlight(a, rest)


if __name__ == "__main__":
    main()
