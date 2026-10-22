"""
run_eval.py
===========
Evaluates stored weights in a fresh episode on CityFlow or SUMO; on SUMO it can
also enable the SSM device for the safety measures.

Why a separate entry point:
    run_d2marl.py / run_baseline.py train from scratch. The paper re-evaluates
    stored weights (round 99 for the grid scenarios), and the safety measures
    need the slow SSM device, which is not used during training.

Why the weights are copied to a separate eval directory:
    utils/model_test.test() writes to model_dir.replace("model", "records"), i.e.
    records/<memo>/<run>/test_round/round_N, and would **overwrite the test results
    of training**. The weights and configs are therefore copied to
    model/<out_memo>/ and records/<out_memo>/ and evaluated there.

Usage:
    # D2-MARL on CityFlow, round 99
    python run_eval.py -mod D2MARL -memo d2marl-cf-hz-seed10 -round 99 \
        -hangzhou -traffic_file anon_4_4_hangzhou_real.json

    # D2-MARL on SUMO with the SSM device (safety measures)
    python run_eval.py -mod D2MARL -memo d2marl-sumo-hz-seed10 -round 99 -sumo \
        -hangzhou -traffic_file anon_4_4_hangzhou_real.json

    # rule-based baseline: no weights, one episode
    python run_eval.py -mod Fixedtime -hangzhou -sumo \
        -traffic_file anon_4_4_hangzhou_real.json
"""

import os
import re
import glob
import json
import time
import shutil
import random
import argparse
from copy import deepcopy

import numpy as np
import tensorflow as tf

from utils import config
from utils.utils import merge, oneline_wrapper
from utils.model_test import test as model_test
from run_baseline import BASELINES, SCENARIOS, SUMO_BEGIN, check_agent_conf

# D2-MARL state/reward (as the defaults of run_d2marl.py); for D2MARL the
# training config is restored from the source run anyway.
D2MARL_SPEC = dict(
    features=["cur_phase", "traffic_movement_pressure_queue_efficient",
              "lane_enter_running_part", "lane_delay_burden",
              "lane_raw_max_wait", "lane_over_limit_num", "adjacency_matrix"],
    reward={"queue_length": -0.5, "max_wait_penalty": -0.05,
            "over_limit_penalty": -0.25},
    agent_conf={"CNN_layers": [[32, 32]]},
    trainable=True,
)


def find_latest_run(memo, run_name=None):
    # never pick silently among several runs: they may be other demands or repeats
    root = os.path.join("records", memo)
    cands = [c for c in sorted(glob.glob(os.path.join(root, "*")))
             if os.path.isdir(c)]
    if run_name is not None:
        if os.path.basename(run_name) != run_name:
            raise ValueError("-source_run must be the name of a run directory under the memo")
        cands = [c for c in cands if os.path.basename(c) == run_name]
    if len(cands) != 1:
        raise ValueError("need exactly one source run, found %d; choose one with -source_run: %s"
                         % (len(cands), cands))
    return cands[0]


def restore_eval_conf(src_records, overrides, model_name, simulator="sumo"):
    """Restore the training config; override only explicit deployment/logging keys."""
    source = next((os.path.join(src_records, name)
                   for name in ("traffic_env.conf", "anon_env.conf")
                   if os.path.isfile(os.path.join(src_records, name))), None)
    if source is None:
        raise FileNotFoundError("source run lacks traffic_env.conf / anon_env.conf: " + src_records)
    with open(source) as f:
        conf = json.load(f)
    required = ("MODEL_NAME", "LIST_STATE_FEATURE", "DIC_REWARD_INFO", "PHASE",
                "NUM_PHASES", "TOP_K_ADJACENCY", "MIN_ACTION_TIME")
    if model_name == "D2MARL":
        required += ("WAIT_SOFT_LIMIT", "WAIT_HARD_LIMIT", "WAIT_COUNT_THRESHOLD",
                     "HUBER_DELTA", "MAX_PENALTY_CAP")
    missing = [k for k in required if k not in conf]
    if missing:
        raise ValueError("source env config lacks required keys: %s" % missing)
    if conf["MODEL_NAME"] != model_name:
        raise ValueError("model does not match the source config: %s / %s" % (model_name, conf["MODEL_NAME"]))
    if str(conf.get("SIMULATOR", "cityflow")).lower() != simulator:
        raise ValueError("source run used another simulator; refusing to switch simulators silently")
    # JSON object keys become strings; phase_code looks up integer phase ids.
    conf["PHASE"] = {int(k): v for k, v in conf["PHASE"].items()}
    allowed = {"NUM_AGENTS", "NUM_INTERSECTIONS", "NUM_ROW", "NUM_COL",
               "RUN_COUNTS", "TRAFFIC_FILE", "ROADNET_FILE", "SEED",
               "SUMO_NET_FILE", "SUMO_ROU_FILE", "SUMO_LABEL", "SUMO_BEGIN",
               "ENABLE_SSM", "RECORD_QUEUE", "SAVE_REPLAY_ROUND"}
    conf.update({k: deepcopy(v) for k, v in overrides.items() if k in allowed})
    conf["LIST_MODEL_NEED_TO_UPDATE"] = [model_name]  # no exploration in evaluation
    for key in ("ACTION_MASK", "SUMO_INTER_IDS"):
        conf.pop(key, None)  # regenerated for the target network
    return conf, source


def pick_round(model_dir, want=None):
    rounds = set()
    for p in glob.glob(os.path.join(model_dir, "round_*_inter_0*")):
        m = re.search(r"round_(\d+)_inter_0", os.path.basename(p))
        if m:
            rounds.add(int(m.group(1)))
    if not rounds:
        raise FileNotFoundError("no round_*_inter_0 model under %s" % model_dir)
    if want is not None:
        if want not in rounds:
            raise ValueError("no model for round %d; the latest available round is %d"
                             % (want, max(rounds)))
        return want
    return max(rounds)


def stage_eval_copy(src_records, src_model, rnd, out_memo, scenario_dir=None, scenario_files=None):
    """Copy the weights of the chosen round and the configs to a separate eval directory.

    If scenario_dir is given, the scenario files are taken from there instead of
    src_records. Needed for demand transfer (train on hz1, evaluate on hz2):
    src_records holds the source demand, while the target demand is needed.
    For same-demand evaluation both are identical.
    """
    run_name = os.path.basename(src_records.rstrip("/"))
    dst_records = os.path.join("records", out_memo, run_name)
    dst_model = os.path.join("model", out_memo, run_name)
    # several rounds may be evaluated into one output, but never the same round twice
    if (os.path.exists(os.path.join(dst_records, "test_round", "round_%d" % rnd))
            or glob.glob(os.path.join(dst_model, "round_%d_inter_*" % rnd))):
        raise FileExistsError("this round was already evaluated; use a new -out_memo: " + dst_records)
    if not os.path.isfile(os.path.join(src_records, "agent.conf")):
        raise FileNotFoundError("source run lacks agent.conf: " + src_records)
    sources = []
    for name in scenario_files or ():
        candidates = [os.path.join(d, name) for d in (scenario_dir, src_records) if d]
        hit = next((f for f in candidates if os.path.isfile(f)), None)
        if hit is None:
            raise FileNotFoundError("target scenario file not found: " + name)
        sources.append(hit)
    for d in (dst_records, dst_model):
        os.makedirs(d, exist_ok=True)

    for f in ("agent.conf",):
        s = os.path.join(src_records, f)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(dst_records, f))
    for source in sources:
        shutil.copy(source, dst_records)
    # copy only the weights of the chosen round
    n = 0
    for s in glob.glob(os.path.join(src_model, "round_%d_inter_*" % rnd)):
        dst = os.path.join(dst_model, os.path.basename(s))
        (shutil.copytree if os.path.isdir(s) else shutil.copy)(s, dst) \
            if not os.path.exists(dst) else None
        n += 1
    if n == 0:
        raise FileNotFoundError("no weight file of round_%d was copied" % rnd)
    print("staged %d weight files in %s" % (n, dst_model))
    return dst_records, dst_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-mod", required=True,
                    help="D2MARL or a baseline name from run_baseline.BASELINES")
    ap.add_argument("-memo", default=None, help="memo of the training run (locates the weights)")
    ap.add_argument("-source_run", default=None, help="run directory under the memo, if there are several")
    ap.add_argument("-round", type=int, default=None, help="round to evaluate (default: latest available; the paper uses 99)")
    for k in SCENARIOS:
        ap.add_argument("-" + k, action="store_true", default=False)
    ap.add_argument("-traffic_file", default=None)
    ap.add_argument("-seed", type=int, default=10)
    ap.add_argument("-sumo", action="store_true", default=False,
                    help="evaluate on SUMO (default: CityFlow)")
    ap.add_argument("-sumo_dir", type=str, default="sumo_scenarios")
    ap.add_argument("-no_ssm", action="store_true", default=False)
    ap.add_argument("-save_replay", action="store_true", default=False,
                    help="record the executed signal states of this episode to tls_trace.csv; "
                         "opens sumo-gui if a DISPLAY is available"
                         " (SUMO only)")
    ap.add_argument("-out_memo", default=None)
    a = ap.parse_args()

    os.environ["PYTHONHASHSEED"] = str(a.seed)
    random.seed(a.seed); np.random.seed(a.seed); tf.random.set_seed(a.seed)

    picked = [k for k in SCENARIOS if getattr(a, k)]
    if len(picked) != 1:
        raise ValueError("specify exactly one network: " + " / ".join("-" + k for k in SCENARIOS))
    net_key = picked[0]
    road_net, template, traffic_list = SCENARIOS[net_key]
    traffic_file = a.traffic_file or traffic_list[0]
    stem = os.path.splitext(traffic_file)[0]

    spec = D2MARL_SPEC if a.mod == "D2MARL" else BASELINES[a.mod]
    sim = "sumo" if a.sumo else "cityflow"
    enable_ssm = a.sumo and not a.no_ssm
    NUM_ROW, NUM_COL = (int(x) for x in road_net.split("_"))
    num_inter = NUM_ROW * NUM_COL
    out_memo = a.out_memo or ("eval-" + (a.memo or "%s-%s" % (a.mod, net_key)))

    env_extra = {
        # Must match run_baseline.py exactly: PressLight is one independent DQN per
        # intersection, and NUM_AGENTS=1 would make action[idx] in env.step fail.
        "NUM_AGENTS": num_inter if a.mod in ("Fixedtime", "MaxPressure",
                                             "Actuated", "PressLight")
                      else 1,
        "NUM_INTERSECTIONS": num_inter,
        "NUM_ROW": NUM_ROW, "NUM_COL": NUM_COL,
        # Full one-hour episode by default; inference timing (inference_cost) only
        # needs a few hundred seconds and shortens it via the environment variable.
        "RUN_COUNTS": int(os.environ.get("D2MARL_RUN_COUNTS", 3600)),
        "MODEL_NAME": a.mod,
        "TRAFFIC_FILE": traffic_file,
        "ROADNET_FILE": "roadnet_%s.json" % road_net,
        "LIST_STATE_FEATURE": list(spec["features"]),
        "DIC_REWARD_INFO": dict(spec["reward"]),
        "LIST_MODEL_NEED_TO_UPDATE": ([a.mod] if spec["trainable"] else []),
        "NUM_GENERATORS": 1, "NUM_ROUNDS": 1,
        "DELAY_THRESHOLD": 90, "SAVE_REPLAY_ROUND": -1,
        "SEED": a.seed, "SIMULATOR": sim,
        "SUMO_NET_FILE": stem + ".net.xml",
        "SUMO_ROU_FILE": stem + ".repaired.rou.xml",
        "SUMO_LABEL": "eval_%s_%s" % (a.mod, stem),
        "SUMO_BEGIN": SUMO_BEGIN.get(net_key, 0),
        # the safety measures need both; SSM is slow and only used in evaluation
        "ENABLE_SSM": enable_ssm, "RECORD_QUEUE": enable_ssm,
    }
    env_conf = merge(config.dic_traffic_env_conf, env_extra)
    agent_conf = merge(getattr(config, "DIC_BASE_AGENT_CONF"), dict(spec["agent_conf"]))
    if a.mod != "D2MARL":
        check_agent_conf(a.mod, agent_conf)

    print("=" * 64)
    print("evaluate: %-16s network: %-12s traffic: %s" % (a.mod, net_key, traffic_file))
    print("SSM safety measures: %s" % ("on" if enable_ssm else "off"))
    print("=" * 64)

    if not spec["trainable"]:
        stamp = time.strftime("%m_%d_%H_%M_%S")
        work = os.path.join("records", out_memo, traffic_file + "_" + stamp)
        path_conf = merge(config.DIC_PATH, {
            "PATH_TO_WORK_DIRECTORY": work,
            "PATH_TO_MODEL": os.path.join("model", out_memo, traffic_file + "_" + stamp),
            "PATH_TO_DATA": (os.path.join(a.sumo_dir, stem) if a.sumo
                             else os.path.join("data", template, road_net)),
            "PATH_TO_ERROR": os.path.join("errors", out_memo),
        })
        if a.sumo:
            from utils.env_factory import prepare_sumo_conf
            prepare_sumo_conf(env_conf, path_conf["PATH_TO_DATA"])
        env_conf["SAVE_REPLAY_ROUND"] = 0 if a.save_replay else -1
        oneline_wrapper(agent_conf, env_conf, path_conf)
        out_root = work
    else:
        if not a.memo:
            raise ValueError("%s needs -memo (memo of the training run)" % a.mod)
        src_records = find_latest_run(a.memo, a.source_run)
        env_conf, source_conf = restore_eval_conf(src_records, env_extra, a.mod,
                                                  simulator=sim)
        src_model = src_records.replace("records", "model", 1)
        rnd = pick_round(src_model, a.round)
        print("source: %s   loading round %d" % (src_model, rnd))
        if a.sumo:
            scen_dir = os.path.join(a.sumo_dir, stem)
            scen_files = [env_conf["SUMO_NET_FILE"], env_conf["SUMO_ROU_FILE"]]
        else:
            scen_dir = os.path.join("data", template, road_net)
            scen_files = [env_conf["ROADNET_FILE"], env_conf["TRAFFIC_FILE"]]
        dst_records, dst_model = stage_eval_copy(
            src_records, src_model, rnd, out_memo, scenario_dir=scen_dir,
            scenario_files=scen_files)

        if a.sumo:
            from utils.env_factory import prepare_sumo_conf
            prepare_sumo_conf(env_conf, dst_records)
        # write the effective config into the eval copy as well
        for name in ("traffic_env.conf", "anon_env.conf"):
            with open(os.path.join(dst_records, name), "w") as f:
                json.dump(env_conf, f, indent=2)
        # keep the effective config of every evaluated round
        with open(os.path.join(dst_records, "eval_config_round_%d.json" % rnd), "w") as f:
            json.dump({"source_records": os.path.abspath(src_records),
                       "source_model": os.path.abspath(src_model),
                       "source_env_config": os.path.abspath(source_conf),
                       "round": rnd, "save_replay": a.save_replay,
                       "environment": env_conf}, f, indent=2)
        model_test(dst_model, rnd, int(os.environ.get("D2MARL_RUN_COUNTS", 3600)),
                   env_conf, save_replay=a.save_replay, use_passed_env_conf=True)
        out_root = dst_records

    if not enable_ssm:
        print("\nDone.")
        return
    print("\nDone. SSM outputs:")
    found = False
    for f in ("ssm.xml", "lane_queue.csv", "collisions.xml"):
        for p in glob.glob(os.path.join(out_root, "**", f), recursive=True):
            print("  " + p); found = True
    if not found:
        print("  (none; if SSM was enabled, check that the SUMO version supports device.ssm)")


if __name__ == "__main__":
    main()
