"""
run_baseline.py
===============
Runs the baselines of the paper with the same code and the same environment as
D2-MARL, on CityFlow or SUMO.

Why:
    The CityFlow baselines of the paper were run with LLMTSCS (see baselines/llmtscs).
    For SUMO, and for any comparison that needs one environment and one metric
    routine for all methods, the baselines are ported into this code base.

The state features / reward / agent config of every baseline are copied from the
run_*.py scripts of LLMTSCS (MIN_ACTION_TIME=15, MEASURE_TIME=15, YELLOW_TIME=5,
FIXED_TIME=[30]*4 for the fixed-time plan).

Usage:
    python run_baseline.py -mod AdvancedColight -hangzhou -sumo -seed 10
    python run_baseline.py -mod Fixedtime -jinan \
        -traffic_file anon_3_4_jinan_real.json -seed 10
"""

import os
import time
import random
import argparse
from multiprocessing import Process

import numpy as np
# TensorFlow is only used for the random seed; LLMLight runs without TF and must
# not fail on this import.
try:
    import tensorflow as tf
except ImportError:
    tf = None

from utils.utils import pipeline_wrapper, oneline_wrapper, merge
from utils import config


# ==========================================================================
# Configuration of each baseline, copied from the run_*.py scripts of LLMTSCS.
# ==========================================================================
BASELINES = {
    "Fixedtime": dict(
        features=["cur_phase", "time_this_phase", "traffic_movement_pressure_queue"],
        reward={"pressure": 0},
        agent_conf={"FIXED_TIME": [30] * 8},
        trainable=False),
    "MaxPressure": dict(
        features=["cur_phase", "traffic_movement_pressure_queue"],
        reward={"pressure": 0},
        agent_conf={"FIXED_TIME": [30] * 8},
        trainable=False),
    # Actuated signal control (ASC).
    # Needs time_this_phase for min/max green and the queues as demand proxy.
    "Actuated": dict(
        features=["cur_phase", "time_this_phase",
                  "traffic_movement_pressure_queue"],
        reward={"pressure": 0},
        agent_conf={"FIXED_TIME": [30] * 8, "MIN_GREEN": 10, "MAX_GREEN": 60},
        trainable=False),
    # ---- non-learning and inference-only baselines ----
    # Random: uniform redraw over the phase set at every decision step; one episode.
    "Random": dict(
        features=["cur_phase"],
        reward={"pressure": 0},
        agent_conf={"FIXED_TIME": [30] * 8},
        trainable=False),
    # Original CoLight: observes lane vehicle counts (not the efficient pressure),
    # which is exactly the difference between CoLight and Eff-CoLight.
    "CoLight": dict(
        features=["cur_phase", "lane_num_vehicle", "adjacency_matrix"],
        reward={"queue_length": -0.25},
        agent_conf={"CNN_layers": [[32, 32]]}, trainable=True),
    # LLMLight-13B: inference only, no training. Needs the lane_cells feature;
    # prompt and parsing in models/llmlight_agent.py, copied from LLMTSCS.
    "LLMLight": dict(
        features=["cur_phase", "lane_num_waiting_vehicle_in", "lane_cells"],
        reward={"queue_length": -0.25},
        agent_conf={"LLM_PATH": "", "NEW_MAX_TOKENS": 1024,
                    "GPU_MEM_UTIL": 0.90, "MAX_MODEL_LEN": 4096},
        trainable=False),
    "MPLight": dict(
        features=["cur_phase", "traffic_movement_pressure_num"],
        reward={"pressure": -0.25},
        agent_conf={}, trainable=True),
    "EfficientColight": dict(
        features=["cur_phase", "traffic_movement_pressure_queue_efficient",
                  "adjacency_matrix"],
        reward={"queue_length": -0.25},
        # The CoLight family reads dic_agent_conf['CNN_layers'] (not in DIC_BASE_AGENT_CONF);
        # same value as LLMTSCS/run_advanced_colight.py and run_d2marl.py.
        agent_conf={"CNN_layers": [[32, 32]]}, trainable=True),
    "AdvancedColight": dict(
        features=["cur_phase", "traffic_movement_pressure_queue_efficient",
                  "lane_enter_running_part", "adjacency_matrix"],
        reward={"queue_length": -0.25},
        agent_conf={"CNN_layers": [[32, 32]]}, trainable=True),
    "PressLight": dict(
        features=["cur_phase", "traffic_movement_pressure_queue"],
        reward={"pressure": -0.25},
        agent_conf={}, trainable=True),
    "Attend": dict(
        features=["num_in_seg_attend"],
        reward={"pressure": -0.25},
        agent_conf={}, trainable=True),
}

SCENARIOS = {
    "hangzhou":   ("4_4",  "Hangzhou",
                   ["anon_4_4_hangzhou_real.json",
                    "anon_4_4_hangzhou_real_5816.json"]),
    "jinan":      ("3_4",  "Jinan",
                   ["anon_3_4_jinan_real.json",
                    "anon_3_4_jinan_real_2000.json",
                    "anon_3_4_jinan_real_2500.json"]),
    "newyork163": ("16_3", "newyork_16_3", ["anon_16_3_newyork_real.json"]),
    # real non-grid network, SUMO only (no CityFlow version).
    # road_net only fills NUM_ROW/NUM_COL here; the real number of intersections
    # is set by prepare_sumo_conf from net.xml.
    "ingolstadt21": ("21_1", "ingolstadt21", ["ingolstadt21.json"]),
    "cologne8":     ("8_1",  "cologne8",     ["cologne8.json"]),
}

# the demand of the non-grid scenario does not start at time 0, see sumo_env._sumo_cmd
SUMO_BEGIN = {"ingolstadt21": 57600, "cologne8": 25200}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-mod", type=str, required=True,
                   choices=sorted(BASELINES.keys()))
    p.add_argument("-memo", type=str, default=None)
    for k in SCENARIOS:
        p.add_argument("-" + k, action="store_true", default=False)
    p.add_argument("-traffic_file", type=str, default=None)
    p.add_argument("-seed", type=int, default=10)
    p.add_argument("-num_rounds", type=int, default=None)
    p.add_argument("-workers", type=int, default=4)
    p.add_argument("-single_process", action="store_true", default=False)
    p.add_argument("-hp", default=None,
                   help="override agent hyperparameters, comma-separated KEY=VALUE, e.g. "
                        "-hp LEARNING_RATE=0.0005,BATCH_SIZE=32. "
                        "Only for baseline sensitivity runs; the reported results use "
                        "the original open-source configuration")
    p.add_argument("-sumo", action="store_true", default=False)
    p.add_argument("-sumo_dir", type=str, default="sumo_scenarios")
    p.add_argument("-enable_ssm", action="store_true", default=False)
    p.add_argument("-record_queue", action="store_true", default=False)
    p.add_argument("-resume", action="store_true", default=False,
                   help="reuse the existing run directory of this memo and continue after "
                        "the last complete round in model/ instead of starting a new run")
    return p.parse_args()


AGENT_FILES = {
    "Fixedtime": "models/fixedtime_agent.py",
    "MaxPressure": "models/maxpressure_agent.py",
    "Actuated": "models/actuated_agent.py",
    "MPLight": "models/mplight_agent.py",
    "EfficientColight": "models/colight_agent.py",
    "AdvancedColight": "models/colight_agent.py",
    "PressLight": "models/presslight.py",
    "Random": "models/random_agent.py",
    "CoLight": "models/colight_agent.py",
    "LLMLight": "models/llmlight_agent.py",
    "Attend": "models/attendlight_agent.py",
}


def check_agent_conf(model, agent_conf):
    """
    Static check before the start: collect every dic_agent_conf["XXX"] read in the
    agent class (and its base classes) and compare with the provided keys.

    A missing key would otherwise surface only when the agent is built, i.e. after
    the job has been queued and SUMO has started.
    """
    import re as _re
    files = [AGENT_FILES.get(model, "")] + ["models/network_agent.py", "models/agent.py"]
    need = set()
    for f in files:
        if not f or not os.path.exists(f):
            continue
        need |= set(_re.findall(
            r"dic_agent_conf\[['\"]([A-Za-z_0-9]+)['\"]\]", open(f).read()))
    missing = sorted(need - set(agent_conf.keys()))
    if missing:
        raise KeyError(
            "{0} lacks agent_conf keys: {1}\n"
            "add them to BASELINES['{0}']['agent_conf'] in run_baseline.py"
            .format(model, ", ".join(missing)))



from utils.resume import find_resume_point as _find_resume_point

def main(a):
    os.environ["PYTHONHASHSEED"] = str(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    if tf is not None:
        tf.random.set_seed(a.seed)

    picked = [k for k in SCENARIOS if getattr(a, k)]
    if len(picked) != 1:
        raise ValueError("specify exactly one network: " +
                         " / ".join("-" + k for k in SCENARIOS))
    net_key = picked[0]
    road_net, template, traffic_list = SCENARIOS[net_key]

    if a.traffic_file:
        if a.traffic_file not in traffic_list:
            raise ValueError("traffic file {0} does not belong to {1}: {2}".format(
                a.traffic_file, net_key, traffic_list))
        traffic_list = [a.traffic_file]

    spec = BASELINES[a.mod]
    num_rounds = a.num_rounds if a.num_rounds is not None else (
        1 if not spec["trainable"] else 100)
    if not spec["trainable"] and a.num_rounds is None:
        print("==> {0} needs no training, running 1 episode".format(a.mod))

    NUM_ROW, NUM_COL = (int(x) for x in road_net.split("_"))
    num_inter = NUM_ROW * NUM_COL
    memo = a.memo or "base-{0}-{1}-seed{2}".format(
        a.mod.lower(), net_key, a.seed)

    print("=" * 60)
    print("Baseline: {0}   network: {1}   simulator: {2}".format(
        a.mod, net_key, "SUMO" if a.sumo else "CityFlow"))
    print("Traffic: {0}".format(traffic_list))
    print("Features: {0}".format(spec["features"]))
    print("Reward: {0}".format(spec["reward"]))
    print("Rounds: {0}   seed: {1}".format(num_rounds, a.seed))
    print("=" * 60)

    process_list = []
    for traffic_file in traffic_list:
        stem = os.path.splitext(traffic_file)[0]
        env_extra = {
            # NUM_AGENTS sets how many agent objects generator builds and the length
            # of action_list. Two kinds:
            #   = num_inter  one independent agent per intersection.
            #                Fixedtime/MaxPressure/Actuated are rule-based;
            #                PressLight is an independent DQN whose choose_action takes
            #                one state and returns a scalar (utils/updater.py calls
            #                prepare_Xs_Y per agent). With 1, action[idx] in env.step
            #                would be out of range.
            #   = 1          parameter sharing: one network for all intersections;
            #                choose_action takes the state list and returns all actions
            #                (CoLight / MPLight / Attend, ...).
            # Note: LLMTSCS ships EfficientPressLight (a parameter-sharing variant);
            # the paper reports PressLight, so the original independent DQN is used.
            "NUM_AGENTS": num_inter if a.mod in ("Fixedtime", "MaxPressure",
                                                 "Actuated", "PressLight")
                          else 1,
            "NUM_INTERSECTIONS": num_inter,
            # As in run_eval.py: inference timing needs only a few decisions, so the
            # horizon can be shortened via an environment variable (default 3600 s).
            "RUN_COUNTS": int(os.environ.get("D2MARL_RUN_COUNTS", 3600)),
            "MODEL_NAME": a.mod,
            "NUM_ROW": NUM_ROW, "NUM_COL": NUM_COL,
            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),
            "LIST_STATE_FEATURE": list(spec["features"]),
            "DIC_REWARD_INFO": dict(spec["reward"]),
            "SEED": a.seed,
            # Rule-based agents have no train_network. config.py lists them in
            # LIST_MODEL_NEED_TO_UPDATE, so it is overridden here from
            # BASELINES[...]["trainable"] without touching the global config.
            "LIST_MODEL_NEED_TO_UPDATE": ([a.mod] if spec["trainable"] else []),
            # the following keys are not in the base config and must be given
            "NUM_GENERATORS": 1,
            "NUM_ROUNDS": num_rounds,
            "DELAY_THRESHOLD": 90,
            "SAVE_REPLAY_ROUND": -1,
            "SIMULATOR": "sumo" if a.sumo else "cityflow",
            "SUMO_NET_FILE": stem + ".net.xml",
            "SUMO_ROU_FILE": stem + ".repaired.rou.xml",
            "SUMO_LABEL": "{0}_{1}".format(memo, stem),
            "SUMO_BEGIN": SUMO_BEGIN.get(net_key, 0),
            "ENABLE_SSM": a.enable_ssm,
            "RECORD_QUEUE": a.record_queue,
        }
        env_conf = merge(config.dic_traffic_env_conf, env_extra)

        agent_conf = merge(getattr(config, "DIC_BASE_AGENT_CONF"),
                           dict(spec["agent_conf"]))
        if a.hp:
            # only existing keys may be overridden; a typo would otherwise be silently ignored
            import ast as _ast
            for kv in a.hp.split(","):
                if not kv.strip():
                    continue
                k, _, v = kv.partition("=")
                k = k.strip()
                if k not in agent_conf:
                    raise KeyError("%s in -hp is not a known hyperparameter. Available: %s"
                                   % (k, ", ".join(sorted(agent_conf))))
                try:
                    agent_conf[k] = _ast.literal_eval(v.strip())
                except (ValueError, SyntaxError):
                    agent_conf[k] = v.strip()
                print("==> hyperparameter override: %s = %r" % (k, agent_conf[k]))
        check_agent_conf(a.mod, agent_conf)

        stamp = time.strftime("%m_%d_%H_%M_%S", time.localtime(time.time()))
        start_round = 0
        if a.resume:
            stamp_r, start_round = _find_resume_point(memo, traffic_file)
            if stamp_r is None:
                print("==> -resume: no run directory to resume, starting a new run")
            else:
                stamp = stamp_r
                env_conf["START_ROUND"] = start_round
                print("==> resuming %s from round %d" % (stamp, start_round))

        path_conf = merge(config.DIC_PATH, {
            "PATH_TO_MODEL": os.path.join("model", memo, traffic_file + "_" + stamp),
            "PATH_TO_WORK_DIRECTORY": os.path.join(
                "records", memo, traffic_file + "_" + stamp),
            "PATH_TO_DATA": (os.path.join(a.sumo_dir, stem) if a.sumo
                             else os.path.join("data", template, road_net)),
            "PATH_TO_ERROR": os.path.join("errors", memo),
        })

        if a.sumo:
            from utils.env_factory import prepare_sumo_conf
            prepare_sumo_conf(env_conf, path_conf["PATH_TO_DATA"])

        # Rule-based agents have no train_network / load_network and cannot use the
        # generator->update->model_test loop of the pipeline. As in LLMTSCS
        # (run_fixedtime.py), they run one episode through oneline_wrapper.
        driver = pipeline_wrapper if spec["trainable"] else oneline_wrapper
        print("==> driver: {0}".format(
            "pipeline (training)" if spec["trainable"] else "oneline (single episode)"))

        if a.single_process:
            driver(dic_agent_conf=agent_conf,
                   dic_traffic_env_conf=env_conf,
                   dic_path=path_conf)
        else:
            process_list.append(Process(
                target=driver, args=(agent_conf, env_conf, path_conf)))

    for i in range(0, len(process_list), a.workers):
        batch = process_list[i:i + a.workers]
        for p in batch:
            p.start()
        for p in batch:
            p.join()


if __name__ == "__main__":
    main(parse_args())
