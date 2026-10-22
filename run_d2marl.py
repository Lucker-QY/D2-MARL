from utils.utils import pipeline_wrapper, merge
from utils import config
import time
from multiprocessing import Process
import argparse
import os
import sys
import random
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()

    # ── basic run settings ──────────────────────────────────────
    parser.add_argument("-memo",           type=str,   default='D2MARL')
    parser.add_argument("-mod",            type=str,   default="D2MARL")
    parser.add_argument("-gen",            type=int,   default=1)
    parser.add_argument("-multi_process",  action="store_true", default=True)
    parser.add_argument("-seed",           type=int,   default=2024, help="global random seed (the paper uses 10, 100, 1000)")

    # number of traffic files trained in parallel (multi-process mode)
    parser.add_argument("-workers",        type=int,   default=8)

    # ── network ─────────────────────────────────────────────────
    parser.add_argument("-hangzhou",   action="store_true", default=False)
    parser.add_argument("-jinan",      action="store_true", default=False)
    parser.add_argument("-newyork163",    action="store_true", default=False)
    # real non-grid network (RESCO benchmark), SUMO only
    parser.add_argument("-ingolstadt21",  action="store_true", default=False)
    parser.add_argument("-cologne8",      action="store_true", default=False)

    # ── ablations, run control and simulator ────────────────────
    parser.add_argument("-fixed_gating_alpha", type=float, default=None,
                        help="fix the gating coefficient to a constant (e.g. 0.5) for the "
                             "gate ablation; learned when omitted")
    parser.add_argument("-traffic_file", type=str, default=None,
                        help="train only this traffic file (e.g. anon_4_4_hangzhou_real.json); "
                             "by default all traffic files of the network are trained")
    parser.add_argument("-num_rounds", type=int, default=None,
                        help="override the number of training rounds (default 100); 1-2 for a smoke test")
    parser.add_argument("-single_process", action="store_true", default=False,
                        help="disable multiprocessing (child processes hide tracebacks)")
    parser.add_argument("-resume", action="store_true", default=False,
                        help="reuse the existing run directory of this memo and continue after "
                             "the last complete round in model/ instead of starting a new "
                             "time-stamped run; for jobs that were interrupted")
    parser.add_argument("-sumo", action="store_true", default=False,
                        help="simulate with SUMO (default: CityFlow); the scenarios must "
                             "first be converted with d2marl_sumo.cityflow2sumo")
    parser.add_argument("-sumo_dir", type=str, default="sumo_scenarios",
                        help="root of the SUMO scenarios, one sub-directory per scenario, "
                             "named like the CityFlow traffic file without .json")
    parser.add_argument("-enable_ssm", action="store_true", default=False,
                        help="enable the SUMO SSM device (TTC/PET/conflicts); slow, "
                             "use it for evaluation only, not for training")
    parser.add_argument("-record_queue", action="store_true", default=False,
                        help="record lane-level queue traces (spillback indicators)")

    # ── debugging / monitoring ──────────────────────────────────
    parser.add_argument("-debug",                    action="store_true", default=False,
                        help="enable debug output")
    parser.add_argument("-save_reward_metrics",      action="store_true", default=False,
                        help="write per-step reward metrics (queue/pressure/q_value) to CSV")
    parser.add_argument("-reward_metrics_save_freq", type=int, default=100,
                        help="save frequency of the reward-metric CSV in steps (default 100)")
    parser.add_argument("-intra_phase_attention",     action="store_true", default=True,
                        help="intra-phase attention (D2MARLAgent), on by default")
    parser.add_argument("-no_intra_phase_attention",  action="store_true", default=False,
                        help="disable intra-phase attention (IA ablation)")

    # ── delay-burden state feature (lane_delay_burden) ──────────
    parser.add_argument("-use_max_wait_time", action="store_true", default=True,
                        help="add the per-lane delay burden to the state (default on)")
    parser.add_argument("-no_max_wait_time", action="store_true", default=False,
                        help="remove the delay-burden feature (MW ablation)")

    # ── delay penalty in the reward ─────────────────────────────
    parser.add_argument("-use_wait_penalty", action="store_true", default=True,
                        help="add the bounded delay penalty to the reward (default on)")
    parser.add_argument("-no_wait_penalty", action="store_true", default=False,
                        help="remove the delay penalty (WP ablation): queue-length reward only")

    # ── replay recording & delay threshold ──────────────────────
    # Default -1 (off): saving a replay makes sumo_env start sumo-gui, which fails
    # on compute nodes without a display.
    parser.add_argument("-save_replay_round", type=int, default=-1,
                        help="save a simulation replay at this round; -1 = never (default). "
                             "Uses sumo-gui with a DISPLAY, a netstate dump otherwise")
    # ── adaptation to networks with short links (Section IV-C; used on the non-grid networks) ──
    parser.add_argument("-spillback", action="store_true", default=False,
                        help="non-grid adaptation: free-storage ratio of the downstream link in the "
                             "spatial, gate and intra-phase inputs, capacity-discounted pressure")
    parser.add_argument("-spillback_min_free", type=float, default=2.0,
                        help="free downstream spaces below which a green movement counts as blocked")
    parser.add_argument("-spillback_r0", type=float, default=0.2,
                        help="free-storage ratio c0 below which the pressure is discounted")
    parser.add_argument("-weight_blocked_green", type=float, default=-0.25,
                        help="weight of the blocked-green term (-spillback)")

    parser.add_argument("-delay_threshold", type=int, default=90,
                        help="waiting time (s) above which a wait counts as a severe delay event (default 90)")

    # ── bounded delay penalty (W_s, W_h, C_max) ─────────────────
    parser.add_argument("-wait_soft_limit", type=float, default=60.0, help="soft waiting-time limit W_s (s)")
    parser.add_argument("-wait_hard_limit", type=float, default=90.0, help="hard waiting-time limit W_h (s)")
    parser.add_argument("-max_penalty_cap", type=float, default=2000.0, help="penalty cap C_max")

    # ── Huber transition (Delta) ────────────────────────────────
    parser.add_argument("-huber_delta", type=float, default=30.0, help="transition width Delta (s beyond W_h at which the penalty turns linear)")

    # ── stratified replay ───────────────────────────────────────
    parser.add_argument("-use_stratified_replay", action="store_true", default=True, help="stratified replay of severe-delay samples")
    parser.add_argument("-no_stratified_replay", action="store_true", default=False, help="plain random replay (RP ablation)")
    # tau_p = -50 and rho = 0.05 as in Table I.
    parser.add_argument("-penalty_replay_threshold", type=float, default=-50.0, help="reward threshold tau_p below which a sample counts as severe")
    parser.add_argument("-penalty_replay_ratio", type=float, default=0.05, help="maximum share rho of severe samples per batch")

    # ── extended state and over-limit term ──────────────────────
    parser.add_argument("-use_extended_state", action="store_true", default=True, help="extended state (max waiting time + over-limit count)")
    parser.add_argument("-no_extended_state", action="store_true", default=False, help="remove the extended state and the over-limit term (ES ablation)")
    parser.add_argument("-wait_count_threshold", type=float, default=60.0, help="over-limit threshold tau (s)")

    # reward weights (omega_q, omega_w, omega_o; negative = penalty)
    parser.add_argument("-weight_queue_length", type=float, default=-0.5, help="queue-length weight")
    parser.add_argument("-weight_max_wait", type=float, default=-0.05, help="delay-penalty weight omega_w")
    parser.add_argument("-weight_over_limit", type=float, default=-0.25, help="over-limit weight")

    return parser.parse_args() 

def main(in_args=None):
    if in_args is None:
        in_args = parse_args()

    # PYTHONHASHSEED only takes effect before the interpreter starts, so the process restarts itself once.
    if os.environ.get("PYTHONHASHSEED") != str(in_args.seed):
        os.environ["PYTHONHASHSEED"] = str(in_args.seed)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    # ================= global random seed =================
    os.environ['PYTHONHASHSEED'] = str(in_args.seed)
    random.seed(in_args.seed)
    np.random.seed(in_args.seed)
    tf.random.set_seed(in_args.seed)
    print(f"==> global random seed: {in_args.seed}")
    # ==========================================================

    # ── network / demand ────────────────────────────────────────
    if in_args.hangzhou:
        count = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_hangzhou_real.json",
                              "anon_4_4_hangzhou_real_5816.json"]
        num_rounds = 100
        template = "Hangzhou"
    elif in_args.jinan:
        count = 3600
        road_net = "3_4"
        traffic_file_list = ["anon_3_4_jinan_real.json",
                              "anon_3_4_jinan_real_2000.json",
                              "anon_3_4_jinan_real_2500.json"]
        num_rounds = 100
        template = "Jinan"
    elif in_args.newyork163:
        count = 3600
        road_net = "16_3"
        traffic_file_list = ["anon_16_3_newyork_real.json"]
        num_rounds = 100
        template = "newyork_16_3"
    elif in_args.ingolstadt21:
        count = 3600
        road_net = "21_1"
        traffic_file_list = ["ingolstadt21.json"]
        num_rounds = 100
        template = "ingolstadt21"
    elif in_args.cologne8:
        count = 3600
        road_net = "8_1"
        traffic_file_list = ["cologne8.json"]
        num_rounds = 100
        template = "cologne8"
    else:
        raise ValueError("choose a network with -hangzhou / -jinan / -newyork163 / -ingolstadt21 / -cologne8")

    if in_args.traffic_file is not None:
        if in_args.traffic_file not in traffic_file_list:
            raise ValueError("traffic file {0} is not in the list of this network: {1}".format(
                in_args.traffic_file, traffic_file_list))
        traffic_file_list = [in_args.traffic_file]
        print("==> single traffic file: {0}".format(in_args.traffic_file))

    if in_args.num_rounds is not None:
        num_rounds = in_args.num_rounds
        print("==> number of rounds overridden: {0}".format(num_rounds))

    NUM_COL = int(road_net.split('_')[1])
    NUM_ROW = int(road_net.split('_')[0])
    num_intersections = NUM_ROW * NUM_COL
    print(f"num_intersections: {num_intersections}")
    print(traffic_file_list)

    # ── feature switches ───────────────────────────────────────
    use_max_wait_time = not in_args.no_max_wait_time if in_args.no_max_wait_time else in_args.use_max_wait_time
    use_extended_state = not in_args.no_extended_state if in_args.no_extended_state else in_args.use_extended_state

    # ── state features ─────────────────────────────────────────
    state_features = [
        "cur_phase",
        "traffic_movement_pressure_queue_efficient",
        "lane_enter_running_part",
    ]
    if use_max_wait_time:
        state_features.append("lane_delay_burden")

    # delay-risk features
    if use_extended_state:
        state_features.append("lane_raw_max_wait")     # normalised max waiting time
        state_features.append("lane_over_limit_num")   # normalised over-limit count
        print(f"==> extended state features on (threshold: {in_args.wait_count_threshold}s)")

    if in_args.spillback:
        # capacity-discounted pressure (same 12 values) and the free-storage ratio c
        state_features = ["traffic_movement_pressure_queue_capacity"
                          if x == "traffic_movement_pressure_queue_efficient" else x
                          for x in state_features]
        state_features.append("lane_downstream_free_capacity")
        print("==> non-grid adaptation on (capacity-discounted pressure, free-storage ratio)")

    state_features.append("adjacency_matrix")  # the adjacency matrix must come last
    # ─────────────────────────────────────────────────────────────

    # ── agent config ────────────────────────────────────────────
    use_stratified_replay = not in_args.no_stratified_replay if in_args.no_stratified_replay else in_args.use_stratified_replay

    if use_stratified_replay:
        print(f"==> stratified replay on (threshold: {in_args.penalty_replay_threshold}, max share: {in_args.penalty_replay_ratio})")
    else:
        print("==> stratified replay off (plain random sampling)")

    dic_agent_conf_extra = {
        "CNN_layers": [[32, 32]],
        "FIXED_GATING_ALPHA": in_args.fixed_gating_alpha,
        "DEBUG": in_args.debug,
        "SAVE_REWARD_METRICS": in_args.save_reward_metrics,
        "REWARD_METRICS_SAVE_FREQ": in_args.reward_metrics_save_freq,
        "INTRA_PHASE_ATTENTION": not in_args.no_intra_phase_attention if in_args.no_intra_phase_attention else in_args.intra_phase_attention,

        # replay parameters
        "USE_STRATIFIED_REPLAY": use_stratified_replay,
        "PENALTY_REPLAY_THRESHOLD": in_args.penalty_replay_threshold,
        "PENALTY_REPLAY_RATIO": in_args.penalty_replay_ratio,
    }
    deploy_dic_agent_conf = merge(getattr(config, "DIC_BASE_AGENT_CONF"), dic_agent_conf_extra)

    # ── training ────────────────────────────────────────────────
    process_list = []
    for traffic_file in traffic_file_list:
        # ── reward switches ─────────────────────────────────────────
        use_wait_penalty = not in_args.no_wait_penalty if in_args.no_wait_penalty else in_args.use_wait_penalty

        # reward weights
        dic_reward_info = {
            "queue_length": in_args.weight_queue_length,
            "max_wait_penalty": in_args.weight_max_wait if use_wait_penalty else 0.0,
            "over_limit_penalty": in_args.weight_over_limit if use_extended_state else 0.0,
        }
        if in_args.spillback:
            dic_reward_info["blocked_green_penalty"] = in_args.weight_blocked_green

        dic_traffic_env_conf_extra = {
            "NUM_ROUNDS": num_rounds,
            "NUM_GENERATORS": in_args.gen,
            "NUM_AGENTS": 1,
            "NUM_INTERSECTIONS": num_intersections,
            "RUN_COUNTS": count,
            "MODEL_NAME": in_args.mod,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,
            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),
            "DEBUG": in_args.debug,
            "SAVE_REPLAY_ROUND": in_args.save_replay_round,
            "DELAY_THRESHOLD": in_args.delay_threshold,
            "LIST_STATE_FEATURE": state_features,
            "DIC_REWARD_INFO": dic_reward_info,

            # bounded delay penalty
            "WAIT_SOFT_LIMIT": in_args.wait_soft_limit,
            "WAIT_HARD_LIMIT": in_args.wait_hard_limit,
            "MAX_PENALTY_CAP": in_args.max_penalty_cap,
            "WAIT_COUNT_THRESHOLD": in_args.wait_count_threshold,

            # Huber transition
            "HUBER_DELTA": in_args.huber_delta,
            "SEED": in_args.seed,
            # fail on errors in test episodes instead of continuing training
            "STRICT_TEST_ERRORS": True,

            # ── SUMO ────────────────────────────────────────────────
            "SIMULATOR": "sumo" if in_args.sumo else "cityflow",
            "SUMO_NET_FILE": os.path.splitext(traffic_file)[0] + ".net.xml",
            "SUMO_ROU_FILE": os.path.splitext(traffic_file)[0] + ".repaired.rou.xml",
            "SUMO_LABEL": "{0}_{1}".format(in_args.memo,
                                           os.path.splitext(traffic_file)[0]),
            # the demand of the non-grid scenario does not start at time 0 (see sumo_env._sumo_cmd)
            "SUMO_BEGIN": {"ingolstadt21": 57600, "cologne8": 25200}.get(template, 0),
            "ENABLE_SSM": in_args.enable_ssm,
            "RECORD_QUEUE": in_args.record_queue,
        }

        if in_args.spillback:
            dic_traffic_env_conf_extra["SPILLBACK_MIN_FREE"] = in_args.spillback_min_free
            dic_traffic_env_conf_extra["SPILLBACK_PRESSURE_R0"] = in_args.spillback_r0

        # ── merge the configs ───────────────────────────────────────
        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
        # One time stamp for both model/ and records/, so that the two directories
        # always match (needed for -resume).
        stamp = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
        if in_args.resume:
            from utils.resume import find_resume_point
            stamp_r, start_round = find_resume_point(in_args.memo, traffic_file)
            if stamp_r is None:
                print("==> -resume: no run directory to resume, starting a new run")
            else:
                stamp = stamp_r
                deploy_dic_traffic_env_conf["START_ROUND"] = start_round
                print("==> resuming %s from round %d (of %d)"
                      % (stamp, start_round, num_rounds))
                if start_round >= num_rounds:
                    print("==> all %d rounds done, nothing to resume" % num_rounds)
                    continue

        deploy_dic_path = merge(config.DIC_PATH, {
            "PATH_TO_MODEL": os.path.join(
                "model", in_args.memo, traffic_file + "_" + stamp),
            "PATH_TO_WORK_DIRECTORY": os.path.join(
                "records", in_args.memo, traffic_file + "_" + stamp),
            "PATH_TO_DATA": (
                os.path.join(in_args.sumo_dir, os.path.splitext(traffic_file)[0])
                if in_args.sumo
                else os.path.join("data", template, str(road_net))),
            "PATH_TO_ERROR": os.path.join("errors", in_args.memo)
        })

        # ── SUMO: parse the network before the agents are created ──
        # NUM_INTERSECTIONS and ACTION_MASK depend on net.xml, and generator.py
        # creates the agents before the environment, so they are computed here.
        if in_args.sumo:
            from utils.env_factory import prepare_sumo_conf
            prepare_sumo_conf(deploy_dic_traffic_env_conf,
                              deploy_dic_path["PATH_TO_DATA"])

        use_mp = in_args.multi_process and not in_args.single_process
        if use_mp:
            ppl = Process(
                target=pipeline_wrapper,
                args=(deploy_dic_agent_conf,
                      deploy_dic_traffic_env_conf,
                      deploy_dic_path))
            process_list.append(ppl)
        else:
            pipeline_wrapper(
                dic_agent_conf=deploy_dic_agent_conf,
                dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                dic_path=deploy_dic_path)

    if in_args.multi_process and not in_args.single_process:
        for i in range(0, len(process_list), in_args.workers):
            i_max = min(len(process_list), i + in_args.workers)
            for j in range(i, i_max):
                print(j)
                print("start_traffic")
                process_list[j].start()
                print("after_traffic")
            for k in range(i, i_max):
                print("traffic to join", k)
                process_list[k].join()
                print("traffic finish join", k)

    return in_args.memo


if __name__ == "__main__":
    args = parse_args()
    main(args)
