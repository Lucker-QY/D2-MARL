"""
env_factory.py
==============
Chooses the CityFlow or SUMO environment from the config and, before the agents
are built, writes the size and action mask of a SUMO network into the config.

Why prepare_sumo_conf is needed:
    In utils/generator.py the agents are created before the environment, but
    NUM_INTERSECTIONS and ACTION_MASK depend on the network. Both depend only on
    net.xml and are obtained by parsing it with sumolib, without starting SUMO.
    Calling this function once before the pipeline gives the agents the right values.

Usage (in run_d2marl.py, after dic_traffic_env_conf has been built):
    from utils.env_factory import prepare_sumo_conf
    if dic_traffic_env_conf.get("SIMULATOR") == "sumo":
        prepare_sumo_conf(dic_traffic_env_conf, dic_path["PATH_TO_WORK_DIRECTORY"])
"""

import os


def _is_sumo(dic_traffic_env_conf):
    return str(dic_traffic_env_conf.get("SIMULATOR", "cityflow")).lower() == "sumo"


def make_env(path_to_log, path_to_work_directory, dic_traffic_env_conf,
             save_replay=False, replay_dir=None):
    """Used by all three call sites (generator / oneline / model_test)."""
    if _is_sumo(dic_traffic_env_conf):
        from d2marl_sumo.sumo_env import SumoEnv
        return SumoEnv(
            path_to_log=path_to_log,
            path_to_work_directory=path_to_work_directory,
            dic_traffic_env_conf=dic_traffic_env_conf,
            save_replay=save_replay,
            replay_dir=replay_dir,
        )
    from .cityflow_env import CityFlowEnv
    return CityFlowEnv(
        path_to_log=path_to_log,
        path_to_work_directory=path_to_work_directory,
        dic_traffic_env_conf=dic_traffic_env_conf,
        save_replay=save_replay,
        replay_dir=replay_dir,
    )


def prepare_sumo_conf(dic_traffic_env_conf, path_to_work_directory, verbose=True):
    """
    Parse net.xml and write into conf:
        NUM_INTERSECTIONS  number of signalised intersections (cannot be NUM_ROW*NUM_COL on non-grid networks)
        ACTION_MASK        [num_agents, NUM_PHASES], 1 = phase exists at this intersection
        SUMO_INTER_IDS     intersection ids, in the row order of ACTION_MASK
        NUM_ROW / NUM_COL  set to (N, 1) for code that multiplies them
    """
    from d2marl_sumo.sumo_topology import NetworkTopology

    net_file = os.path.join(path_to_work_directory,
                            dic_traffic_env_conf["SUMO_NET_FILE"])
    if not os.path.exists(net_file):
        raise FileNotFoundError("SUMO network file not found: {0}".format(net_file))

    topo = NetworkTopology(
        net_file,
        top_k=dic_traffic_env_conf["TOP_K_ADJACENCY"],
        max_phases=dic_traffic_env_conf["NUM_PHASES"],
    )

    inter_ids = topo.inter_ids
    n = len(inter_ids)
    if n == 0:
        raise RuntimeError("no controllable signalised intersection in {0}".format(net_file))

    mask = [topo.topologies[i].action_mask for i in inter_ids]

    # For methods with one agent per intersection (rule-based controllers and
    # PressLight) NUM_AGENTS must follow the real number of intersections; the
    # caller guessed NUM_ROW*NUM_COL, which is wrong on non-grid networks.
    # Parameter-sharing methods always have NUM_AGENTS = 1.
    _old_n = dic_traffic_env_conf.get("NUM_INTERSECTIONS")
    if dic_traffic_env_conf.get("NUM_AGENTS", 1) != 1 and _old_n != n:
        print("[SUMO]   NUM_AGENTS {0} -> {1} (number of intersections)".format(
            dic_traffic_env_conf["NUM_AGENTS"], n))
        dic_traffic_env_conf["NUM_AGENTS"] = n

    dic_traffic_env_conf["NUM_INTERSECTIONS"] = n
    dic_traffic_env_conf["NUM_ROW"] = n
    dic_traffic_env_conf["NUM_COL"] = 1
    dic_traffic_env_conf["ACTION_MASK"] = mask
    dic_traffic_env_conf["SUMO_INTER_IDS"] = inter_ids

    if verbose:
        stats, _rows = topo.report()
        n_hetero = sum(1 for m in mask if 0 in m)
        print("[SUMO] network {0}".format(os.path.basename(net_file)))
        print("[SUMO]   signalised intersections: {0}".format(n))
        print("[SUMO]   phase count histogram : {0}".format(stats["phase_count_hist"]))
        print("[SUMO]   leg count histogram   : {0}".format(stats["leg_count_hist"]))
        print("[SUMO]   valid movement slots  : {0}".format(stats["valid_slot_hist"]))
        print("[SUMO]   intersections needing a mask: {0} / {1}".format(n_hetero, n))
        if stats["num_truncated_phases"]:
            print("[SUMO]   [warning] {0} intersections have more phases than NUM_PHASES={1}; "
                  "the extra phases are not used".format(
                      stats["num_truncated_phases"],
                      dic_traffic_env_conf["NUM_PHASES"]))
    return dic_traffic_env_conf
