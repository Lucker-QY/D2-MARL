"""
sumo_env.py
===========
Traffic signal control environment on SUMO/TraCI with the same interface as
utils/cityflow_env.py::CityFlowEnv, so that models/d2marl_agent.py and
utils/pipeline.py run unchanged.

Main differences from the CityFlow version:

1. Intersections are read from the network topology instead of being built from
   "intersection_{col}_{row}" grid coordinates, so non-grid networks are supported.
2. Approach lanes are mapped to the canonical 12 movement slots by the turn
   direction of their connections; missing slots are masked by slot_mask.
3. The number of phases may differ (2-8); the action space is padded to
   MAX_PHASES and an action_mask is provided.
4. Waiting time is defined exactly as in CityFlow (speed < 0.1 m/s counts as
   waiting, timed continuously, reset when the vehicle moves), so Awt / Mwt / C_d
   are comparable across the two simulators.
5. Lane-level queue traces (RECORD_QUEUE) are recorded for the spillback
   indicators in ssm_metrics.py.

Requires traci and sumolib (shipped with SUMO).
"""

import os
import sys
import csv
import time
import json
import pickle
import numpy as np
from collections import defaultdict

if "SUMO_HOME" in os.environ:
    _tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if _tools not in sys.path:
        sys.path.append(_tools)

import traci                                            # noqa: E402
import traci.constants as tc                            # noqa: E402
import sumolib                                          # noqa: E402

from .sumo_topology import NetworkTopology, NUM_SLOTS   # noqa: E402

# Retries around the SUMO start (useful when many jobs share a node).
_START_RETRIES = int(os.environ.get("SUMO_START_RETRIES", "5"))
_START_BACKOFF = float(os.environ.get("SUMO_START_BACKOFF", "3.0"))


WAIT_SPEED_THRESHOLD = 0.1     # m/s, same waiting criterion as CityFlow


# ==========================================================================
class SumoIntersection:
    """Agent-side view of one signalised intersection; field names follow cityflow_env.Intersection."""

    def __init__(self, tls_id, topo, dic_traffic_env_conf, adjacency_row, env_ref):
        self.inter_id = tls_id
        self.inter_name = tls_id
        self.topo = topo
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.env_ref = env_ref

        self.slot_lanes = topo.slot_lanes
        self.exit_slot_lanes = topo.exit_slot_lanes
        self.slot_mask = topo.slot_mask
        self.list_entering_lanes = topo.entering_lanes
        self.list_exiting_lanes = topo.exiting_lanes
        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        self.adjacency_row = adjacency_row
        self.neighbor_ENWS = [None] * 4          # meaningless on non-grid networks, kept as placeholder

        self.green_phases = topo.green_phases
        self.num_actions = topo.num_actions
        self.action_mask = topo.action_mask

        # ===== signal state machine, line-by-line equivalent to cityflow_env.Intersection =====
        # Phase indices are 1-based (1..num_actions); -1 means all-yellow.
        # This matters: FixedtimeAgent looks up DIC_PHASE_MAP{1:0,...} and set_signal
        # sets next_phase_to_set_index = action + 1; 0-based indices would shift
        # the timing by one phase.
        self.all_yellow_phase_index = -1
        self.current_phase_index = 1
        self.previous_phase_index = 1
        self.current_phase_duration = -1
        self.next_phase_to_set_index = None
        self.all_yellow_flag = False
        self.flicker = 0

        self.dic_feature = {}
        self.dic_feature_previous_step = {}
        # Same definition as cityflow_env.Intersection:
        #   enter_time = first time the vehicle is on an approach lane of this intersection
        #   leave_time = time the vehicle leaves the approach lanes of this intersection
        # ATT merges these records over intersections (max(leave) - min(enter)), which
        # differs from network entry/exit; the same definition is needed on both simulators.
        self.dic_vehicle_arrive_leave_time = {}
        self._prev_entering_vehicles = set()

        # Cache of the phase state strings. The tlLogic is laid out as G0 Y0 G1 Y1 ...;
        # green_phases holds the program index of each green, the next index is its yellow.
        logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        self._phase_states = [p.state for p in logic.phases]
        self._green_state = [self._phase_states[i] for i in self.green_phases]
        self._yellow_state = []
        for i in self.green_phases:
            j = i + 1
            self._yellow_state.append(
                self._phase_states[j] if j < len(self._phase_states)
                else self._phase_states[i])
        self._apply(self._green_state[0])

        # ---- downstream storage (non-grid adaptation, Section IV-C) ----
        # Storage capacity of the exit lanes of each movement slot = total lane length
        # divided by the space of one vehicle (7.5 m). Static, computed once.
        _SLOT = float(dic_traffic_env_conf.get("VEH_SLOT_METERS", 7.5))
        self.slot_out_capacity = []
        for lanes in self.exit_slot_lanes:
            cap = 0.0
            for l in lanes:
                try:
                    cap += traci.lane.getLength(l) / _SLOT
                except traci.TraCIException:
                    pass
            self.slot_out_capacity.append(cap)
        # refreshed in update_current_measurements
        self.slot_out_free = list(self.slot_out_capacity)
        self.blocked_min_free = float(dic_traffic_env_conf.get("SPILLBACK_MIN_FREE", 2.0))

    # ------------------------------------------------------------------
    # signal control
    # ------------------------------------------------------------------
    def _apply(self, state_str):
        traci.trafficlight.setRedYellowGreenState(self.inter_name, state_str)
        # Replay recording: the signal states are written here, so they are logged
        # here (no SaveTLSStates output and no extra TraCI round trip).
        # Only active in the save_replay episode (when the env has created _tls_trace).
        tr = getattr(self.env_ref, "_tls_trace", None)
        if tr is not None:
            # De-duplicate **per intersection**: intersections write in turn within
            # a step, so comparing with the last global entry would never match.
            last = self.env_ref._tls_last
            if last.get(self.inter_name) != state_str:
                last[self.inter_name] = state_str
                t = getattr(self.env_ref, "current_time", 0.0)
                tr.append((float(t), self.inter_name, state_str))

    def update_previous_measurements(self):
        """Counterpart of cityflow_env.Intersection.update_previous_measurements."""
        self.previous_phase_index = self.current_phase_index

    def set_signal(self, action, action_pattern, yellow_time, path_to_log=None):
        """
        Line-by-line copy of cityflow_env.Intersection.set_signal.

        Why line by line: the actual timing is not a parameter but emerges from
        FixedtimeAgent.choose_action + the MIN_ACTION_TIME inner loop of env.step
        + set_signal + the reset rule of current_phase_duration.
        In particular, current_phase_index is set to -1 when yellow starts, so
        current_phase_index != previous in update_current_measurements and
        current_phase_duration is reset to 1. Because of this reset FixedtimeAgent
        sees time_this_phase < FIXED_TIME at the next decision point and holds one
        more decision interval; without it the cycle would be 80 s instead of 120 s.
        """
        action = int(action)
        if action >= self.num_actions:            # fallback for intersections with fewer phases
            action = action % self.num_actions


        if self.all_yellow_flag:
            self.flicker = 0
            if self.current_phase_duration >= yellow_time:
                self.current_phase_index = self.next_phase_to_set_index
                self._apply(self._green_state[self.current_phase_index - 1])
                self.all_yellow_flag = False
        else:
            if action_pattern == "switch":
                if action == 0:
                    self.next_phase_to_set_index = self.current_phase_index
                else:
                    self.next_phase_to_set_index = (
                        self.current_phase_index % self.num_actions) + 1
            else:                                  # "set"
                self.next_phase_to_set_index = action + 1

            if self.current_phase_index == self.next_phase_to_set_index:
                pass
            else:
                # switch to the yellow phase of the current green
                idx = self.current_phase_index - 1
                if 0 <= idx < len(self._yellow_state):
                    self._apply(self._yellow_state[idx])
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True
                self.flicker = 1

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def _slot_agg(self, per_lane, reducer=sum):
        """Aggregate lane-level values into the 12 movement slots; missing slots are 0."""
        out = []
        for lanes in self.slot_lanes:
            vals = [per_lane.get(l, 0) for l in lanes]
            out.append(float(reducer(vals)) if vals else 0.0)
        return out

    def update_current_measurements(self, sim_state):
        self.dic_feature_previous_step = self.dic_feature
        # same as cityflow_env.py: reset to 1 whenever the phase changes
        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        halting = sim_state["lane_halting"]
        counts = sim_state["lane_count"]
        waits = sim_state["vehicle_wait"]
        lane_vehicles = sim_state["lane_vehicles"]

        f = {}
        # 1-based, -1 for yellow. FixedtimeAgent's DIC_PHASE_MAP and the
        # `if state["cur_phase"][0] == -1` branches rely on this convention.
        # -- arrival/departure times on the approach lanes (as in CityFlow) --
        cur_entering = set()
        for lanes in self.slot_lanes:
            for l in lanes:
                cur_entering.update(lane_vehicles.get(l, ()))
        now = self.env_ref.current_time if self.env_ref else 0.0
        for v in cur_entering - self._prev_entering_vehicles:
            if v not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[v] = {
                    "enter_time": now, "leave_time": float("nan")}
        for v in self._prev_entering_vehicles - cur_entering:
            if v in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[v]["leave_time"] = now
        self._prev_entering_vehicles = cur_entering

        f["cur_phase"] = [self.current_phase_index]
        f["time_this_phase"] = [self.current_phase_duration]

        f["lane_num_vehicle"] = self._slot_agg(counts)
        f["lane_num_vehicle_downstream"] = [
            float(sum(counts.get(l, 0) for l in lanes)) if lanes else 0.0
            for lanes in self.exit_slot_lanes
        ]
        f["delta_lane_num_vehicle"] = [
            f["lane_num_vehicle"][i] - f["lane_num_vehicle_downstream"][i]
            for i in range(NUM_SLOTS)
        ]
        f["lane_num_waiting_vehicle_in"] = self._slot_agg(halting)
        f["lane_num_waiting_vehicle_out"] = [
            float(sum(halting.get(l, 0) for l in lanes)) if lanes else 0.0
            for lanes in self.exit_slot_lanes
        ]

        # free downstream storage in vehicle spaces, and its ratio c (Eq. of Section IV-C);
        # a slot without downstream storage uses c = 1
        self.slot_out_free = [
            max(0.0, cap - sum(counts.get(l, 0) for l in lanes))
            for cap, lanes in zip(self.slot_out_capacity, self.exit_slot_lanes)
        ]
        cap_ratio = [
            (self.slot_out_free[i] / self.slot_out_capacity[i])
            if self.slot_out_capacity[i] > 1e-6 else 1.0
            for i in range(NUM_SLOTS)
        ]
        f["lane_downstream_free_capacity"] = cap_ratio

        f["traffic_movement_pressure_queue"] = [
            a - b for a, b in zip(f["lane_num_waiting_vehicle_in"],
                                  f["lane_num_waiting_vehicle_out"])
        ]
        # efficient pressure: downstream discounted by its number of lanes (as in CityFlow)
        f["traffic_movement_pressure_queue_efficient"] = [
            a - b / max(1.0, m) for a, b, m in zip(
                f["lane_num_waiting_vehicle_in"],
                f["lane_num_waiting_vehicle_out"],
                [max(1, len(lanes)) for lanes in self.exit_slot_lanes])
        ]
        # capacity-discounted pressure m * min(1, c / c0) (-spillback); c0 = SPILLBACK_PRESSURE_R0
        _r0 = float(self.dic_traffic_env_conf.get("SPILLBACK_PRESSURE_R0", 0.2))
        f["traffic_movement_pressure_queue_capacity"] = [
            p * min(1.0, r / _r0) if _r0 > 1e-6 else p
            for p, r in zip(f["traffic_movement_pressure_queue_efficient"], cap_ratio)
        ]
        f["traffic_movement_pressure_num"] = [
            a - b for a, b in zip(f["lane_num_vehicle"],
                                  f["lane_num_vehicle_downstream"])
        ]
        # ---- segment occupancy for LLMLight (12 slots x 4 cells) ----
        # Same definition as LLMTSCS/utils/my_utils.py:get_state_detail:
        #   lane_pos = lane length - distance travelled on the lane (= distance to the stop line)
        #   bins  (0, L/10] -> 0   (L/10, L/3] -> 1   (L/3, 2L/3] -> 2   rest -> 3
        #   only approaching vehicles with speed > 0.1 m/s; stopped vehicles are in queue_len,
        #   counting them twice would change the numbers in the prompt.
        if "lane_cells" in self.dic_traffic_env_conf.get("LIST_STATE_FEATURE", []):
            veh_pos = sim_state.get("veh_pos", {})
            veh_speed = sim_state.get("veh_speed", {})
            lane_len = sim_state.get("lane_len", {})
            cells = []
            for lanes in self.slot_lanes:
                c = [0, 0, 0, 0]
                for l in lanes:
                    L = float(lane_len.get(l, 0.0)) or 1.0
                    for v in lane_vehicles.get(l, ()):
                        if veh_speed.get(v, 0.0) <= 0.1:
                            continue
                        pos = L - float(veh_pos.get(v, 0.0))
                        if pos <= L / 10.0:
                            c[0] += 1
                        elif pos <= L / 3.0:
                            c[1] += 1
                        elif pos <= L * 2.0 / 3.0:
                            c[2] += 1
                        else:
                            c[3] += 1
                cells.append(c)
            f["lane_cells"] = cells

        f["lane_enter_running_part"] = [
            c - q for c, q in zip(f["lane_num_vehicle"],
                                  f["lane_num_waiting_vehicle_in"])
        ]

        # -- delay-risk features (extended state) --
        hard_limit = self.dic_traffic_env_conf.get("WAIT_HARD_LIMIT", 90.0)
        count_thresh = self.dic_traffic_env_conf.get("WAIT_COUNT_THRESHOLD", 60.0)

        lane_max_wait, lane_burden = [], []
        lane_raw_max_wait, lane_over_limit, lane_raw_over_limit = [], [], []
        for lanes in self.slot_lanes:
            max_t, n_wait, n_over = 0.0, 0, 0
            for l in lanes:
                for v in lane_vehicles.get(l, ()):
                    wt = waits.get(v, 0.0)
                    if wt > 0:
                        n_wait += 1
                        max_t = max(max_t, wt)
                        if wt > count_thresh:
                            n_over += 1
            lane_max_wait.append(max_t)
            lane_burden.append((max_t / hard_limit) * n_wait)
            lane_raw_max_wait.append(max_t / hard_limit)
            lane_over_limit.append(n_over / 10.0)
            lane_raw_over_limit.append(float(n_over))

        f["lane_max_waiting_time"] = lane_max_wait
        f["lane_delay_burden"] = lane_burden
        f["lane_raw_max_wait"] = lane_raw_max_wait
        f["lane_over_limit_num"] = lane_over_limit
        f["lane_raw_over_limit_num"] = lane_raw_over_limit

        f["pressure"] = f["lane_num_waiting_vehicle_in"] + \
                        [-x for x in f["lane_num_waiting_vehicle_out"]]
        f["adjacency_matrix"] = self.adjacency_row

        # slot and action masks, used by the agents to ignore missing movements/phases
        f["slot_mask"] = self.slot_mask
        f["action_mask"] = self.action_mask

        f["downstream_availability"] = [
            1.0 - min(1.0, b / 20.0) for b in f["lane_num_waiting_vehicle_out"]
        ]
        # ---- segmented observation of AttendLight (12 in + 12 out, 4 values each = 96) ----
        # Same definition as LLMTSCS/utils/cityflow_env.py `_orgnize_several_segments_attend`;
        # AttendLightAgent.build_network expects num_lane*8 = 96 inputs:
        #   obs_length = fixed 100 m segments (not a fraction of the lane length)
        #   part1 = 0-100 m from the stop line, part2 = 100-200 m, part3 = 200-300 m
        #   only moving vehicles (speed > 0.1 m/s); stopped vehicles are in the queue
        #   layout = [part1, part2, part3, queue] per slot, 12 incoming then 12 outgoing
        # CityFlow's vehicle_distance (distance travelled on the lane) and SUMO's
        # getLanePosition have the same definition.
        if "num_in_seg_attend" in self.dic_traffic_env_conf.get("LIST_STATE_FEATURE", []):
            veh_pos = sim_state.get("veh_pos", {})
            veh_speed = sim_state.get("veh_speed", {})
            lane_len = sim_state.get("lane_len", {})

            def _segs(lanes):
                p = [0.0, 0.0, 0.0]
                for l in lanes:
                    L = float(lane_len.get(l, 0.0))
                    for v in lane_vehicles.get(l, ()):
                        if veh_speed.get(v, 0.0) <= 0.1:
                            continue
                        d = float(veh_pos.get(v, 0.0))
                        if d > L - 100.0:
                            p[0] += 1.0
                        elif L - 200.0 < d <= L - 100.0:
                            p[1] += 1.0
                        elif L - 300.0 < d <= L - 200.0:
                            p[2] += 1.0
                return p

            total_in, total_out = [], []
            for i in range(NUM_SLOTS):
                total_in.extend(_segs(self.slot_lanes[i])
                                + [f["lane_num_waiting_vehicle_in"][i]])
                total_out.extend(_segs(self.exit_slot_lanes[i])
                                 + [f["lane_num_waiting_vehicle_out"][i]])
            f["num_in_seg_attend"] = total_in + total_out

        self.dic_feature = f

    # ------------------------------------------------------------------
    def get_dic_vehicle_arrive_leave_time(self):
        return self.dic_vehicle_arrive_leave_time

    def get_feature(self):
        return self.dic_feature

    def get_state(self, list_state_features):
        return {k: self.dic_feature[k] for k in list_state_features
                if k in self.dic_feature}

    def get_reward(self, dic_reward_info):
        """Equivalent to cityflow_env.Intersection.get_reward term by term (reward of the paper)."""
        dic_reward = {}
        dic_reward["pressure"] = float(np.absolute(np.sum(self.dic_feature["pressure"])))
        dic_reward["queue_length"] = float(
            np.absolute(np.sum(self.dic_feature["lane_num_waiting_vehicle_in"])))

        if "max_wait_penalty" in dic_reward_info:
            soft = self.dic_traffic_env_conf.get("WAIT_SOFT_LIMIT", 60.0)
            hard = self.dic_traffic_env_conf.get("WAIT_HARD_LIMIT", 90.0)
            cap = self.dic_traffic_env_conf.get("MAX_PENALTY_CAP", 2000.0)
            delta = self.dic_traffic_env_conf.get("HUBER_DELTA", 30.0)

            total = 0.0
            for w in self.dic_feature["lane_max_waiting_time"]:
                if w > hard:
                    excess = w - hard
                    if excess <= delta:
                        total += 20.0 + (excess ** 2) / 50.0
                    else:
                        total += (20.0 + (delta ** 2) / 50.0
                                  + (excess - delta) * (delta / 25.0))
                elif w > soft:
                    total += ((w - soft) / (hard - soft)) * 10.0
            dic_reward["max_wait_penalty"] = min(total, cap)

        if "over_limit_penalty" in dic_reward_info:
            dic_reward["over_limit_penalty"] = float(
                sum(self.dic_feature.get("lane_raw_over_limit_num", [0] * NUM_SLOTS)))

        if "blocked_green_penalty" in dic_reward_info:
            # queued vehicles on green movements whose exit has fewer than
            # SPILLBACK_MIN_FREE free spaces (-spillback)
            rows = getattr(self.topo, "phase_slot_matrix", None)
            k_ph = self.current_phase_index - 1
            blocked = 0.0
            if rows and 0 <= k_ph < len(rows):
                q_in = self.dic_feature["lane_num_waiting_vehicle_in"]
                for i, v in enumerate(rows[k_ph]):
                    if (v and i < len(self.slot_out_free) and i < len(q_in)
                            and self.slot_out_free[i] < self.blocked_min_free):
                        blocked += float(q_in[i])
            dic_reward["blocked_green_penalty"] = blocked

        reward = 0.0
        for k, w in dic_reward_info.items():
            if w != 0 and k in dic_reward:
                reward += w * dic_reward[k]
        return reward


# ==========================================================================
class SumoEnv:
    """SUMO environment with the interface of CityFlowEnv."""

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf,
                 save_replay=False, replay_dir=None):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.save_replay = save_replay
        self.replay_dir = replay_dir or path_to_log

        self.net_file = os.path.join(path_to_work_directory,
                                     dic_traffic_env_conf["SUMO_NET_FILE"])
        self.rou_file = os.path.join(path_to_work_directory,
                                     dic_traffic_env_conf["SUMO_ROU_FILE"])
        self.label = dic_traffic_env_conf.get("SUMO_LABEL", "d2marl")
        self._begin = float(dic_traffic_env_conf.get("SUMO_BEGIN", 0))

        self.current_time = 0
        self.list_intersection = []
        self.list_inter_log = []
        self.id_to_index = {}

        # metric accumulators
        # waiting_vehicle_list has the same name and structure as in cityflow_env.CityFlowEnv:
        #   {v_id: {"time": current continuous wait (s), "link": current segment, "recorded_severe_delay": bool}}
        # It holds only vehicles that are currently waiting. Awt in model_test.py is the
        # mean of these times at every decision step, so the names must match.
        self.waiting_vehicle_list = {}
        self.vehicle_total_wait_time = defaultdict(float)  # cumulative wait W_v
        # The longest single continuous wait of every vehicle; Mwt is its network maximum.
        # Deriving Mwt from delay_events_over_90s.csv only works if some wait exceeds
        # 90 s; otherwise the file is empty and Mwt could only be reported as "<90"
        # (right-censored). Recording the value directly has no such ceiling, and no
        # lower threshold would fix it, because the longest single wait may stay below it.
        self.vehicle_max_single_wait = defaultdict(float)
        self.dic_vehicle_arrive_leave_time = {}
        self.severe_delay_events = []
        self._queue_rows = []

        # Only AttendLight (num_in_seg_attend) and LLMLight (lane_cells) need the
        # per-vehicle position/speed; other methods pay nothing for it.
        _feats = dic_traffic_env_conf.get("LIST_STATE_FEATURE", [])
        self._need_veh_detail = ("lane_cells" in _feats
                                 or "num_in_seg_attend" in _feats)
        self._lane_len_cache = {}

        if dic_traffic_env_conf["MIN_ACTION_TIME"] <= dic_traffic_env_conf["YELLOW_TIME"]:
            raise ValueError("MIN_ACTION_TIME must be larger than YELLOW_TIME")

    # ------------------------------------------------------------------
    def _sumo_cmd(self):
        # save_replay uses sumo-gui only if a display is available; on compute nodes
        # without X, sumo-gui exits immediately and TraCI reports
        # "TraCI server already finished". Headless runs use plain sumo instead.
        want_gui = self.save_replay and bool(os.environ.get("DISPLAY"))
        binary = sumolib.checkBinary("sumo-gui" if want_gui else "sumo")
        _TTT = str(os.environ.get("D2MARL_TIME_TO_TELEPORT", "-1"))
        if _TTT != "-1" and not getattr(SumoEnv, "_ttt_announced", False):
            # announce the non-default teleport setting in the log
            print("[sumo] --time-to-teleport = %s (non-default; blocked vehicles are "
                  "teleported, so Mwt no longer includes permanent gridlock)" % _TTT)
            SumoEnv._ttt_announced = True
        cmd = [
            binary,
            "-n", self.net_file,
            "-r", self.rou_file,
            "--step-length", str(self.dic_traffic_env_conf["INTERVAL"]),
            # The demand of the RESCO non-grid network does not start at time 0:
            #   ingolstadt21 departures at 57600-61201 s (16:00 evening peak)
            #   cologne8     departures at 25200-28798 s (07:00 morning peak)
            # With --begin 0 no vehicle would be simulated. SUMO_BEGIN comes from the
            # scenario config and is 0 for the grid scenarios.
            "--begin", str(self._begin),
            "--end", str(self._begin + self.dic_traffic_env_conf["RUN_COUNTS"]),
            "--seed", str(self.dic_traffic_env_conf.get("SEED", 0)),
            # Default -1 = teleporting disabled. SUMO's default of 300 s would move
            # vehicles that cannot move downstream, removing extreme delays so that Mwt
            # could not be measured. The price: on real non-grid networks a spillback on
            # a short link becomes a permanent gridlock. Grid scenarios with long links
            # are hardly affected. D2MARL_TIME_TO_TELEPORT overrides the value for a
            # sensitivity run (teleporting on vs. off), which shows how much of Mwt is
            # caused by gridlock rather than by the signal policy.
            "--time-to-teleport", _TTT,
            "--waiting-time-memory", "10000", # keep getAccumulatedWaitingTime from truncating
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--collision.action", "warn",
            "--collision-output", os.path.join(self.path_to_log, "collisions.xml"),
        ]
        # Headless runs do not write --netstate-dump (all vehicle states every step,
        # hundreds of MB per episode, and sumo-gui cannot load it as a replay).
        # _dump_tls_trace records the executed signal states instead (tens of KB).
        if _TTT != "-1":
            # the summary has a cumulative teleports column per step (gridlocked vehicles);
            # written only for the teleport sensitivity run.
            cmd += ["--summary-output",
                    os.path.join(self.path_to_log, "summary.xml")]
        if self.dic_traffic_env_conf.get("ENABLE_SSM", False):
            from .ssm_metrics import sumo_cmd_ssm_flags
            cmd += sumo_cmd_ssm_flags(
                os.path.join(self.path_to_log, "ssm.xml"),
                measure_range=self.dic_traffic_env_conf.get("SSM_RANGE", 50.0))
        return cmd

    # ------------------------------------------------------------------
    def reset(self):
        try:
            traci.close()
        except Exception:
            pass

        os.makedirs(self.path_to_log, exist_ok=True)

        # Under high concurrency traci.start can fail: it probes a free port and then
        # lets SUMO bind it, and another job may take the port in between. traci
        # retries itself, but raises once its retries are used up; an outer retry
        # with back-off bridges this window.
        last = None
        for attempt in range(_START_RETRIES):
            try:
                traci.start(self._sumo_cmd(), label=self.label)
                break
            except Exception as e:       # traci.TraCIException or a socket error
                last = e
                try:
                    traci.close()
                except Exception:
                    pass
                wait = _START_BACKOFF * (attempt + 1)
                print("[sumo] start attempt %d/%d failed (%s), retrying in %.1fs"
                      % (attempt + 1, _START_RETRIES, e, wait))
                time.sleep(wait)
        else:
            raise RuntimeError(
                "SUMO failed to start %d times in a row, last error: %s" % (_START_RETRIES, last))
        self._closed = False
        self._lane_len_cache = {}   # new simulation instance: clear the lane length cache
        # Recording switch: record if the list exists, not if it is None. Cleared at
        # every reset so that several evaluation rounds are not merged into one trace.
        self._tls_trace = [] if self.save_replay else None
        self._tls_last = {}         # tls_id -> last written state string, for de-duplication

        self.net_topo = NetworkTopology(
            self.net_file,
            top_k=self.dic_traffic_env_conf["TOP_K_ADJACENCY"],
            max_phases=self.dic_traffic_env_conf["NUM_PHASES"])

        self.list_intersection = []
        for tls_id in self.net_topo.inter_ids:
            self.list_intersection.append(SumoIntersection(
                tls_id, self.net_topo.topologies[tls_id],
                self.dic_traffic_env_conf,
                self.net_topo.adjacency[tls_id], self))
        self.id_to_index = {t: i for i, t in enumerate(self.net_topo.inter_ids)}
        self.list_inter_log = [[] for _ in self.list_intersection]

        # write the real network size back into conf for the agents
        self.dic_traffic_env_conf["NUM_INTERSECTIONS"] = len(self.list_intersection)

        # Cross-check with the mask precomputed by prepare_sumo_conf. Both build a
        # NetworkTopology; if order or content differed, the agents' masks would be
        # misaligned without any visible sign in the training curves.
        conf_mask = self.dic_traffic_env_conf.get("ACTION_MASK")
        if conf_mask is not None:
            env_mask = [i.action_mask for i in self.list_intersection]
            if [list(m) for m in conf_mask] != [list(m) for m in env_mask]:
                raise RuntimeError(
                    "ACTION_MASK does not match the topology of the environment:\n"
                    "  conf: {0}\n  env : {1}\n"
                    "check that prepare_sumo_conf and SumoEnv read the same net.xml".format(
                        conf_mask, env_mask))
        else:
            self.dic_traffic_env_conf["ACTION_MASK"] = [
                i.action_mask for i in self.list_intersection]

        self.current_time = 0
        self.waiting_vehicle_list = {}
        self.vehicle_total_wait_time.clear()
        self.vehicle_max_single_wait.clear()
        self.dic_vehicle_arrive_leave_time = {}
        self.severe_delay_events = []
        self._queue_rows = []

        sim_state = self._collect()
        for inter in self.list_intersection:
            inter.update_current_measurements(sim_state)
        state, _ = self.get_state()
        return state

    # ------------------------------------------------------------------
    def _collect(self):
        """Fetch all TraCI quantities of this step at once instead of per intersection."""
        lane_ids = set()
        for inter in self.list_intersection:
            lane_ids.update(inter.list_lanes)

        lane_count, lane_halting, lane_vehicles = {}, {}, {}
        for l in lane_ids:
            try:
                lane_vehicles[l] = traci.lane.getLastStepVehicleIDs(l)
                lane_count[l] = len(lane_vehicles[l])
                lane_halting[l] = traci.lane.getLastStepHaltingNumber(l)
            except traci.TraCIException:
                lane_vehicles[l], lane_count[l], lane_halting[l] = (), 0, 0

        # LLMLight's prompt and AttendLight's segments need the distance of each vehicle
        # to the stop line, i.e. per-vehicle lanePosition/speed queries, which are much
        # more expensive; they are fetched only if lane_cells / num_in_seg_attend is requested.
        veh_pos, veh_speed = {}, {}
        lane_len = {}
        if self._need_veh_detail:
            # lane lengths are static: query once per episode and cache
            if not self._lane_len_cache:
                for l in lane_ids:
                    try:
                        self._lane_len_cache[l] = traci.lane.getLength(l)
                    except traci.TraCIException:
                        self._lane_len_cache[l] = 1.0
            lane_len = self._lane_len_cache
            for v, r in traci.vehicle.getAllSubscriptionResults().items():
                veh_pos[v] = r.get(tc.VAR_LANEPOSITION, 0.0)
                veh_speed[v] = r.get(tc.VAR_SPEED, 0.0)

        return {
            "lane_count": lane_count,
            "lane_halting": lane_halting,
            "lane_vehicles": lane_vehicles,
            "veh_pos": veh_pos,
            "veh_speed": veh_speed,
            "lane_len": lane_len,
            # used by SumoIntersection for lane_max_waiting_time and similar features;
            # the current continuous wait from waiting_vehicle_list, as
            # Intersection._update_feature reads env_ref.waiting_vehicle_list in CityFlow.
            "vehicle_wait": {v: r["time"]
                             for v, r in self.waiting_vehicle_list.items()},
        }

    # ------------------------------------------------------------------
    def _update_waiting(self):
        """
        Line-by-line counterpart of cityflow_env.CityFlowEnv._update_waiting_vehicle_list.

        Key point (otherwise Mwt/Cd would be biased upward on SUMO):
        a waiting event ends not only when the vehicle moves but **also when it changes
        segment**. CityFlow uses v_info['drivable'], here the TraCI lane id is used.
        """
        dt = self.dic_traffic_env_conf["INTERVAL"]
        alive = set(traci.vehicle.getIDList())

        for v in alive:
            speed = traci.vehicle.getSpeed(v)
            link = traci.vehicle.getLaneID(v)

            if speed < WAIT_SPEED_THRESHOLD:
                self.vehicle_total_wait_time[v] += dt
                rec = self.waiting_vehicle_list.get(v)
                if rec is None:
                    self.waiting_vehicle_list[v] = {
                        "time": dt, "link": link, "recorded_severe_delay": False}
                elif rec["link"] != link:
                    self._check_and_record_severe_delay(v)
                    self.waiting_vehicle_list[v] = {
                        "time": dt, "link": link, "recorded_severe_delay": False}
                else:
                    rec["time"] += dt
            else:
                if v in self.waiting_vehicle_list:
                    self._check_and_record_severe_delay(v)
                    self.waiting_vehicle_list.pop(v, None)

        # close the open waits of vehicles that have left the network
        for v in list(self.waiting_vehicle_list.keys()):
            if v not in alive:
                self._check_and_record_severe_delay(v)
                self.waiting_vehicle_list.pop(v, None)

    def _check_and_record_severe_delay(self, v_id, threshold=None):
        """Counterpart of cityflow_env._check_and_record_severe_delay, same field names."""
        if threshold is None:
            threshold = self.dic_traffic_env_conf.get("DELAY_THRESHOLD", 90.0)
        rec = self.waiting_vehicle_list.get(v_id)
        if not rec:
            return
        wt = rec.get("time", 0)
        # update the maximum whether or not the threshold is exceeded; this is what makes Mwt uncensored
        if wt > self.vehicle_max_single_wait.get(v_id, 0.0):
            self.vehicle_max_single_wait[v_id] = wt
        if wt and wt > threshold and not rec.get("recorded_severe_delay", False):
            self.severe_delay_events.append({
                "vehicle_id": v_id,
                "link": rec.get("link", "unknown"),
                "wait_time": round(wt, 1),
                "exceed_time": round(wt - threshold, 1),
                "end_time": self.current_time,
            })
            rec["recorded_severe_delay"] = True

    def _record_queue(self):
        """Record the queue back of each lane for the spillback indicators in ssm_metrics."""
        if not self.dic_traffic_env_conf.get("RECORD_QUEUE", False):
            return
        for inter in self.list_intersection:
            for l in inter.list_entering_lanes:
                try:
                    n = traci.lane.getLastStepHaltingNumber(l)
                    L = traci.lane.getLength(l)
                    vehs = traci.lane.getLastStepVehicleIDs(l)
                    back = 0.0
                    for v in vehs:
                        if traci.vehicle.getSpeed(v) < WAIT_SPEED_THRESHOLD:
                            back = max(back, L - traci.vehicle.getLanePosition(v))
                except traci.TraCIException:
                    continue
                self._queue_rows.append(
                    {"time": self.current_time, "lane_id": l,
                     "queue_len_veh": n, "queue_back_m": round(back, 2),
                     "lane_length_m": round(L, 2)})

    def dump_queue_csv(self, path):
        if not self._queue_rows:
            return None
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self._queue_rows[0].keys()))
            w.writeheader()
            w.writerows(self._queue_rows)
        return path

    # ------------------------------------------------------------------
    def step(self, action):
        """
        Line-by-line counterpart of cityflow_env.CityFlowEnv.step.

        self.log() must be called in the inner loop: list_inter_log is the source
        of the inter_*.pkl files written by bulk_log_multi_process, from which
        utils/construct_sample.py builds the training samples. Without log() the
        replay buffer stays empty and nothing is learned, while training and the
        reward log look normal.
        """
        min_action_time = self.dic_traffic_env_conf["MIN_ACTION_TIME"]

        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for _ in range(min_action_time - 1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            else:
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(
                np.full_like(action, fill_value=-1).tolist())

        average_reward_action_list = [0] * len(action)
        next_state, done, reward = None, False, None

        for i in range(min_action_time):
            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]

            instant_time = self.get_current_time()
            before_action_feature = self.get_feature()

            self._inner_step(action_in_sec)

            reward = self.get_reward()
            for j in range(len(reward)):
                average_reward_action_list[j] = (
                    average_reward_action_list[j] * i + reward[j]) / (i + 1)

            self.log(cur_time=instant_time,
                     before_action_feature=before_action_feature,
                     action=action_in_sec_display)

            next_state, done = self.get_state()
            if done:
                break

        return next_state, reward, done, average_reward_action_list

    def _inner_step(self, action):
        """Same structure as cityflow_env.CityFlowEnv._inner_step."""
        yellow_time = self.dic_traffic_env_conf["YELLOW_TIME"]

        for inter in self.list_intersection:
            inter.update_previous_measurements()

        # With ACTION_PATTERN="set" CityFlow repeats the action in every inner step;
        # yellow is advanced by the state machine in set_signal, so it must be called every step.
        for idx, inter in enumerate(self.list_intersection):
            inter.set_signal(action[idx],
                             self.dic_traffic_env_conf["ACTION_PATTERN"],
                             yellow_time)

        traci.simulationStep()
        # subtract the start offset so that current_time always starts at 0; RUN_COUNTS,
        # wait timing and log time stamps then need not know the offset.
        self.current_time = traci.simulation.getTime() - self._begin

        self._update_waiting()
        self._record_queue()

        for v in traci.simulation.getDepartedIDList():
            self.dic_vehicle_arrive_leave_time[v] = {
                "enter_time": self.current_time, "leave_time": np.nan}
            # lane_cells / num_in_seg_attend need per-vehicle lanePosition + speed.
            # Two getXxx calls per vehicle are 2N TraCI round trips per step (about 4000
            # at the New York peak), which would stretch an episode to many hours.
            # Instead each vehicle is subscribed once on departure and all results are
            # fetched with one getAllSubscriptionResults per step; subscriptions end on arrival.
            if self._need_veh_detail:
                traci.vehicle.subscribe(v, (tc.VAR_LANEPOSITION, tc.VAR_SPEED))
        for v in traci.simulation.getArrivedIDList():
            if v in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[v]["leave_time"] = self.current_time

        sim_state = self._collect()
        for inter in self.list_intersection:
            inter.update_current_measurements(sim_state)

    # ------------------------------------------------------------------
    def get_feature(self):
        return [i.get_feature() for i in self.list_intersection]

    def get_state(self):
        feats = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        state = [i.get_state(feats) for i in self.list_intersection]
        done = self.current_time >= self.dic_traffic_env_conf["RUN_COUNTS"]
        return state, done

    def get_reward(self):
        info = self.dic_traffic_env_conf["DIC_REWARD_INFO"]
        return [i.get_reward(info) for i in self.list_intersection]

    def get_current_time(self):
        return self.current_time

    def get_max_single_wait(self):
        """{vehicle: longest single continuous wait in seconds}, including a wait still open at the end.

        _check_and_record_severe_delay is only called when a wait **ends** (the vehicle
        moves, changes segment or leaves), so the wait of a vehicle still stuck at the end
        is open. Stuck vehicles matter most; ignoring them would bias Mwt down.
        """
        d = dict(self.vehicle_max_single_wait)
        for v, rec in self.waiting_vehicle_list.items():
            wt = rec.get("time", 0.0)
            if wt > d.get(v, 0.0):
                d[v] = wt
        return d

    def get_severe_delay_events(self, threshold=None):
        """As in CityFlow: close the waits that are still open before returning."""
        if threshold is None:
            threshold = self.dic_traffic_env_conf.get("DELAY_THRESHOLD", 90.0)
        for v_id in list(self.waiting_vehicle_list.keys()):
            self._check_and_record_severe_delay(v_id, threshold)
        return self.severe_delay_events

    def get_traffic_metrics(self):
        total_q = sum(np.sum(i.dic_feature.get("lane_num_waiting_vehicle_in", [0]))
                      for i in self.list_intersection)
        total_p = sum(np.sum(i.dic_feature.get("pressure", [0]))
                      for i in self.list_intersection)
        return {"total_queue_length": float(total_q),
                "total_pressure": float(total_p),
                "num_vehicles": float(len(traci.vehicle.getIDList()))}

    # ------------------------------------------------------------------
    def log(self, cur_time, before_action_feature, action):
        for idx in range(len(self.list_intersection)):
            self.list_inter_log[idx].append(
                {"time": cur_time,
                 "state": before_action_feature[idx],
                 "action": action[idx]})

    def bulk_log_multi_process(self, batch_size=100):
        os.makedirs(self.path_to_log, exist_ok=True)
        for idx, inter in enumerate(self.list_intersection):
            path = os.path.join(self.path_to_log,
                                f"inter_{idx}.pkl")
            with open(path, "wb") as f:
                pickle.dump(self.list_inter_log[idx], f)

        # vehicle-level metrics (input of ATT / Mwt / Gini / C_d)
        veh = os.path.join(self.path_to_log, "vehicle_metrics.json")
        with open(veh, "w") as f:
            json.dump({
                "arrive_leave": {k: v for k, v in
                                 self.dic_vehicle_arrive_leave_time.items()},
                "total_wait": dict(self.vehicle_total_wait_time),
                "severe_delay_events": self.severe_delay_events,
            }, f, default=str)

        self.dump_queue_csv(os.path.join(self.path_to_log, "lane_queue.csv"))

    def batch_log_2(self):
        """
        Same name and behaviour as cityflow_env.CityFlowEnv.batch_log_2:
        exports vehicle_inter_*.csv only, from which model_test computes ATT.
        Called by utils/model_test.py at the end of a test episode.
        """
        import pandas as pd
        os.makedirs(self.path_to_log, exist_ok=True)
        for inter_ind in range(len(self.list_intersection)):
            if int(inter_ind) % 100 == 0:
                print("Batch log for inter ", inter_ind)
            path_to_log_file = os.path.join(
                self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind]\
                .get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle, orient="index")
            df.to_csv(path_to_log_file, na_rep="nan")

        # The lane_queue.csv needed for the spillback indicators is also written here,
        # because evaluation goes through model_test -> batch_log_2 (not
        # bulk_log_multi_process). dump_queue_csv returns immediately when RECORD_QUEUE
        # is off (_queue_rows empty), so other callers are unaffected.
        self.dump_queue_csv(os.path.join(self.path_to_log, "lane_queue.csv"))

    def end_sumo(self):
        """
        Close the TraCI connection. traci.close() waits (wait=True) until SUMO exits,
        so that the files it writes (collisions.xml / ssm.xml) are complete.
        Idempotent: repeated calls do not raise.
        """
        if getattr(self, "_closed", False):
            return
        self._dump_tls_trace()
        try:
            traci.close(wait=True)
        except Exception:
            pass
        finally:
            self._closed = True

    def _dump_tls_trace(self):
        """Write the signal states executed in this episode to tls_trace.csv.

        Together with the net and route files and the same seed, the trace allows
        the episode to be replayed in sumo-gui as a static signal program, without
        the agent, its weights or a Python environment.
        Times are seconds relative to SUMO_BEGIN, as env.current_time.
        """
        tr = getattr(self, "_tls_trace", None)
        if not tr:
            return
        path = os.path.join(self.path_to_log, "tls_trace.csv")
        try:
            os.makedirs(self.path_to_log, exist_ok=True)
            with open(path, "w") as f:
                f.write("time,tls_id,state\n")
                f.write("# begin=%s run_counts=%s step_length=%s seed=%s\n"
                        % (self._begin,
                           self.dic_traffic_env_conf["RUN_COUNTS"],
                           self.dic_traffic_env_conf["INTERVAL"],
                           self.dic_traffic_env_conf.get("SEED", 0)))
                for t, tls, st in tr:
                    f.write("%.2f,%s,%s\n" % (t, tls, st))
            print("[replay] signal trace written to %s (%d changes, %d traffic lights)"
                  % (path, len(tr), len(set(x[1] for x in tr))))
        except (IOError, OSError) as e:
            print("[replay] failed to write %s: %s" % (path, e))

    # utils/generator.py and utils/model_test.py call env.end_cityflow(); the method
    # is kept under that name so that they work with both simulators.
    def end_cityflow(self):
        self.end_sumo()


# alias for code that expects the CityFlow name
CityFlowEnv = SumoEnv
