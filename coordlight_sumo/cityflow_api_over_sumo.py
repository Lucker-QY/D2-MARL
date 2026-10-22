"""
cityflow_api_over_sumo.py
=========================
An engine with **exactly the interface of the CityFlow engine**, implemented on
TraCI, so that CoordLight runs on SUMO without changes to its observation, reward
or model code.

Why not port CoordLight into this code base:
    CoordLight uses PyTorch + PPO (multi-agent actor-critic) + Ray, while D2-MARL
    uses TensorFlow + DQN + multiprocessing with the TD target built in
    prepare_Xs_Y. Porting would mean re-implementing CoordLight in TF, and a
    re-implemented baseline may deviate from the original.

    Only 13 self.eng.* calls in MATSC_CityFlow.py touch the simulator; the
    observation (QDSE), neighbourhood reward and phase logic all use the dicts
    they return. With an engine of the same interface, that file stays unchanged
    and the method itself is provably the same.

Two conventions differ:
  1. Lane numbering: CityFlow inner->outer 0..N-1, SUMO outer->inner.
     CityFlow-style ids are exposed; internally the index is reversed (N-1-k).
  2. Vehicle position: CityFlow's get_vehicle_distance is the distance travelled on
     the current lane, the same as SUMO's getLanePosition.
     (the env computes the distance to the stop line as lane_length - distance on both)
"""

import os
import sys
import time
from collections import OrderedDict, defaultdict

if "SUMO_HOME" in os.environ:
    _tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if _tools not in sys.path:
        sys.path.append(_tools)

import traci                                            # noqa: E402
import sumolib                                          # noqa: E402

# outer retry around the SUMO start (useful with many concurrent processes)
_START_RETRIES = int(os.environ.get("SUMO_START_RETRIES", "5"))
_START_BACKOFF = float(os.environ.get("SUMO_START_BACKOFF", "3.0"))


class CityFlowSumoEngine(object):
    """
    Exposes the same 13 methods as cityflow.Engine:
        get_lane_vehicles / get_lane_vehicle_count /
        get_lane_waiting_vehicle_count / get_vehicle_speed /
        get_vehicle_distance / get_vehicles / get_vehicle_count /
        get_current_time / get_average_travel_time /
        next_step / reset / set_random_seed / set_tl_phase
    """

    HALT_SPEED = 0.1        # same waiting criterion as CityFlow

    def __init__(self, net_file, rou_file, horizon=3600, interval=1.0,
                 seed=0, label="coordlight", enable_ssm=False,
                 ssm_file=None, log_dir=None):
        self.net_file = net_file
        self.rou_file = rou_file
        self.horizon = horizon
        self.interval = interval
        self.seed = seed
        self.label = label
        self.enable_ssm = enable_ssm
        self.ssm_file = ssm_file
        self.log_dir = log_dir or os.path.dirname(net_file)

        self._net = sumolib.net.readNet(net_file, withPrograms=True)
        self._build_lane_maps()
        self._build_tls_maps()

        self._t = 0.0
        self._depart, self._travel = {}, {}
        self._started = False
        self.reset()

    # ------------------------------------------------------------------
    # lane numbering: CityFlow inner->outer, SUMO outer->inner
    # ------------------------------------------------------------------
    def _build_lane_maps(self):
        self.cf2sumo, self.sumo2cf = {}, {}
        self.lane_length = {}
        for edge in self._net.getEdges():
            if edge.getFunction() == "internal":
                continue
            lanes = edge.getLanes()
            n = len(lanes)
            eid = edge.getID()
            for sumo_idx, ln in enumerate(lanes):
                cf_idx = n - 1 - sumo_idx
                cf_id = "{0}_{1}".format(eid, cf_idx)
                sumo_id = ln.getID()
                self.cf2sumo[cf_id] = sumo_id
                self.sumo2cf[sumo_id] = cf_id
                self.lane_length[cf_id] = ln.getLength()
        self._all_cf_lanes = sorted(self.cf2sumo)

    # ------------------------------------------------------------------
    # phase mapping: CityFlow lightphase index 0 = yellow, 1..N = green
    # matches the tlLogic layout (G0 Y0 G1 Y1 ...) produced by d2marl_sumo/cityflow2sumo.py
    # ------------------------------------------------------------------
    def _build_tls_maps(self):
        self.tls_states = {}
        for tls in self._net.getTrafficLights():
            tid = tls.getID()
            progs = tls.getPrograms()
            prog = progs.get("0") or list(progs.values())[0]
            phases = list(prog.getPhases())
            greens, yellows = [], []
            for i, ph in enumerate(phases):
                if "y" in ph.state:
                    yellows.append(ph.state)
                elif any(c in "Gg" for c in ph.state):
                    greens.append(ph.state)
            if not greens:
                continue
            self.tls_states[tid] = {
                "green": greens,
                # yellow: from the tll if present, otherwise derived from the green (G->y)
                "yellow": yellows if yellows else
                          ["".join("y" if c in "Gg" else c for c in greens[0])],
            }

    # ------------------------------------------------------------------
    def _cmd(self):
        cmd = [sumolib.checkBinary("sumo"),
               "-n", self.net_file, "-r", self.rou_file,
               "--begin", "0", "--end", str(int(self.horizon)),
               "--step-length", str(self.interval),
               "--seed", str(int(self.seed)),
               "--time-to-teleport", "-1",
               "--waiting-time-memory", "10000",
               "--no-step-log", "true", "--no-warnings", "true",
               "--xml-validation", "never",
               "--collision.action", "warn"]
        if self.enable_ssm:
            from d2marl_sumo.ssm_metrics import sumo_cmd_ssm_flags
            cmd += sumo_cmd_ssm_flags(
                self.ssm_file or os.path.join(self.log_dir, "ssm.xml"))
        return cmd

    # ================= CityFlow-compatible interface =================
    def reset(self):
        if self._started:
            try:
                traci.switch(self.label)
                traci.close()
            except Exception:
                pass
        # Under high concurrency traci.start can fail: it probes a free port and then
        # lets SUMO bind it, and another process may take it in between (e.g. many
        # Ray workers per job). A failure in the middle of training loses all
        # previous work, so the start is retried with back-off,
        # as in reset() of d2marl_sumo/sumo_env.py.
        _last = None
        for _attempt in range(_START_RETRIES):
            try:
                traci.start(self._cmd(), label=self.label)
                break
            except Exception as e:
                _last = e
                try:
                    traci.close()
                except Exception:
                    pass
                _wait = _START_BACKOFF * (_attempt + 1)
                print("[cl-sumo] start attempt %d/%d failed (%s), retrying in %.1fs"
                      % (_attempt + 1, _START_RETRIES, e, _wait))
                time.sleep(_wait)
        else:
            raise RuntimeError("SUMO failed to start %d times in a row, last error: %s"
                               % (_START_RETRIES, _last))
        self._started = True
        self._t = 0.0
        self._depart, self._travel = {}, {}
        self._cache_step = -1
        self._cache = {}

    def set_random_seed(self, seed):
        """CityFlow sets the seed at run time; SUMO takes it at start-up, so it is
        stored and applied at the next reset()."""
        self.seed = int(seed)

    def next_step(self):
        traci.switch(self.label)
        traci.simulationStep()
        self._t = traci.simulation.getTime()
        for v in traci.simulation.getDepartedIDList():
            self._depart[v] = self._t
        for v in traci.simulation.getArrivedIDList():
            if v in self._depart:
                self._travel[v] = self._t - self._depart[v]
        self._cache = {}

    def get_current_time(self):
        return self._t

    # ---- fetched once per step, later calls use the cache (CityFlow also returns full dicts) ----
    def _snapshot(self):
        if self._cache:
            return self._cache
        traci.switch(self.label)
        lane_veh, lane_cnt, lane_halt = OrderedDict(), OrderedDict(), OrderedDict()
        speed, dist = {}, {}
        for cf_id in self._all_cf_lanes:
            s_id = self.cf2sumo[cf_id]
            try:
                vs = traci.lane.getLastStepVehicleIDs(s_id)
                lane_veh[cf_id] = list(vs)
                lane_cnt[cf_id] = len(vs)
                lane_halt[cf_id] = traci.lane.getLastStepHaltingNumber(s_id)
            except traci.TraCIException:
                lane_veh[cf_id], lane_cnt[cf_id], lane_halt[cf_id] = [], 0, 0
                continue
            for v in vs:
                try:
                    speed[v] = traci.vehicle.getSpeed(v)
                    # CityFlow's get_vehicle_distance = distance travelled on the current lane,
                    # same as SUMO's getLanePosition
                    dist[v] = traci.vehicle.getLanePosition(v)
                except traci.TraCIException:
                    speed[v], dist[v] = 0.0, 0.0
        self._cache = {"lane_veh": lane_veh, "lane_cnt": lane_cnt,
                       "lane_halt": lane_halt, "speed": speed, "dist": dist}
        return self._cache

    def get_lane_vehicles(self):
        return self._snapshot()["lane_veh"]

    def get_lane_vehicle_count(self):
        return self._snapshot()["lane_cnt"]

    def get_lane_waiting_vehicle_count(self):
        return self._snapshot()["lane_halt"]

    def get_vehicle_speed(self):
        return self._snapshot()["speed"]

    def get_vehicle_distance(self):
        return self._snapshot()["dist"]

    def get_vehicles(self, include_waiting=False):
        traci.switch(self.label)
        return list(traci.vehicle.getIDList())

    def get_vehicle_count(self):
        traci.switch(self.label)
        return traci.vehicle.getIDCount()

    def get_average_travel_time(self):
        """CityFlow counts unfinished trips with their elapsed time; the same here."""
        vals = list(self._travel.values())
        traci.switch(self.label)
        for v in traci.vehicle.getIDList():
            if v in self._depart:
                vals.append(self._t - self._depart[v])
        return (sum(vals) / len(vals)) if vals else 0.0

    def set_tl_phase(self, inter_id, phase_index):
        """
        As in CityFlow: phase_index=0 is yellow, 1..N the 1st..N-th green phase.
        (see MATSC_CityFlow.set_green_phase: set_tl_phase(id, phase_index+1))
        """
        st = self.tls_states.get(inter_id)
        if st is None:
            return
        traci.switch(self.label)
        if phase_index == 0:
            state = st["yellow"][0]
        else:
            g = st["green"]
            state = g[(int(phase_index) - 1) % len(g)]
        try:
            traci.trafficlight.setRedYellowGreenState(inter_id, state)
        except traci.TraCIException:
            pass

    def close(self):
        try:
            traci.switch(self.label)
            traci.close()
        except Exception:
            pass
        self._started = False
