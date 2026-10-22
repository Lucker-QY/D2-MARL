"""
Actuated Signal Control (ASC) agent.

Actuated signal control (ASC) baseline.

Why not SUMO's native actuated program
----------------------------------------
SUMO supports tlLogic type="actuated", but this environment drives the phases
through TraCI at every step, which would override the native program. It would
also put ASC behind a different action interface than all other methods.

The textbook NEMA actuated logic is therefore implemented at the agent level,
with the same interface and the same metrics as Fixedtime / MaxPressure.

Control logic
--------
The three rules of actuated control:
  min green : a phase is held for at least g_min seconds
  gap out   : the phase ends early when it has no demand
  max out   : under continuous demand a phase is held for at most g_max seconds

A real ASC uses stop-line detector headways (gaps) to judge whether vehicles are
still arriving. No loop detectors are observed here, only the per-movement
queues (traffic_movement_pressure_queue), so the queue is the demand proxy: a
phase has demand when one of its movements has a queue > 0. This is slightly
conservative (gap out only once the queue has cleared), i.e. in favour of ASC.

Phase skipping is implemented: the controller cycles only through phases with
demand, and falls back to sequential cycling when no phase has demand.

Parameters
----
    MIN_GREEN / MAX_GREEN, passed via dic_agent_conf, default 10 / 60 s.
    Common NEMA values; the 30 s fixed-time plan used here lies in between.
"""

from .agent import Agent

import numpy as np


# phase -> indices of the movements it serves. Identical to the definition in
# MaxPressureAgent, so both rule-based baselines use the same phase coverage.
PHASE_TO_MOVEMENTS = {
    4: [(1, 4),    # WT_ET
        (7, 10),   # NT_ST
        (0, 3),    # WL_EL
        (6, 9)],   # NL_SL
    8: [(1, 4), (7, 10), (0, 3), (6, 9),
        (0, 1),    # WL_WT
        (3, 4),    # EL_ET
        (9, 10),   # SL_ST
        (6, 7)],   # NL_NT
}


class ActuatedAgent(Agent):

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path,
                 cnt_round, intersection_id):
        super(ActuatedAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)

        self.phase_length = len(self.dic_traffic_env_conf["PHASE"])
        self.min_green = int(self.dic_agent_conf.get("MIN_GREEN", 10))
        self.max_green = int(self.dic_agent_conf.get("MAX_GREEN", 60))
        self.movements = PHASE_TO_MOVEMENTS.get(
            self.phase_length, PHASE_TO_MOVEMENTS[4])

        self.action = 0
        self.current_phase_time = 0

    def _demand(self, queues, phase_idx):
        """Total queue on the movements served by the phase."""
        if phase_idx >= len(self.movements):
            return 0.0
        return float(sum(queues[m] for m in self.movements[phase_idx]
                         if m < len(queues)))

    def choose_action(self, count, state):
        if state["cur_phase"][0] == -1:
            return self.action

        queues = np.asarray(state["traffic_movement_pressure_queue"],
                            dtype=float).ravel()
        cur = int(state["cur_phase"][0])
        if cur < 0 or cur >= self.phase_length:
            cur = self.action
        elapsed = float(state["time_this_phase"][0])

        # --- min green: not reached yet, hold ---
        if elapsed < self.min_green:
            self.action = cur
            self.current_phase_time += 1
            return self._emit(cur, cur)

        keep = (self._demand(queues, cur) > 0) and (elapsed < self.max_green)
        if keep:
            # demand remains and max green not reached: extend
            self.action = cur
            self.current_phase_time += 1
            return self._emit(cur, cur)

        # --- gap out or max out: go to the next phase with demand ---
        # The range ends at phase_length + 1, so it wraps around to cur itself. This
        # is intended: at max out with no call from another phase, NEMA rests in
        # green instead of switching to an empty phase.
        nxt = cur
        for step in range(1, self.phase_length + 1):
            cand = (cur + step) % self.phase_length
            if self._demand(queues, cand) > 0:
                nxt = cand
                break
        else:
            # no queue anywhere: sequential cycling, i.e. fixed-time behaviour
            nxt = (cur + 1) % self.phase_length

        self.action = nxt
        self.current_phase_time = 0
        return self._emit(nxt, cur)

    def _emit(self, nxt, cur):
        """Output by ACTION_PATTERN: phase index for 'set', otherwise keep/switch."""
        if self.dic_traffic_env_conf.get("ACTION_PATTERN", "set") == "set":
            return nxt
        return 0 if nxt == cur else 1
