"""
Random agent.

The "Random" baseline: at every decision step a phase is drawn uniformly
from the phase set of the intersection.

Why per decision step rather than per second:
    choose_action is only called at action steps (MIN_ACTION_TIME = 15 s),
    so one uniform draw here is the standard definition of the Random
    baseline: an independent draw in every control interval.
    Redrawing every second would make the phase flicker at 1 Hz.

The random generator is separate from the environment seed (SEED + intersection index):
    * the three seeds give three different trajectories;
    * a run with the same seed is reproducible;
    * the environment's random stream (demand generation) is not consumed.
"""

import numpy as np

from .agent import Agent


class RandomAgent(Agent):

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path,
                 cnt_round, intersection_id):
        super(RandomAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)

        self.phase_length = len(self.dic_traffic_env_conf["PHASE"])
        self.action = 0

        try:
            inter_idx = int(str(intersection_id).split("_")[-1])
        except (ValueError, IndexError):
            inter_idx = abs(hash(str(intersection_id))) % 10007
        seed = int(self.dic_traffic_env_conf.get("SEED", 0)) + 7919 * inter_idx
        self.rng = np.random.RandomState(seed % (2 ** 31 - 1))

    def choose_action(self, count, state):
        # cur_phase == -1 means a yellow transition; the phase cannot change now,
        # same as in Fixedtime / MaxPressure
        if state["cur_phase"][0] == -1:
            return self.action

        if self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
            self.action = int(self.rng.randint(self.phase_length))
        else:
            # "switch" mode: 0 = keep, 1 = next phase, equal probability
            self.action = int(self.rng.randint(2))
        return self.action
