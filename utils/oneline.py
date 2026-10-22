from .config import DIC_AGENTS
from .env_factory import make_env   # CityFlow or SUMO, chosen from the config
from .pipeline import path_check, copy_cityflow_file, copy_conf_file
import os
import time
import numpy as np


class OneLine:

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.agents = []
        self.env = None
        self.initialize()

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
            agent = DIC_AGENTS[agent_name](
                dic_agent_conf=self.dic_agent_conf,
                dic_traffic_env_conf=self.dic_traffic_env_conf,
                dic_path=self.dic_path,
                cnt_round=0,
                intersection_id=str(i)
            )
            self.agents.append(agent)
        # Output goes to test_round/round_0/, the same layout as model_test, so that
        # d2marl_sumo.collect_results summarises all methods with the same code.
        self.path_to_log = os.path.join(
            self.dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", "round_0")
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)
        self.env = make_env(
            path_to_log=self.path_to_log,
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            # Non-learning methods (Fixedtime / MaxPressure / Random / Actuated) use
            # oneline and run a single episode without rounds, so
            # SAVE_REPLAY_ROUND >= 0 means: record this episode.
            save_replay=self.dic_traffic_env_conf.get(
                "SAVE_REPLAY_ROUND", -1) >= 0,
        )

    def train(self):
        print("================ start train ================")
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        # initialize output streams
        file_name_memory = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "memories.txt")
        done = False
        state = self.env.reset()
        step_num = 0
        print("end reset")
        current_time = self.env.get_current_time()  # in seconds

        start_time = time.time()
        # same sampling as utils/model_test.py: one sample per decision step
        queue_length_episode, waiting_time_episode = [], []
        while not done and current_time < total_run_cnt:
            action_list = []

            # Batching hook for agents that serve many intersections with one
            # inference call (currently LLMLight): the prompts of the whole network
            # are collected into one batch. Agents without the hook are unaffected.
            _cls = type(self.agents[0])
            if hasattr(_cls, "prepare_batch"):
                _cls.prepare_batch(step_num, state, self.dic_agent_conf)

            for i in range(len(state)):
                one_state = state[i]
                count = step_num
                action = self.agents[i].choose_action(count, one_state)
                action_list.append(action)
            next_state, reward, done, _ = self.env.step(action_list)

            # AQL: sum of the approach queues; AWT: mean current continuous wait of waiting vehicles
            queue_length_episode.append(sum(
                sum(inter.dic_feature['lane_num_waiting_vehicle_in'])
                for inter in self.env.list_intersection))
            waits = [v['time'] for v in self.env.waiting_vehicle_list.values()]
            waiting_time_episode.append(float(np.mean(waits)) if waits else 0.0)

            f_memory = open(file_name_memory, "a")
            # output to std out and file
            memory_str = 'time = {0}\taction = {1}\tcurrent_phase = {2}\treward = {3}'.\
                format(current_time, str(action_list), str([state[i]["cur_phase"][0] for i in range(len(state))]),
                       str(reward),)
            f_memory.write(memory_str + "\n")
            f_memory.close()
            current_time = self.env.get_current_time()  # in seconds

            state = next_state
            step_num += 1

        print("Training time: ", time.time()-start_time)

        # ===== metric summary, same format as utils/model_test.py =====
        awt = round(float(np.mean(waiting_time_episode)), 3) if waiting_time_episode else 0
        aql = round(float(np.mean(queue_length_episode)), 3) if queue_length_episode else 0

        severe_threshold = self.dic_traffic_env_conf.get("DELAY_THRESHOLD", 90)
        delay_events = self.env.get_severe_delay_events(threshold=severe_threshold)

        all_wait = list(getattr(self.env, "vehicle_total_wait_time", {}).values())
        gini = 0.0
        if all_wait:
            arr = np.sort(np.array(all_wait, dtype=np.float64))
            n = len(arr)
            idx = np.arange(1, n + 1)
            gini = round(float(((2 * idx - n - 1) * arr).sum()
                               / (n * arr.sum() + 1e-8)), 4)
            import pandas as pd
            # as in model_test: extra column with the longest single continuous wait,
            # so that Mwt does not depend on the >90 s event file.
            _df = pd.DataFrame(list(self.env.vehicle_total_wait_time.items()),
                               columns=["vehicle_id", "total_wait_time"])
            _msw = (self.env.get_max_single_wait()
                    if hasattr(self.env, "get_max_single_wait") else {})
            # do not write 0 if the environment lacks the column; collect_results falls back if it is missing
            if _msw:
                _df["max_single_wait"] = _df["vehicle_id"].map(
                    lambda v: round(float(_msw.get(v, 0.0)), 1))
            _df \
              .sort_values(by="total_wait_time", ascending=False) \
              .to_csv(os.path.join(self.path_to_log,
                                   "all_vehicles_wait_time.csv"), index=False)

        if delay_events:
            import pandas as pd
            pd.DataFrame(delay_events).sort_values(by="exceed_time", ascending=False) \
              .to_csv(os.path.join(
                  self.path_to_log,
                  "delay_events_over_{0}s.csv".format(int(severe_threshold))),
                  index=False)

        # final self-check of batching agents (LLMLight reports its <signal> parse rate)
        _cls = type(self.agents[0])
        if hasattr(_cls, "report"):
            _cls.report()

        with open(os.path.join(self.path_to_log, "reward_log.txt"), "w") as f:
            f.write("Episode average reward: {0}\n".format(0.0))
            f.write("Total steps: {0}\n".format(step_num))
            f.write("AWT (average waiting time): {0}\n".format(awt))
            f.write("AQL (average queue length): {0}\n".format(aql))
            f.write("Gini coefficient: {0}\n".format(gini))
            f.write("Severe delay events (>{0}s): {1}\n".format(
                int(severe_threshold), len(delay_events)))
        print("AWT={0}  AQL={1}  Gini={2}  Cd={3}".format(
            awt, aql, gini, len(delay_events)))

        self.env.batch_log_2()

        # Close the simulator properly. Otherwise the Python process exits, the socket
        # closes, SUMO reports "peer shutdown" and exits with an error, and files that
        # SUMO writes itself (collisions.xml, ssm.xml in evaluation) may be incomplete.
        try:
            self.env.end_cityflow()
        except Exception as e:
            print("Exception while closing the simulator (metrics already written):", e)
