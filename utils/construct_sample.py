import numpy as np
import pickle
import os
import traceback


def get_reward_from_features(rs, threshold_unused=None, env_conf=None, current_round=0):
    queue_list = rs.get("lane_num_waiting_vehicle_in", [])
    queue_len = float(np.sum(queue_list)) if len(queue_list) > 0 else 0.0
    pressure = float(np.absolute(np.sum(rs.get("pressure", 0))))

    # raw 12-dimensional waiting times (dic_feature stores both versions)
    lane_waits = rs.get("lane_max_waiting_time", [])

    soft_limit = env_conf.get("WAIT_SOFT_LIMIT", 150.0) if env_conf else 150.0
    target_hard_limit = env_conf.get("WAIT_HARD_LIMIT", 300.0) if env_conf else 300.0
    penalty_cap = env_conf.get("MAX_PENALTY_CAP", 2000.0) if env_conf else 2000.0

    hard_limit = target_hard_limit

    # Huber parameters
    huber_delta = env_conf.get("HUBER_DELTA", 100.0) if env_conf else 100.0

    total_penalty = 0.0
    for wait_time in lane_waits:
        if wait_time > hard_limit:
            excess = wait_time - hard_limit

            # Bounded (Huber-like) delay penalty
            if excess <= huber_delta:
                # stage 1: quadratic region just above the hard limit
                total_penalty += 20.0 + (excess ** 2) / 50.0
            else:
                # stage 2: linear region for severe exceedance (bounded gradient)
                # value at the transition point delta
                base_penalty = 20.0 + (huber_delta ** 2) / 50.0
                # slope at delta (d(x^2/50)/dx = 2x/50 = x/25)
                smooth_slope = huber_delta / 25.0
                # linear growth
                total_penalty += base_penalty + (excess - huber_delta) * smooth_slope

        elif wait_time > soft_limit:
            excess = wait_time - soft_limit
            total_penalty += (excess / (hard_limit - soft_limit)) * 10.0

    final_penalty = min(total_penalty, penalty_cap)

    # third reward term: number of over-limit vehicles
    over_limit_list = rs.get("lane_raw_over_limit_num", [])
    total_over_limit = float(np.sum(over_limit_list)) if len(over_limit_list) > 0 else 0.0

    reward_dict = {
        "queue_length": queue_len,
        "pressure": pressure,
        "max_wait_penalty": final_penalty,
        "over_limit_penalty": total_over_limit  # weighted in cal_reward
    }
    return reward_dict


def cal_reward(rs, rewards_components, use_morl=False):
    """
    Return a vector reward when MORL is on, a scalar reward otherwise.
    """
    if use_morl:
        # MORL mode (not used in the paper): [r_efficiency, r_fairness]
        queue_len = rs.get("queue_length", 0)
        norm_queue = queue_len / 100.0
        r_eff = -norm_queue

        penalty = rs.get("max_wait_penalty", 0)
        r_fair = -penalty

        return [r_eff, r_fair]
    else:
        # scalar mode: weighted sum of the reward components
        r = 0
        for component, weight in rewards_components.items():
            if weight == 0:
                continue
            if component not in rs.keys() or rs[component] is None:
                continue
            r += rs[component] * weight
        return r


class ConstructSample:

    def __init__(self, path_to_samples, cnt_round, dic_traffic_env_conf):
        self.parent_dir = path_to_samples
        self.path_to_samples = path_to_samples + "/round_" + str(cnt_round)
        self.cnt_round = cnt_round
        self.dic_traffic_env_conf = dic_traffic_env_conf

        self.logging_data_list_per_gen = None
        self.hidden_states_list = None
        self.samples = []
        self.samples_all_intersection = [None]*self.dic_traffic_env_conf['NUM_INTERSECTIONS']

        self.interval = self.dic_traffic_env_conf["MIN_ACTION_TIME"]
        self.measure_time = self.dic_traffic_env_conf["MEASURE_TIME"]

        self.debug = getattr(self, 'debug', False)

        # ================= environment parameters =================
        self.use_morl = self.dic_traffic_env_conf.get("USE_MORL", False)
        self.threshold = self.dic_traffic_env_conf.get("DELAY_THRESHOLD", 90.0)
        # ==========================================================

    def load_data(self, folder, i):
        try:
            f_logging_data = open(os.path.join(self.path_to_samples, folder, "inter_{0}.pkl".format(i)), "rb")
            logging_data = pickle.load(f_logging_data)
            f_logging_data.close()
            return 1, logging_data

        except Exception:
            print("Error occurs when making samples for inter {0}".format(i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0, None

    def load_data_for_system(self, folder):
        self.logging_data_list_per_gen = []
        # print("Load data for system in ", folder)  # [DEBUG]
        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            pass_code, logging_data = self.load_data(folder, i)
            if pass_code == 0:
                return 0
            self.logging_data_list_per_gen.append(logging_data)
        return 1

    def construct_state(self, features, time, i):
        state = self.logging_data_list_per_gen[i][time]
        assert time == state["time"]
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            state_after_selection = {}
            for key, value in state["state"].items():
                if key in features:
                    if "cur_phase" in key:
                        state_after_selection[key] = self.dic_traffic_env_conf['PHASE'][value[0]]
                    else:
                        state_after_selection[key] = value
        else:
            state_after_selection = {key: value for key, value in state["state"].items() if key in features}
        return state_after_selection

    def construct_reward(self, rewards_components, time, i):
        """
        Instantaneous and average reward (MORL-compatible).
        """
        rs = self.logging_data_list_per_gen[i][time + self.measure_time - 1]
        assert time + self.measure_time - 1 == rs["time"]

        # reward components at the measurement step
        r_instant_dict = get_reward_from_features(rs['state'], self.threshold, self.dic_traffic_env_conf, self.cnt_round)

        list_r_dicts = []
        for t in range(time, time + self.measure_time):
            rs_t = self.logging_data_list_per_gen[i][t]
            assert t == rs_t["time"]
        # reward components at every step of the interval
            rs_dict = get_reward_from_features(rs_t['state'], self.threshold, self.dic_traffic_env_conf, self.cnt_round)
            list_r_dicts.append(rs_dict)

        # ================= pass the use_morl flag =================
        r_instant = cal_reward(r_instant_dict, rewards_components, self.use_morl)

        if list_r_dicts:
            r_average_dict = {
                key: np.mean([d[key] for d in list_r_dicts])
                for key in list_r_dicts[0].keys()
            }
        else:
            r_average_dict = r_instant_dict

        r_average = cal_reward(r_average_dict, rewards_components, self.use_morl)
        # ==========================================================

        return r_instant, r_average

    def judge_action(self, time, i):
        if self.logging_data_list_per_gen[i][time]['action'] == -1:
            raise ValueError
        else:
            return self.logging_data_list_per_gen[i][time]['action']

    def make_reward(self, folder, i):
        if self.samples_all_intersection[i] is None:
            self.samples_all_intersection[i] = []
        # if i % 100 == 0:
        #     print("make reward for inter {0} in folder {1}".format(i, folder))  # [DEBUG]
        list_samples = []
        try:
            total_time = int(self.logging_data_list_per_gen[i][-1]['time'] + 1)
            for time in range(0, total_time - self.measure_time + 1, self.interval):
                state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"], time, i)
                reward_instant, reward_average = self.construct_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"],
                                                                       time, i)
                action = self.judge_action(time, i)

                if time + self.interval == total_time:
                    next_state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                                                      time + self.interval - 1, i)

                else:
                    next_state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                                                      time + self.interval, i)
                sample = [state, action, next_state, reward_average, reward_instant, time,
                          folder+"-"+"round_{0}".format(self.cnt_round)]
                list_samples.append(sample)

            self.samples_all_intersection[i].extend(list_samples)
            return 1
        except:
            print("Error occurs when making rewards in generator {0} for intersection {1}".format(folder, i))
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            return 0

    def make_reward_for_system(self):
        for folder in os.listdir(self.path_to_samples):
            # print(folder)  # [DEBUG]
            if "generator" not in folder:
                continue
            if not self.load_data_for_system(folder):
                continue
            for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
                pass_code = self.make_reward(folder, i)
                if pass_code == 0:
                    continue

        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            self.dump_sample(self.samples_all_intersection[i], "inter_{0}".format(i))

    def dump_sample(self, samples, folder):
        if folder == "":
            with open(os.path.join(self.parent_dir, "total_samples.pkl"), "ab+") as f:
                pickle.dump(samples, f, -1)
        elif "inter" in folder:
            with open(os.path.join(self.parent_dir, "total_samples_{0}.pkl".format(folder)), "ab+") as f:
                pickle.dump(samples, f, -1)
        else:
            with open(os.path.join(self.path_to_samples, folder, "samples_{0}.pkl".format(folder)), 'wb') as f:
                pickle.dump(samples, f, -1)
