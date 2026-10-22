from .config import DIC_AGENTS
import pickle
import os
import time
import traceback
import random


class Updater:

    def __init__(self, cnt_round, dic_agent_conf, dic_traffic_env_conf, dic_path):

        self.cnt_round = cnt_round
        self.dic_path = dic_path
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        self.agents = []
        self.sample_set_list = []
        self.sample_indexes = None

        # Debug flag for conditional logging
        self.debug = getattr(self, 'debug', False)

        if not self.debug:
            print("Number of agents: ", dic_traffic_env_conf['NUM_AGENTS'])

        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
            agent= DIC_AGENTS[agent_name](
                self.dic_agent_conf, self.dic_traffic_env_conf,
                self.dic_path, self.cnt_round, intersection_id=str(i))
            self.agents.append(agent)

    def load_memory_after_forget(self, i):
        """Load the samples from disk, truncate them and apply the forget window."""
        memory_after_forget = []
        try:
            sample_file = open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                            "total_samples_inter_{0}".format(i) + ".pkl"), "rb")
            cur_sample_set = []
            while True:
                try:
                    cur_sample_set += pickle.load(sample_file)
                except EOFError:
                    break
            sample_file.close()

            ind_end = len(cur_sample_set)
            ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
            memory_after_forget = cur_sample_set[ind_sta: ind_end]

            if self.cnt_round % self.dic_traffic_env_conf["FORGET_ROUND"] == 0:
                with open(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                       "total_samples_inter_{0}".format(i) + ".pkl"), "wb+") as f:
                    pickle.dump(memory_after_forget, f, -1)
        except Exception as e:
            # keep the original error handling
            error_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"]).replace("records", "errors")
            if not os.path.exists(error_dir):
                os.makedirs(error_dir)
            f = open(os.path.join(error_dir, "error_info_inter_{0}.txt".format(i)), "a")
            f.write("Fail to load samples for inter {0}\n".format(i))
            f.write('traceback.format_exc():\n%s\n' % traceback.format_exc())
            f.close()
            print('traceback.format_exc():\n%s' % traceback.format_exc())

        return memory_after_forget

    def load_sample_for_agents(self):
        """Load the memory of all intersections and draw a time-aligned stratified replay batch."""
        start_time = time.time()
        num_intersections = self.dic_traffic_env_conf['NUM_INTERSECTIONS']

        # 1. load the memory of all intersections
        all_memories = []
        for i in range(num_intersections):
            mem = self.load_memory_after_forget(i)
            all_memories.append(mem)

        memory_len = len(all_memories[0]) if len(all_memories) > 0 else 0
        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], memory_len)

        # 2. time-aligned sample indexes (stratified sampling)
        use_stratified = self.dic_agent_conf.get("USE_STRATIFIED_REPLAY", False)

        if use_stratified and memory_len > 0:
            penalty_threshold = self.dic_agent_conf.get("PENALTY_REPLAY_THRESHOLD", -50.0)
            penalty_ratio = self.dic_agent_conf.get("PENALTY_REPLAY_RATIO", 0.20)

            normal_indices = []
            penalty_indices = []

            # a time step is a severe sample if any intersection has a severe penalty at that step
            for k in range(memory_len):
                is_penalty = False
                for i in range(num_intersections):
                    # sample[3] is reward_average
                    if k < len(all_memories[i]) and all_memories[i][k][3] < penalty_threshold:
                        is_penalty = True
                        break
                if is_penalty:
                    penalty_indices.append(k)
                else:
                    normal_indices.append(k)

            max_penalty_samples = int(sample_size * penalty_ratio)

            if len(penalty_indices) > 0:
                actual_penalty_count = min(max_penalty_samples, len(penalty_indices))
                actual_normal_count = min(sample_size - actual_penalty_count, len(normal_indices))

                # if there are too few regular samples, fill up with severe ones (rare)
                if actual_normal_count + actual_penalty_count < sample_size:
                    actual_penalty_count = min(len(penalty_indices), sample_size - actual_normal_count)

                sampled_penalty = random.sample(penalty_indices, actual_penalty_count)
                sampled_normal = random.sample(normal_indices, actual_normal_count)

                self.sample_indexes = sampled_normal + sampled_penalty
                random.shuffle(self.sample_indexes)  # shuffle for batch diversity

                # print the replay composition every 10 rounds
                if self.cnt_round % 10 == 0:
                    print(f"--> [Round {self.cnt_round}] stratified replay: {actual_penalty_count} severe samples, {actual_normal_count} regular samples")
            else:
                # no severe samples: plain random sampling
                self.sample_indexes = random.sample(range(memory_len), sample_size)
        else:
            if memory_len > 0:
                self.sample_indexes = random.sample(range(memory_len), sample_size)
            else:
                self.sample_indexes = []

        # 3. extract the samples at the aligned indexes
        if self.dic_traffic_env_conf['MODEL_NAME'] in ["Attend", "MPLight"]:
            sample_set_all = []
            for i in range(num_intersections):
                if len(all_memories[i]) > 0:
                    sample_set = [all_memories[i][k] for k in self.sample_indexes if k < len(all_memories[i])]
                    sample_set_all.extend(sample_set)
            if sample_set_all:
                self.agents[0].prepare_Xs_Y(sample_set_all)

        elif self.dic_traffic_env_conf['MODEL_NAME'] in ["PressLight"]:
            for i in range(num_intersections):
                if len(all_memories[i]) > 0:
                    sample_set = [all_memories[i][k] for k in self.sample_indexes if k < len(all_memories[i])]
                    self.agents[i].prepare_Xs_Y(sample_set)

        # The original "CoLight" goes here: it shares CoLightAgent with EfficientColight
        # (utils/config.py:DIC_AGENTS), and prepare_Xs_Y expects one sample list per
        # intersection, not a flattened list.
        elif self.dic_traffic_env_conf['MODEL_NAME'] in ["CoLight", "EfficientColight", "AdvancedColight", "D2MARL"]:
            samples_list = []
            for i in range(num_intersections):
                if len(all_memories[i]) > 0:
                    sample_set = [all_memories[i][k] for k in self.sample_indexes if k < len(all_memories[i])]
                    samples_list.append(sample_set)
            if samples_list and len(samples_list[0]) > 0:
                self.agents[0].prepare_Xs_Y(samples_list)

        else:
            # No branch claimed this model name, so prepare_Xs_Y would never be called
            # and train_network would keep fitting stale targets. Report it.
            print("!! [updater] MODEL_NAME=%r is not handled by any prepare_Xs_Y branch; "
                  "no training targets were built this round. Add it in utils/updater.py."
                  % self.dic_traffic_env_conf['MODEL_NAME'])

    def update_network(self, i):
        # print('update agent %d' % i)  # [DEBUG]
        self.agents[i].train_network()
        self.agents[i].save_network("round_{0}_inter_{1}".format(self.cnt_round, self.agents[i].intersection_id))

    def update_network_for_agents(self):
        # print("update_network_for_agents", self.dic_traffic_env_conf['NUM_AGENTS'])  # [DEBUG]
        for i in range(self.dic_traffic_env_conf['NUM_AGENTS']):
            self.update_network(i)
