from .config import DIC_AGENTS
from .env_factory import make_env   # CityFlow or SUMO, chosen from the config
import time
import os
import copy
import numpy as np
import csv
import logging
import threading

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, cnt_round, cnt_gen, dic_path, dic_agent_conf, dic_traffic_env_conf):

        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_path = dic_path
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.agents = [None]*dic_traffic_env_conf['NUM_AGENTS']
        self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                        "round_"+str(self.cnt_round), "generator_"+str(self.cnt_gen))
        # Creating the directory and creating the agents are separate steps: when a
        # run is resumed, the log directory of the interrupted round already exists,
        # and tying agent creation to directory creation would leave self.agents empty.
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)

        start_time = time.time()
        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
            agent = DIC_AGENTS[agent_name](
                dic_agent_conf=self.dic_agent_conf,
                dic_traffic_env_conf=self.dic_traffic_env_conf,
                dic_path=self.dic_path,
                cnt_round=self.cnt_round,
                intersection_id=str(i)
            )
            self.agents[i] = agent
        print("Create intersection agent time: ", time.time()-start_time)

        self.dic_traffic_env_conf["CURRENT_ROUND"] = self.cnt_round

        self.env = make_env(
            path_to_log=self.path_to_log,
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf
        )

    def generate(self):

        reset_env_start_time = time.time()
        done = False
        state = self.env.reset()
        step_num = 0
        reset_env_time = time.time() - reset_env_start_time
        running_start_time = time.time()
        
        # Initialize total reward and metrics tracking
        total_reward = 0
        reward_list = []

        # Track traffic metrics for dynamic reward shaping
        total_queue_length = 0
        total_pressure = 0
        
        while not done and step_num < int(self.dic_traffic_env_conf["RUN_COUNTS"] /
                                          self.dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []
            step_start_time = time.time()
            for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]):

                # Parameter-sharing agents: choose_action receives the whole state list
                # (as in LLMTSCS/utils/generator.py). CoLightAgent.convert_state_to_input
                # reads s[i] for intersection i, so a single dict would raise KeyError: 0.
                if self.dic_traffic_env_conf["MODEL_NAME"] in ["Attend", "EfficientColight", "MPLight", "AdvancedColight", "D2MARL", "CoLight"]:
                    one_state = state
                    action = self.agents[i].choose_action(step_num, one_state)
                    action_list = action
                else:
                    one_state = state[i]
                    action = self.agents[i].choose_action(step_num, one_state)
                    action_list.append(action)

            next_state, reward, done, _ = self.env.step(action_list)

            # Collect traffic metrics for dynamic reward shaping
            if hasattr(self.env, 'get_traffic_metrics'):
                metrics = self.env.get_traffic_metrics()
                total_queue_length += metrics.get("total_queue_length", 0)
                total_pressure += metrics.get("total_pressure", 0)

            # record the reward
            if isinstance(reward, list):
                episode_reward = np.mean(reward)  # mean reward over all agents
            else:
                episode_reward = reward
            total_reward += episode_reward
            reward_list.append(episode_reward)

            print("time: {0}, running_time: {1}".format(self.env.get_current_time() -
                                                        self.dic_traffic_env_conf["MIN_ACTION_TIME"],
                                                        time.time()-step_start_time))

            state = next_state
            step_num += 1
            
        # average reward
        avg_reward = total_reward / step_num if step_num > 0 else 0
        
        # write the reward log
        reward_log_path = os.path.join(self.path_to_log, "train_reward_log.txt")
        with open(reward_log_path, "w") as f:
            f.write(f"Training round: Round_{self.cnt_round}, Generator_{self.cnt_gen}\n")
            f.write(f"Episode average reward: {avg_reward}\n")
            f.write(f"Total steps: {step_num}\n")
            f.write(f"Total reward: {total_reward}\n")
            f.write("\nPer-step reward:\n")
            for i, r in enumerate(reward_list):
                f.write(f"Step {i}: {r}\n")
        
        print(f"Training episode finished, average reward: {avg_reward}, details in {reward_log_path}")
            
        running_time = time.time() - running_start_time
        log_start_time = time.time()
        print("start logging.......................")
        self.env.bulk_log_multi_process()
        log_time = time.time() - log_start_time
        self.env.end_cityflow()
        print("reset_env_time: ", reset_env_time)
        print("running_time: ", running_time)
        print("log_time: ", log_time)
        
        # append to the summary log
        summary_log_path = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round", 
                                      f"round_{self.cnt_round}", "reward_summary.txt")
        with open(summary_log_path, "a") as f:
            f.write(f"Generator_{self.cnt_gen}: average reward={avg_reward}, steps={step_num}\n")

        # end of episode: attention logs and episode counter
        for i in range(self.dic_traffic_env_conf["NUM_AGENTS"]):
            agent = self.agents[i]
            # attention weights are written in choose_action; here only the
            # episode counter is advanced (if the agent implements the hook).
            if hasattr(agent, "save_attention") and getattr(agent, "save_attention"):
                if hasattr(agent, "on_episode_end"):
                    agent.on_episode_end()

        return
