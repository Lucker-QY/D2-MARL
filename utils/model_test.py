from .config import DIC_AGENTS
from copy import deepcopy
from .env_factory import make_env   # CityFlow or SUMO, chosen from the config
import json
import os
import numpy as np



def test(model_dir, cnt_round, run_cnt, _dic_traffic_env_conf, save_replay=False,
         use_passed_env_conf=False):
    dic_traffic_env_conf = deepcopy(_dic_traffic_env_conf)
    records_dir = model_dir.replace("model", "records")
    model_round = "round_%d" % cnt_round
    dic_path = {"PATH_TO_MODEL": model_dir, "PATH_TO_WORK_DIRECTORY": records_dir}
    with open(os.path.join(records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    if not use_passed_env_conf and os.path.exists(os.path.join(records_dir, "anon_env.conf")):
        with open(os.path.join(records_dir, "anon_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)
    dic_traffic_env_conf["RUN_COUNTS"] = run_cnt

    if dic_traffic_env_conf["MODEL_NAME"] in dic_traffic_env_conf["LIST_MODEL_NEED_TO_UPDATE"]:
        dic_agent_conf["EPSILON"] = 0
        dic_agent_conf["MIN_EPSILON"] = 0

    agents = []
    for i in range(dic_traffic_env_conf['NUM_AGENTS']):
        agent_name = dic_traffic_env_conf["MODEL_NAME"]
        agent = DIC_AGENTS[agent_name](
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            cnt_round=cnt_round,
            intersection_id=str(i)
        )
        agents.append(agent)
    try:
        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            agents[i].load_network("{0}_inter_{1}".format(model_round, agents[i].intersection_id))
        path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
        if not os.path.exists(path_to_log):
            os.makedirs(path_to_log)
        dic_traffic_env_conf["CURRENT_ROUND"] = cnt_round
        env = make_env(
            path_to_log=path_to_log,
            path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=dic_traffic_env_conf,
            save_replay=save_replay,
            replay_dir=path_to_log
        )

        done = False

        step_num = 0
        # total reward and reward list
        total_reward = 0
        reward_list = []

        # accumulators for AWT and AQL
        queue_length_episode = []
        waiting_time_episode = []

        total_time = dic_traffic_env_conf["RUN_COUNTS"]
        state = env.reset()

        while not done and step_num < int(total_time / dic_traffic_env_conf["MIN_ACTION_TIME"]):
            action_list = []

            for i in range(dic_traffic_env_conf["NUM_AGENTS"]):

                # same list as in generator.py (parameter-sharing agents)
                if dic_traffic_env_conf["MODEL_NAME"] in ["CoLight", "EfficientColight", "Attend", "MPLight", "AdvancedColight", "D2MARL"]:
                    one_state = state
                    action_list = agents[i].choose_action(step_num, one_state)
                else:
                    one_state = state[i]
                    action = agents[i].choose_action(step_num, one_state)
                    action_list.append(action)

            next_state, reward, done, _ = env.step(action_list)

            # AQL: total number of queued vehicles over all intersections at this step
            queue_length_inter = []
            for inter in env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
            queue_length_episode.append(sum(queue_length_inter))

            # AWT: mean waiting time of the waiting vehicles at this step
            waiting_times = []
            for veh in env.waiting_vehicle_list:
                waiting_times.append(env.waiting_vehicle_list[veh]['time'])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)

            # accumulate the reward
            if isinstance(reward, list):
                episode_reward = np.mean(reward)  # mean reward over all agents
            else:
                episode_reward = reward
            total_reward += episode_reward
            reward_list.append(episode_reward)

            state = next_state
            step_num += 1

        # average reward
        avg_reward = total_reward / step_num if step_num > 0 else 0

        # AWT and AQL
        awt = round(np.mean(waiting_time_episode), 3) if len(waiting_time_episode) > 0 else 0
        aql = round(np.mean(queue_length_episode), 3) if len(queue_length_episode) > 0 else 0

        # ================= Gini coefficient and per-vehicle waiting times =================
        all_wait_times = list(env.vehicle_total_wait_time.values())
        if len(all_wait_times) > 0:
            # Gini coefficient
            wait_array = np.array(all_wait_times, dtype=np.float64)
            wait_array = np.sort(wait_array)
            n = len(wait_array)
            index = np.arange(1, n + 1)
            # Gini formula
            gini_coefficient = ((2 * index - n - 1) * wait_array).sum() / (n * wait_array.sum() + 1e-8)
            gini_coefficient = round(gini_coefficient, 4)

            # per-vehicle waiting times (for distribution plots)
            import pandas as pd
            df_wait = pd.DataFrame(list(env.vehicle_total_wait_time.items()), columns=['vehicle_id', 'total_wait_time'])
            # Extra column: the longest single continuous wait of each vehicle. Without
            # it, Mwt would have to be derived from delay_events_over_90s.csv, which
            # only holds waits above 90 s and is empty when a method never exceeds them
            # (Mwt could then only be reported as "<90"). This column has no threshold.
            # For older outputs without the column, collect_results falls back.
            _msw = env.get_max_single_wait() if hasattr(env, "get_max_single_wait") else {}
            # If the environment does not provide the column, do **not** write 0:
            # collect_results would take 0 as a real value.
            # If the column is missing, collect_results falls back and leaves it empty.
            if _msw:
                df_wait["max_single_wait"] = df_wait["vehicle_id"].map(
                    lambda v: round(float(_msw.get(v, 0.0)), 1))
            df_wait = df_wait.sort_values(by='total_wait_time', ascending=False)
            df_wait.to_csv(os.path.join(path_to_log, "all_vehicles_wait_time.csv"), index=False)
        else:
            gini_coefficient = 0.0
        # =================================================================

        # ==========================================================
        # export all waiting events above the threshold
        severe_threshold = 90
        delay_events = env.get_severe_delay_events(threshold=severe_threshold)
        severe_count = len(delay_events)

        if severe_count > 0:
            import pandas as pd
            df_events = pd.DataFrame(delay_events)
            df_events = df_events.sort_values(by="exceed_time", ascending=False)
            events_csv_path = os.path.join(path_to_log, f"delay_events_over_{severe_threshold}s.csv")
            df_events.to_csv(events_csv_path, index=False)
            print(f"!!! {severe_count} waiting events above {severe_threshold}s, details in {events_csv_path}")
        else:
            print(f"No waiting event above {severe_threshold}s in this test episode.")
        # ==========================================================

        # write the reward log
        reward_log_path = os.path.join(path_to_log, "reward_log.txt")
        with open(reward_log_path, "w") as f:
            f.write(f"Episode average reward: {avg_reward}\n")
            f.write(f"Total steps: {step_num}\n")
            f.write(f"Total reward: {total_reward}\n")
            f.write(f"AWT (average waiting time): {awt}\n")
            f.write(f"AQL (average queue length): {aql}\n")
            f.write(f"Gini coefficient: {gini_coefficient}\n")
            f.write(f"Severe delay events (>{severe_threshold}s): {severe_count}\n")
            f.write("\nPer-step reward:\n")
            for i, r in enumerate(reward_list):
                f.write(f"Step {i}: {r}\n")
        
        print(f"Test finished, average reward: {avg_reward}, details in {reward_log_path}")

        # ================= flush and save the attention logs of all agents =================
        for i in range(dic_traffic_env_conf['NUM_AGENTS']):
            if hasattr(agents[i], "on_episode_end"):
                agents[i].on_episode_end()
        # ====================================================================================

        env.batch_log_2()
        env.end_cityflow()

    except Exception as e:
        print("============== error occurs in model_test ============")
        print(f"Error: {e}")
        if use_passed_env_conf or _dic_traffic_env_conf.get("STRICT_TEST_ERRORS", False):
            raise  # a failed evaluation must fail the job instead of reporting success
