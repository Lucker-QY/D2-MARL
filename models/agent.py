from copy import deepcopy


class Agent(object):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id="0"):
        # Agents update run-time state such as EPSILON; keep it out of the base
        # config shared by Pipeline/Updater, otherwise serial training decays
        # an already-decayed EPSILON again every round.
        self.dic_agent_conf = deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.intersection_id = intersection_id

    def choose_action(self):

        raise NotImplementedError
