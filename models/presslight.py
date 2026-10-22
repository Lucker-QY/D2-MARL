"""
PressLight agent, based on LIT model structure.
Observations: [cur_phase, lane_num_vechile_in, lane_num_vehicle_out]
Reward: -Pressure
"""

from .network_agent import NetworkAgent, Selector
from tensorflow.keras.layers import Dense, concatenate, Add, Multiply
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import os
import csv
import time
import pandas as pd


class PressLightAgent(NetworkAgent):
    def __init__(self, dic_agent_conf=None, dic_traffic_env_conf=None, dic_path=None, cnt_round=None,
                 intersection_id="0"):
        # options for saving reward metrics
        self.save_metrics = dic_agent_conf.get('SAVE_METRICS', False)
        self.metrics_save_freq = dic_agent_conf.get('METRICS_SAVE_FREQ', 100)
        
        # initialise metric logging if enabled
        if self.save_metrics:
            self.metrics_history = []
            self.episode_count = cnt_round if cnt_round is not None else 0
            print(f"Metric logging enabled, every {self.metrics_save_freq} steps")
        
        # Parent initialisation.
        # Keyword arguments are required; the NetworkAgent signature is
        #   (dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id)
        # Passing only four positional arguments puts intersection_id into cnt_round,
        # which fails silently in two ways:
        #   1) cnt_round becomes the string "0", `if cnt_round == 0` is never true,
        #      and every round tries to load the weights of round '0'-1, which fails
        #      and leaves freshly initialised weights, i.e. no learning across rounds.
        #   2) intersection_id falls back to "0", so all intersections share one
        #      model file and overwrite each other's weights.
        super(PressLightAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path,
            cnt_round=cnt_round, intersection_id=intersection_id)
        
        # make sure q_network is initialised
        if not hasattr(self, 'q_network') or self.q_network is None:
            try:
                self.q_network = self.build_network()
                self.q_network_bar = self.build_network()
                self.q_network_bar.set_weights(self.q_network.get_weights())
            except Exception as e:
                print(f"Error while building the network: {e}")
        
        # directory for metrics
        self.metrics_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "metrics")
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)
        
        # current step
        self.current_step = 0
        
        # debug output
        print(f"PressLightAgent initialised, q_network: {self.q_network is not None}")

    def build_network(self):
        # The input ports are generated from LIST_STATE_FEATURE, as in
        # LLMTSCS/models/presslight_one.py. PressLight uses two features
        # (cur_phase + traffic_movement_pressure_queue), as in LLMTSCS/run_presslight.py,
        # so a fixed number of ports would not match convert_state_to_input.
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        dic_input_node = {}
        for feat_name in used_feature:
            _shape = (8,) if "cur_phase" in feat_name else (12,)
            dic_input_node[feat_name] = Input(shape=_shape, name="input_" + feat_name)

        # concatenate features
        list_all_flatten_feature = []
        for feature_name in used_feature:
            list_all_flatten_feature.append(dic_input_node[feature_name])
        all_flatten_feature = concatenate(list_all_flatten_feature, axis=1, name="all_flatten_feature")

        # shared dense layer
        shared_dense = Dense(self.dic_agent_conf["D_DENSE"], activation="sigmoid",
                             name="shared_hidden")(all_flatten_feature)

        # build phase selector layer
        list_selected_q_values = []
        for phase_id in range(1, self.num_phases + 1):
            if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
                phase_expansion = self.dic_traffic_env_conf["PHASE"][phase_id]
            else:
                phase_expansion = phase_id
            locals()["q_values_{0}".format(phase_id)] = self._separate_network_structure(
                shared_dense, self.dic_agent_conf["D_DENSE"], self.num_actions, memo=phase_id)
            locals()["selector_{0}".format(phase_id)] = Selector(
                phase_expansion, d_phase_encoding=8, d_action=self.num_actions,
                name="selector_{0}".format(phase_id))(dic_input_node["cur_phase"])
            locals()["q_values_{0}_selected".format(phase_id)] = Multiply(name="multiply_{0}".format(phase_id))(
                [locals()["q_values_{0}".format(phase_id)],
                 locals()["selector_{0}".format(phase_id)]]
            )
            list_selected_q_values.append(locals()["q_values_{0}_selected".format(phase_id)])
        q_values = Add()(list_selected_q_values)

        network = Model(inputs=[dic_input_node[feature_name]
                                for feature_name in used_feature],
                        outputs=q_values)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"]),
                        loss="mean_squared_error")
        network.summary()

        return network

    @staticmethod
    def _separate_network_structure(state_features, dense_d, num_actions, memo=""):
        hidden_1 = Dense(dense_d, activation="relu", name="hidden_separate_branch_{0}_1".format(memo))(
            state_features)
        q_values = Dense(num_actions, activation="linear", name="q_values_separate_branch_{0}".format(memo))(hidden_1)
        return q_values
        
    def choose_action(self, count, states):
        """
        Choose an action for the current state with an epsilon-greedy policy.
        
        Args:
            count: current step
            states: list of state dicts, one per intersection
            
        Returns:
            numpy.array: the chosen action of every agent
        """
        # current step
        self.current_step = count
        
        # actions from the parent choose_action
        action = super(PressLightAgent, self).choose_action(count, states)
        
        # collect traffic indicators if metric logging is enabled
        if hasattr(self, 'save_metrics') and self.save_metrics:
            self._collect_traffic_metrics(states, action)
            
            # save the indicators every 100 steps
            if count % self.metrics_save_freq == 0 and count > 0:
                self.save_metrics_to_csv()
                
        return action
    
    def _collect_traffic_metrics(self, states, actions):
        """Collect traffic indicators."""
        metrics = {}
        
        # indicators of every intersection
        for i, state in enumerate(states):
            if i >= len(states):
                continue
                
            # queue length of the intersection
            queue_length_in = sum(state.get('lane_num_vehicle_in', [0]))
            queue_length_out = sum(state.get('lane_num_vehicle_out', [0]))
            
            # pressure of the intersection
            pressure = 0
            if 'pressure' in state:
                pressure = sum(state['pressure']) if isinstance(state['pressure'], list) else state['pressure']
            
            # current phase
            current_phase = state.get('cur_phase', [0])[0] if 'cur_phase' in state else 0
            
            # add to the indicator dict
            metrics[f'intersection_{i}'] = {
                'queue_length_in': queue_length_in,
                'queue_length_out': queue_length_out,
                'total_queue_length': queue_length_in + queue_length_out,
                'pressure': pressure,
                'phase': current_phase,
                'action': int(actions[i]) if i < len(actions) else -1
            }
        
        # build the record
        record = {
            'step': self.current_step,
            'timestamp': time.time(),
            'metrics': metrics
        }
        
        # append to the history
        if not hasattr(self, 'metrics_history'):
            self.metrics_history = []
        self.metrics_history.append(record)
    
    def save_metrics_to_csv(self):
        """Write the collected indicators to CSV."""
        if not hasattr(self, 'metrics_history') or not self.metrics_history:
            return
        
        # file name
        file_name = f"metrics_ep{self.episode_count}_inter{self.intersection_id}.csv"
        file_path = os.path.join(self.metrics_dir, file_name)
        
        try:
            # prepare the rows
            data = []
            for record in self.metrics_history:
                step = record['step']
                timestamp = record['timestamp']
                
                for inter_id, metrics in record['metrics'].items():
                    row = {
                        'step': step,
                        'timestamp': timestamp,
                        'intersection_id': inter_id,
                        'queue_length_in': metrics['queue_length_in'],
                        'queue_length_out': metrics['queue_length_out'],
                        'total_queue_length': metrics['total_queue_length'],
                        'pressure': metrics['pressure'],
                        'phase': metrics['phase'],
                        'action': metrics['action']
                    }
                    data.append(row)
            
            # save as CSV with pandas
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
            print(f"Saved indicator data to {file_path}")
            
            # clear the history to save memory
            self.metrics_history = []
            
        except Exception as e:
            print(f"Error while saving indicator data: {e}")
    
    def remember(self, state, action, next_state, reward, is_terminal=0, info=None, path_index=0):
        """Store a transition and record its reward."""
        # parent remember()
        super().remember(state, action, next_state, reward, is_terminal, info, path_index)
        
        # record the reward if metric logging is enabled
        if hasattr(self, 'save_metrics') and self.save_metrics:
            if not hasattr(self, 'metrics_history'):
                self.metrics_history = []
            
            # reward record
            record = {
                'step': self.current_step,
                'timestamp': time.time(),
                'metrics': {
                    f'intersection_{self.intersection_id}': {
                        'reward': reward,
                        'action': action,
                        'is_terminal': is_terminal
                    }
                }
            }
            
            self.metrics_history.append(record)

    def prepare_Xs_Y(self, memory):
        """
        designed for update simple dqn models
        """
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting

        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        memory_after_forget = memory[ind_sta: ind_end]
        print("memory size after forget:", len(memory_after_forget))

        # sample the memory
        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
        sample_slice = random.sample(memory_after_forget, sample_size)
        print("memory samples number:", sample_size)

        # used_feature = ["phase_2", "phase_num_vehicle"]
        # Likewise the number of ports equals the number of features; a fixed 3 would
        # leave an empty third slot for a two-input network.
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        _state = [[] for _ in used_feature]
        _next_state = [[] for _ in used_feature]
        _action = []
        _reward = []
        for i in range(len(sample_slice)):
            state, action, next_state, reward, _, _, _ = sample_slice[i]
            for feat_idx, feat_name in enumerate(used_feature):
                _state[feat_idx].append(state[feat_name])
                _next_state[feat_idx].append(next_state[feat_name])
            _action.append(action)
            _reward.append(reward)
        # well prepared states
        _state2 = [np.array(ss) for ss in _state]
        _next_state2 = [np.array(ss) for ss in _next_state]

        cur_qvalues = self.q_network.predict(_state2)
        next_qvalues = self.q_network_bar.predict(_next_state2)
        # [batch, 4]
        target = np.copy(cur_qvalues)

        for i in range(len(sample_slice)):
            target[i, _action[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                    np.max(next_qvalues[i, :])
        self.Xs = _state2
        self.Y = target
        
    def save_network(self, file_name):
        """Save the network together with the indicator data."""
        # parent save_network()
        super().save_network(file_name)
        
        # save the indicator data
        if hasattr(self, 'save_metrics') and self.save_metrics and hasattr(self, 'metrics_history') and self.metrics_history:
            self.save_metrics_to_csv()