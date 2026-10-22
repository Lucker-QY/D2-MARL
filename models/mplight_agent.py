"""
MPLight agent, based on FRAP model structure.
Observations: [cur_phase, traffic_movement_pressure_num]
Reward: -Pressure
"""

from tensorflow.keras.layers import Input, Dense, Reshape,  Lambda,  Activation, Embedding, Conv2D, concatenate, add,\
    multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .network_agent import NetworkAgent, slice_tensor, relation
from .phase_util import phase_code
from tensorflow.keras import backend as K
import numpy as np
import random
import os
import csv
import time
import pandas as pd


class MPLightAgent(NetworkAgent):
    """
    optimize the build_network function
    assert the features are [ cur_phase, other feature]
    in this part, the determined feature name are removed
    """
    def __init__(self, dic_agent_conf=None, dic_traffic_env_conf=None, dic_path=None, cnt_round=None,
                 intersection_id="0"):
        # options for saving attention weights
        self.save_attention = dic_agent_conf.get('SAVE_ATTENTION', False)
        self.attention_save_freq = dic_agent_conf.get('ATTENTION_SAVE_FREQ', 100)
        
        # initialise attention-weight logging if enabled
        if self.save_attention:
            self.attention_weights_history = []
            self.episode_count = cnt_round if cnt_round is not None else 0
            print(f"Attention weight logging enabled, every {self.attention_save_freq} steps")
            
        super(MPLightAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id)
        
        # make sure q_network is initialised
        if not hasattr(self, 'q_network') or self.q_network is None:
            try:
                self.q_network = self.build_network()
                self.q_network_bar = self.build_network()
                self.q_network_bar.set_weights(self.q_network.get_weights())
            except Exception as e:
                print(f"Error while building the network: {e}")
        
        # directory for reward metrics
        self.metrics_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "metrics")
        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)
            
        # reward history
        self.reward_history = []
        self.current_step = 0
        
        # debug output
        print(f"MPLightAgent initialised, q_network: {self.q_network is not None}")

    def build_network(self):
        dic_input_node = {"feat1": Input(shape=(8,), name="input_cur_phase"),
                          "feat2": Input(shape=(12,), name="input_feat2")}

        p = Activation('sigmoid')(Embedding(2, 4, input_length=8)(dic_input_node["feat1"]))
        d = Dense(4, activation="sigmoid", name="num_vec_mapping")
        dic_lane = {}
        dic_index = {
            "WL": 0,
            "WT": 1,
            "EL": 3,
            "ET": 4,
            "NL": 6,
            "NT": 7,
            "SL": 9,
            "ST": 10,
        }
        for i, m in enumerate(self.dic_traffic_env_conf["list_lane_order"]):
            idx = dic_index[m]
            tmp_vec = d(
                Lambda(slice_tensor, arguments={"index": idx}, name="vec_%d" % i)(dic_input_node["feat2"]))
            tmp_phase = Lambda(slice_tensor, arguments={"index": i}, name="phase_%d" % i)(p)
            dic_lane[m] = concatenate([tmp_vec, tmp_phase], name="lane_%d" % i)

        if self.num_actions == 8 or self.num_actions == 4:
            list_phase_pressure = []
            lane_embedding = Dense(16, activation="relu", name="lane_embedding")
            for phase in self.dic_traffic_env_conf["PHASE_LIST"]:
                m1, m2 = phase.split("_")
                list_phase_pressure.append(add([lane_embedding(dic_lane[m1]),
                                                lane_embedding(dic_lane[m2])], name=phase))
            # [batch, 4, 3], initialed zeros
            constant_o = Lambda(relation, arguments={"phase_list": self.dic_traffic_env_conf["PHASE_LIST"]},
                                name="constant_o")(dic_input_node["feat2"])
        relation_embedding = Embedding(2, 4, name="relation_embedding")(constant_o)

        # rotate the phase pressure
        list_phase_pressure_recomb = []
        num_phase = len(list_phase_pressure)

        for i in range(num_phase):
            for j in range(num_phase):
                if i != j:
                    list_phase_pressure_recomb.append(
                        concatenate([list_phase_pressure[i], list_phase_pressure[j]],
                                    name="concat_compete_phase_%d_%d" % (i, j)))

        list_phase_pressure_recomb = concatenate(list_phase_pressure_recomb, name="concat_all")
        if num_phase == 8:
            feature_map = Reshape((8, 7, 32))(list_phase_pressure_recomb)
        else:
            feature_map = Reshape((4, 3, 32))(list_phase_pressure_recomb)
        lane_conv = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1),
                           activation="relu", name="lane_conv")(feature_map)
        relation_conv = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu",
                               name="relation_conv")(relation_embedding)
        combine_feature = multiply([lane_conv, relation_conv], name="combine_feature")

        # [batch, 4, 3, D_DENSE]
        hidden_layer = Conv2D(self.dic_agent_conf["D_DENSE"], kernel_size=(1, 1), activation="relu",
                              name="combine_conv")(combine_feature)
        # [batch, 4, 3, 1 ]
        before_merge = Conv2D(1, kernel_size=(1, 1), activation="linear", name="before_merge")(hidden_layer)
        if self.num_actions == 8:
            q_values = Lambda(lambda x: K.sum(x, axis=2), name="q_values")(Reshape((8, 7))(before_merge))
        else:
            q_values = Lambda(lambda x: K.sum(x, axis=2), name="q_values")(Reshape((4, 3))(before_merge))

        network = Model(inputs=[dic_input_node[feature_name] for feature_name in ["feat1", "feat2"]],
                        outputs=q_values)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"], epsilon=1e-08),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        network.summary()

        return network

    def convert_state_to_input(self, s):
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            inputs = []
            for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature == "cur_phase":
                    # cur_phase is -1 during yellow; see models/phase_util.py
                    inputs.append(np.array([phase_code(self, s[feature][0])]))
                else:
                    inputs.append(np.array([s[feature]]))
            return inputs
        else:
            return [np.array([s[feature]]) for feature in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]]

    def choose_action(self, count, states):
        """
        Choose the best action for the current state.
        """
        # current step
        self.current_step = count
        
        # make sure q_network is initialised
        if not hasattr(self, 'q_network') or self.q_network is None:
            try:
                print("q_network not initialised, building the network...")
                self.q_network = self.build_network()
                self.q_network_bar = self.build_network()
                self.q_network_bar.set_weights(self.q_network.get_weights())
            except Exception as e:
                print(f"Failed to build the network: {e}")
                # fall back to a random action
                return np.random.randint(self.num_actions, size=len(states))
        
        dic_state_feature_arrays = {}  # {feature1: [inter1, inter2,..], feature2: [inter1, inter 2...]}
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:2]
        for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            dic_state_feature_arrays[feature_name] = []
        for s in states:
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature_name == "cur_phase":
                    dic_state_feature_arrays[feature_name].append(
                        phase_code(self, s[feature_name][0]))
                else:
                    dic_state_feature_arrays[feature_name].append(s[feature_name])
        state_input = [np.array(dic_state_feature_arrays[feature_name]) for feature_name in used_feature]

        try:
            q_values = self.q_network.predict(state_input)
            # e-greedy
            if random.random() <= self.dic_agent_conf["EPSILON"]:  # continue explore new Random Action
                action = np.random.randint(len(q_values[0]), size=len(q_values))
            else:
                action = np.argmax(q_values, axis=1)
        except Exception as e:
            print(f"Q-value prediction failed: {e}")
            # fall back to a random action
            action = np.random.randint(self.num_actions, size=len(states))
        
        # store traffic indicators
        if hasattr(self, 'save_metrics') and self.save_metrics:
            self._collect_traffic_metrics(states, q_values, action)

        return action
        
    def _collect_traffic_metrics(self, states, q_values, actions):
        """Collect traffic indicators and rewards."""
        metrics = {}
        
        # indicators of every intersection
        for i, state in enumerate(states):
            if i >= self.num_agents:
                continue
                
            metrics[f'intersection_{i}'] = {
                'queue_length': sum(state.get('lane_num_waiting_vehicle_in', [0])),
                'pressure': np.sum(state.get('pressure', [0])) if 'pressure' in state else 0,
                'phase': state.get('cur_phase', [0])[0] if 'cur_phase' in state else 0,
                'max_q_value': float(np.max(q_values[i])) if i < len(q_values) else 0,
                'action': int(actions[i]) if i < len(actions) else 0
            }
            
        # append to the history
        record = {
            'step': self.current_step,
            'timestamp': time.time(),
            'metrics': metrics
        }
        self.reward_history.append(record)
        
        # save every 100 steps
        if self.current_step % 100 == 0 and self.current_step > 0:
            self.save_metrics_to_csv()
            
    def save_metrics_to_csv(self):
        """Write the collected indicators to CSV."""
        if not hasattr(self, 'reward_history') or not self.reward_history:
            return
            
        # file name
        episode_count = getattr(self, 'episode_count', 0)
        file_name = f"metrics_ep{episode_count}_inter{self.intersection_id}.csv"
        file_path = os.path.join(self.metrics_dir, file_name)
        
        try:
            # flatten the records
            flat_data = []
            for record in self.reward_history:
                step = record['step']
                timestamp = record['timestamp']
                
                for inter_id, metrics in record['metrics'].items():
                    row = {
                        'step': step,
                        'timestamp': timestamp,
                        'intersection_id': inter_id,
                        'queue_length': metrics['queue_length'],
                        'pressure': metrics['pressure'],
                        'phase': metrics['phase'],
                        'max_q_value': metrics['max_q_value'],
                        'action': metrics['action']
                    }
                    flat_data.append(row)
                    
            # save as CSV with pandas
            df = pd.DataFrame(flat_data)
            df.to_csv(file_path, index=False)
            
            print(f"Saved indicator data to {file_path}")
            
            # clear the history to save memory
            self.reward_history = []
            
        except Exception as e:
            print(f"Error while saving indicator data: {e}")

    def remember(self, state, action, next_state, reward, is_terminal=0, info=None, path_index=0):
        """Store a transition and record its reward."""
        # raw reward
        if hasattr(self, 'save_metrics') and self.save_metrics:
            metrics = {
                'step': self.current_step,
                'reward': reward,
                'action': action,
                'is_terminal': is_terminal,
                'intersection_id': self.intersection_id
            }
            
            if not hasattr(self, 'reward_history'):
                self.reward_history = []
                
            self.reward_history.append({
                'step': self.current_step,
                'timestamp': time.time(),
                'metrics': {
                    f'intersection_{self.intersection_id}': {
                        'reward': reward,
                        'action': action,
                        'is_terminal': is_terminal
                    }
                }
            })
        
        # call the parent remember()
        super().remember(state, action, next_state, reward, is_terminal, info, path_index)

    def prepare_Xs_Y(self, memory):
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
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:2]
        _state = [[] for _ in range(len(used_feature))]  # one list per feature
        _next_state = [[] for _ in range(len(used_feature))]  # one list per feature
        _action = []
        _reward = []
        
        for i in range(len(sample_slice)):
            try:
                sample = sample_slice[i]
                # at least state, action, next_state and reward
                if len(sample) >= 4:
                    state = sample[0]
                    action = sample[1]
                    next_state = sample[2]
                    reward = sample[3]
                    
                    # state and next_state must be dicts
                    if isinstance(state, dict) and isinstance(next_state, dict):
                        # regular dict states
                        for feat_idx, feat_name in enumerate(used_feature):
                            if feat_name in state and feat_name in next_state:
                                _state[feat_idx].append(state[feat_name])
                                _next_state[feat_idx].append(next_state[feat_name])
                            else:
                                raise ValueError(f"feature {feat_name} is missing from the state")
                        _action.append(action)
                        _reward.append(reward)
                    else:
                        print(f"Warning: state of sample {i} is not a dict but {type(state)}, {type(next_state)}")
                else:
                    print(f"Warning: sample {i} is malformed, length {len(sample)}")
            except Exception as e:
                print(f"Error while processing sample {i}: {e}")
        
        # return early when there is no valid sample
        if not _action or not _reward:
            print("Warning: no valid sample, skipping training")
            self.Xs = None
            self.Y = None
            return
            
        # well prepared states
        _state2 = [np.array(ss) for ss in _state]
        _next_state2 = [np.array(ss) for ss in _next_state]

        # the state array must not be empty
        if any(s.size == 0 for s in _state2) or any(s.size == 0 for s in _next_state2):
            print("Warning: empty state array")
            self.Xs = None
            self.Y = None
            return

        try:
            cur_qvalues = self.q_network.predict(_state2)
            next_qvalues = self.q_network_bar.predict(_next_state2)
            # [batch, 4]
            target = np.copy(cur_qvalues)

            for i in range(len(_action)):
                target[i, _action[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + self.dic_agent_conf["GAMMA"] * \
                                        np.max(next_qvalues[i, :])
            self.Xs = _state2
            self.Y = target
        except Exception as e:
            print(f"Error while computing Q-values: {e}")
            self.Xs = None
            self.Y = None
        
    def save_network(self, file_name):
        """Save the network together with the indicator data."""
        # parent save_network()
        super().save_network(file_name)
        
        # save the indicator data
        if hasattr(self, 'save_metrics') and self.save_metrics and self.reward_history:
            self.save_metrics_to_csv()