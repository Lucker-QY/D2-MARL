"""
AttendLight agent with custom attention mechanism compatible with TensorFlow 2.4.
Observations: [num_in_seg_attend]
Reward: -Pressure
"""

import os
from .network_agent import NetworkAgent
from tensorflow.keras.layers import Dense, concatenate, Reshape, Conv1D, Lambda, LSTM, Multiply, Dot
from tensorflow.keras import Input, Model
from tensorflow.keras.models import save_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from tensorflow.keras import backend as K
import tensorflow as tf


class AttendLightAgent(NetworkAgent):
    def build_network(self):
        ins0 = Input(shape=(self.num_lane * 8,))

        # Reshape to [batch, num_lane, dim]
        feat1 = Reshape((self.num_lane * 2, 4))(ins0)
        feat1 = Dense(32, activation="relu")(feat1)

        # Custom attention mechanism compatible with TF 2.4
        # Split into lanes
        lane_feats = tf.split(feat1, self.num_lane * 2, axis=1)

        # Custom multi-head attention using Dense layers
        # Since MultiHeadAttention is not available in TF 2.4, we implement a simplified version
        def custom_attention(query, keys, values, num_heads=4):
            """Simplified multi-head attention compatible with TF 2.4"""
            # Project to multiple heads
            head_dim = 8
            query_proj = Dense(head_dim)(query)
            key_proj = Dense(head_dim)(keys)
            value_proj = Dense(head_dim)(values)

            # Split into heads
            query_heads = tf.split(query_proj, num_heads, axis=-1)
            key_heads = tf.split(key_proj, num_heads, axis=-1)
            value_heads = tf.split(value_proj, num_heads, axis=-1)

            attention_outputs = []
            for q_h, k_h, v_h in zip(query_heads, key_heads, value_heads):
                # Scaled dot-product attention
                dk = tf.cast(tf.shape(k_h)[-1], tf.float32)
                scores = tf.matmul(q_h, k_h, transpose_b=True) / tf.sqrt(dk)
                attention_weights = tf.nn.softmax(scores, axis=-1)
                attention_output = tf.matmul(attention_weights, v_h)
                attention_outputs.append(attention_output)

            # Concatenate all heads
            concat_attention = tf.concat(attention_outputs, axis=-1)
            return concat_attention

        # Compute phase-level features with attention
        phase_feats_map_1 = []
        mean_layer = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))

        for i in range(self.num_phases):
            # Get lanes for this phase
            phase_lane_indices = self.phase_map[i] if hasattr(self, 'phase_map') else list(range(i*2, (i+1)*2))
            tmp_feats = tf.concat([lane_feats[idx] for idx in phase_lane_indices], axis=1)
            tmp_feat_mean = mean_layer(tmp_feats)

            # Apply attention
            tmp_feat_attended = custom_attention(tmp_feat_mean, feat1, feat1, num_heads=4)
            phase_feats_map_1.append(tmp_feat_attended)

        # Concatenate all phase features
        phase_feat_all = concatenate(phase_feats_map_1, axis=1)

        # Apply attention between phases (self-attention)
        phase_attention = custom_attention(phase_feat_all, phase_feat_all, phase_feat_all, num_heads=4)

        hidden = Dense(20, activation="relu")(phase_attention)
        hidden = Dense(20, activation="relu")(hidden)
        q_values = Dense(1, activation="linear")(hidden)
        q_values = Reshape((self.num_actions,))(q_values)

        network = Model(inputs=ins0, outputs=q_values)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"], epsilon=1e-08),
                        loss=self.dic_agent_conf["LOSS_FUNCTION"])
        network.summary()
        return network

    def prepare_Xs_Y(self, memory):
        ind_end = len(memory)
        print("memory size before forget: {0}".format(ind_end))

        ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])
        memory_after_forget = memory[ind_sta: ind_end]
        print("memory size after forget:", len(memory_after_forget))

        sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
        sample_slice = random.sample(memory_after_forget, sample_size)
        print("memory samples number:", sample_size)

        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        _state = []
        _next_state = []
        _action = []
        _reward = []

        for i in range(len(sample_slice)):
            state, action, next_state, reward, _, _, _ = sample_slice[i]
            _state.append(state[used_feature[0]])
            _next_state.append(next_state[used_feature[0]])
            _action.append(action)
            _reward.append(reward)

        _state2 = np.array(_state)
        _next_state2 = np.array(_next_state)

        cur_qvalues = self.q_network(_state2)
        next_qvalues = self.q_network_bar(_next_state2)

        target = np.copy(cur_qvalues)
        for i in range(len(sample_slice)):
            target[i, _action[i]] = _reward[i] / self.dic_agent_conf["NORMAL_FACTOR"] + \
                                    self.dic_agent_conf["GAMMA"] * np.max(next_qvalues[i, :])

        self.Xs = _state2
        self.Y = target

    def choose_action(self, step_num, states):
        feats = []
        for s in states:
            tmp_feat0 = s[self.dic_traffic_env_conf["LIST_STATE_FEATURE"][0]]
            feats.append(tmp_feat0)

        feats = np.array(feats)
        q_values = self.q_network.predict(feats)

        action = self.epsilon_choice(q_values)
        return action

    def epsilon_choice(self, q_values):
        max_1 = np.expand_dims(np.argmax(q_values, axis=-1), axis=-1)
        rand_1 = np.random.randint(self.num_actions, size=(len(q_values), 1))
        _p = np.concatenate([max_1, rand_1], axis=-1)
        select = np.random.choice([0, 1], size=len(q_values), p=[1 - self.dic_agent_conf["EPSILON"],
                                                                   self.dic_agent_conf["EPSILON"]])
        act = _p[np.arange(len(q_values)), select]
        return act


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, loss):
        score = -loss
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop
