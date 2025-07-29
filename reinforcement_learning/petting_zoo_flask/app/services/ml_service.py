
from dataclasses import astuple
import cv2
from app import app
from app.config.env_config import *
from app.config.session_state import PlayerState, SessionState

import random
import numpy as np
import tensorflow as tf

random.seed(SEED)
tf.random.set_seed(SEED)


class MLService:
    _instance = None

    def __init__(self):
        pass

    def train_model(self, sid, session: SessionState):
        with session.env_config.lock:
            for agent_name, agent_state in session.agents.items():
                app.logger.info(
                    f"{sid}: Teaching model, for {agent_name.capitalize()} with score: {agent_state.total_reward}")
                for _ in range(10):
                    ml_service.train_step(session, agent_state)

            session.env_config.epsilon = ml_service.get_new_eps(
                session.env_config.epsilon)
            session.q_network.save(session.env_config.model_path)
            session.target_q_network.save_weights(
                session.env_config.weights_path)

    def train_step(self, session: SessionState, current_agent: PlayerState):
        if len(current_agent.memory_buffer) < MINIBATCH_SIZE:
            return
        experiences = self.get_experiences(
            current_agent.memory_buffer, MINIBATCH_SIZE)
        self.agent_learn(
            experiences,
            GAMMA,
            session.target_q_network,
            session.optimizer,
            session.q_network,
        )

    def preprocess_state(self, input_shape: tuple[int], state: np.ndarray):
        gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(
            gray, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_AREA)
        processed = np.expand_dims(resized, axis=-1)
        return processed.astype(np.float32)

    def build_q_network(self, input_shape, num_actions):
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=input_shape),
            tf.keras.layers.Rescaling(1. / 255),
            tf.keras.layers.Conv2D(32, (8, 8), strides=4, activation='relu'),
            tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(num_actions)
        ])

    def build_multi_head_q_network(input_shape, env_heads):
        """
        Builds a shared Q-network with multiple heads.

        Args:
            input_shape: (52, 40, 1)
            dict like {"wizard_of_wor_v3": 9, "mario_bros_v3": 18}
        Returns:
            model: tf.keras.Model with multiple outputs for pro player
        """
        inputs = tf.keras.Input(shape=input_shape)

        x = tf.keras.layers.Rescaling(1./255)(inputs)
        x = tf.keras.layers.Conv2D(32, (8, 8), strides=4, activation='relu')(x)
        x = tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        shared = tf.keras.layers.Dense(512, activation='relu')(x)

        outputs = {}
        for env_name, num_actions in env_heads.items():
            outputs[env_name] = tf.keras.layers.Dense(
                num_actions, name=f"{env_name}_head")(shared)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def update_target_network(self, q_network, target_q_network):
        """
        Updates the weights of the target Q-Network using a soft update.

        The weights of the target_q_network are updated using the soft update rule:

            w_target = (TAU * w) + (1 - TAU) * w_target

        where w_target are the weights of the target_q_network, TAU is the soft update
        parameter, and w are the weights of the q_network.

        Args:
            q_network (tf.keras.Sequential): 
                The Q-Network. 
            target_q_network (tf.keras.Sequential):
                The Target Q-Network.
        """

        for target_weights, q_net_weights in zip(
            target_q_network.weights, q_network.weights
        ):
            target_weights.assign(TAU * q_net_weights +
                                  (1.0 - TAU) * target_weights)

    @tf.function
    def compute_loss_discreate(self, experiences, gamma, q_network, target_q_network, env_name=False):
        """ 
        Calculates the loss.

        Args:
        experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
        gamma: (float) The discount factor.
        q_network: (tf.keras.Sequential) Keras model for predicting the q_values
        target_q_network: (tf.keras.Sequential) Keras model for predicting the targets
        env_name: str, which head to use if any and the model multihead

        Returns:
        loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
                the y targets and the Q(s,a) values.
        """

        states, actions, rewards, next_states, done_vals = experiences

        # Get max Q-values from target network for next states
        if env_name:
            next_q_values = target_q_network(next_states)[env_name]
        else:
            next_q_values = target_q_network(next_states)

        max_qsa = tf.reduce_max(next_q_values, axis=1)

        # Compute target values: y = r + (1 - done) * gamma * max Q'(s', a')
        y_targets = rewards + (1.0 - done_vals) * gamma * max_qsa
        y_targets = tf.stop_gradient(y_targets)

        # Predict Q-values from current q_network
        if env_name:
            q_values_all = q_network(states)[env_name]
        else:
            q_values_all = q_network(states)

        # Get Q-values for the actions actually taken
        batch_size = tf.shape(actions)[0]
        indices = tf.stack(
            [tf.range(batch_size), tf.cast(actions, tf.int32)], axis=1)
        q_values = tf.gather_nd(q_values_all, indices)

        # Compute Mean Squared Error loss
        loss = tf.reduce_mean(tf.keras.losses.MSE(y_targets, q_values))
        return loss

    @tf.function(jit_compile=True)
    def agent_learn(self, experiences, gamma,
                    target_q_network, optimizer, q_network,
                    env_name = False):
        """
        Updates the weights of the Q networks.

        Args:
        experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
        gamma: (float) The discount factor.
        q_network: (tf.keras.Sequential) Keras model for predicting the q_values
        target_q_network: (tf.keras.Sequential) Keras model for predicting the targets
        env_name: str, which head to use if any and the model multihead
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss_discreate(
                experiences, gamma, q_network, target_q_network, env_name)
        gradients = tape.gradient(loss, q_network.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, q_network.trainable_variables))
        self.update_target_network(q_network, target_q_network)

    def get_action(self, q_values, epsilon, num_actions=4):
        """
        Efficient ε-greedy action selection.

        Args:
            q_values (tf.Tensor): Q-values for actions, shape (1, num_actions)
            epsilon (float): Exploration rate
            num_actions (int): Number of possible actions

        Returns:
            int: Chosen action
        """
        if np.random.rand() > epsilon:
            return int(self.get_greedy_action(q_values))
        return np.random.randint(num_actions)

    @tf.function
    def get_greedy_action(self, q_values):
        return tf.argmax(q_values, axis=1)[0]

    def get_experiences(self, memory_buffer, k):
        experiences = random.sample(memory_buffer, k)
        tuples = [astuple(exp) for exp in experiences]
        states, actions, rewards, next_states, dones = zip(*tuples)

        states = np.stack(states).astype(np.float32)
        next_states = np.stack(next_states).astype(np.float32)

        return (
            tf.convert_to_tensor(states),
            tf.convert_to_tensor(actions, dtype=tf.int32),
            tf.convert_to_tensor(rewards, dtype=tf.float32),
            tf.convert_to_tensor(next_states),
            tf.convert_to_tensor(dones, dtype=tf.float32)
        )

    def get_new_eps(self, epsilon, decrease=True):
        """
        Updates the epsilon value for the ε-greedy policy.

        Either decreases epsilon towards E_MIN (exploration decay) or increases it 
        towards E_MAX (exploration growth), depending on the `decrease` flag.

        Args:
            epsilon (float): The current value of epsilon.
            decrease (bool): Whether to decrease or increase epsilon.

        Returns:
            float: The updated value of epsilon.
        """
        return max(E_MIN, E_DECAY * epsilon) if decrease else min(E_MAX, E_GROW * epsilon)

    def load_model(self, sid, env_config, session_state, obs_shape, num_actions):
        if os.path.exists(env_config.model_path) and os.path.exists(env_config.weights_path):
            app.logger.info(
                f"{sid}: Loading existing models ${env_config.model_path}")
            session_state.q_network = tf.keras.models.load_model(
                env_config.model_path)
            session_state.target_q_network = ml_service.build_q_network(
                obs_shape, num_actions)
            session_state.target_q_network.load_weights(
                session_state.env_config.weights_path)
            session_state.optimizer = session_state.q_network.optimizer
        else:
            app.logger.info(f"{sid}: Initializing new models...")
            session_state.q_network = ml_service.build_q_network(
                obs_shape, num_actions)
            session_state.target_q_network = ml_service.build_q_network(
                obs_shape, num_actions)
            session_state.optimizer = tf.keras.optimizers.Adam(
                learning_rate=ALPHA)
            dummy_input = tf.zeros((1, *obs_shape), dtype=tf.float32)
            _ = session_state.q_network(dummy_input)
            _ = session_state.target_q_network(dummy_input)
            session_state.target_q_network.set_weights(
                session_state.q_network.get_weights())
            session_state.q_network.compile(
                optimizer=session_state.optimizer, loss='mse')
        zero_grads = [tf.zeros_like(
            v) for v in session_state.q_network.trainable_variables]
        session_state.optimizer.apply_gradients(
            zip(zero_grads, session_state.q_network.trainable_variables))


ml_service = MLService()
