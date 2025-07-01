from config import *

import random
import numpy as np
import tensorflow as tf
random.seed(SEED)
tf.random.set_seed(SEED)

def check_update_conditions(t, memory_buffer):
    """
    Determines if the conditions are met to perform a learning update.

    Checks if the current time step t is a multiple of num_steps_upd and if the
    memory_buffer has enough experience tuples to fill a mini-batch (for example, if the
    mini-batch size is 64, then the memory buffer should have more than 64 experience
    tuples in order to perform a learning update).

    Args:
        t (int):
            The current time step.
        memory_buffer (deque):
            A deque containing experiences. The experiences are stored in the memory
            buffer as namedtuples: namedtuple("Experience", field_names=["state",
            "action", "reward", "next_state", "done"]).

    Returns:
       A boolean that will be True if conditions are met and False otherwise. 
    """

    return len(memory_buffer) > MINIBATCH_SIZE and (t + 1) % NUM_STEPS_FOR_UPDATE == 0


def get_experiences_smart(memory_buffer, observation_space, action_space):
    """
    Returns a random sample of experience tuples drawn from the memory buffer.

    Args:
        memory_buffer (deque): A deque of Experience namedtuples:
            (state, action, reward, next_state, done)
        observation_space (gym.spaces.Box): Environment's observation space
        action_space (gym.spaces.Box): Environment's action space

    Returns:
        Tuple of TensorFlow tensors: (states, actions, rewards, next_states, dones)
    """
    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)

    # Dynamically allocate based on environment specs
    obs_shape = observation_space.shape
    obs_dtype = observation_space.dtype

    act_shape = action_space.shape
    act_dtype = action_space.dtype

    states = np.empty((MINIBATCH_SIZE, *obs_shape), dtype=obs_dtype)
    actions = np.empty((MINIBATCH_SIZE, *act_shape), dtype=act_dtype)
    rewards = np.empty((MINIBATCH_SIZE,), dtype=np.float32)
    next_states = np.empty((MINIBATCH_SIZE, *obs_shape), dtype=obs_dtype)
    dones = np.empty((MINIBATCH_SIZE,), dtype=np.float32)

    for i, e in enumerate(experiences):
        states[i] = e.state
        actions[i] = e.action
        rewards[i] = e.reward
        next_states[i] = e.next_state
        dones[i] = float(e.done)

    return (
        tf.convert_to_tensor(states, dtype=tf.float32),
        tf.convert_to_tensor(actions, dtype=tf.float32),
        tf.convert_to_tensor(rewards, dtype=tf.float32),
        tf.convert_to_tensor(next_states, dtype=tf.float32),
        tf.convert_to_tensor(dones, dtype=tf.float32),
    )


def get_experiences(memory_buffer, k):
    experiences = random.sample(memory_buffer, k)
    states, actions, rewards, next_states, dones = zip(*experiences)

    states = np.stack(states).astype(np.float32)
    next_states = np.stack(next_states).astype(np.float32)

    return (
        tf.convert_to_tensor(states),
        tf.convert_to_tensor(actions, dtype=tf.int32),
        tf.convert_to_tensor(rewards, dtype=tf.float32),
        tf.convert_to_tensor(next_states),
        tf.convert_to_tensor(dones, dtype=tf.float32)
    )


def get_new_eps(epsilon, decrease=True):
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


def update_target_network(q_network, target_q_network):
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
def compute_loss_discreate(experiences, gamma, q_network, target_q_network):
    """ 
    Calculates the loss.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Keras model for predicting the targets

    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """

    states, actions, rewards, next_states, done_vals = experiences

    # Get max Q-values from target network for next states
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=1)

    # Compute target values: y = r + (1 - done) * gamma * max Q'(s', a')
    y_targets = rewards + (1.0 - done_vals) * gamma * max_qsa
    y_targets = tf.stop_gradient(y_targets)

    # Predict Q-values from current q_network
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
def agent_learn(experiences, gamma, target_q_network, optimizer, q_network):
    """
    Updates the weights of the Q networks.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss_discreate(
            experiences, gamma, q_network, target_q_network)
    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
    update_target_network(q_network, target_q_network)


def get_action(q_values, epsilon, num_actions=4):
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
        return int(get_greedy_action(q_values))
    return np.random.randint(num_actions)


@tf.function
def get_greedy_action(q_values):
    return tf.argmax(q_values, axis=1)[0]
