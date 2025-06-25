import random

import imageio
import numpy as np
import tensorflow as tf

from config import *

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

    return (t + 1) % NUM_STEPS_FOR_UPDATE == 0 and len(memory_buffer) > MINIBATCH_SIZE

def get_experiences(memory_buffer):
    """
    Returns a random sample of experience tuples drawn from the memory buffer.

    Retrieves a random sample of experience tuples from the given memory_buffer and
    returns them as TensorFlow Tensors. The size of the random sample is determined by
    the mini-batch size (MINIBATCH_SIZE). 

    Args:
        memory_buffer (deque):
            A deque containing experiences. The experiences are stored in the memory
            buffer as namedtuples: namedtuple("Experience", field_names=["state",
            "action", "reward", "next_state", "done"]).

    Returns:
        A tuple (states, actions, rewards, next_states, done_vals) where:

            - states are the starting states of the agent.
            - actions are the actions taken by the agent from the starting states.
            - rewards are the rewards received by the agent after taking the actions.
            - next_states are the new states of the agent after taking the actions.
            - done_vals are the boolean values indicating if the episode ended.

        All tuple elements are TensorFlow Tensors whose shape is determined by the
        mini-batch size and the given Gym environment. All
        TensorFlow Tensors have elements with dtype=tf.float32.
    """
    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)

    # Preallocate numpy arrays to avoid overhead of list comprehensions + conversions
    states = np.empty((MINIBATCH_SIZE, *experiences[0].state.shape), dtype=np.float32)
    actions = np.empty((MINIBATCH_SIZE,), dtype=np.float32)
    rewards = np.empty((MINIBATCH_SIZE,), dtype=np.float32)
    next_states = np.empty((MINIBATCH_SIZE, *experiences[0].next_state.shape), dtype=np.float32)
    dones = np.empty((MINIBATCH_SIZE,), dtype=np.float32)

    for i, e in enumerate(experiences):
        states[i] = e.state
        actions[i] = e.action
        rewards[i] = e.reward
        next_states[i] = e.next_state
        dones[i] = float(e.done)

    # Convert once at the end
    return (
        tf.convert_to_tensor(states),
        tf.convert_to_tensor(actions),
        tf.convert_to_tensor(rewards),
        tf.convert_to_tensor(next_states),
        tf.convert_to_tensor(dones),
    )


def get_new_eps(epsilon):
    """
    Updates the epsilon value for the ε-greedy policy.

    Gradually decreases the value of epsilon towards a minimum value (E_MIN) using the
    given ε-decay rate (E_DECAY).

    Args:
        epsilon (float):
            The current value of epsilon.

    Returns:
       A float with the updated value of epsilon.
    """

    return max(E_MIN, E_DECAY * epsilon)

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
def compute_loss(experiences, gamma, q_network, target_q_network):
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

    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences

    # Compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)

    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    y_targets = tf.stop_gradient(rewards + (1 - done_vals) * gamma * max_qsa)

    # Get the q_values and reshape to match y_targets
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))

    # Compute the loss
    loss = tf.keras.losses.MSE(y_targets, q_values)
    return loss


def create_video(filename, env, q_network, fps=30):
    """
    Creates a video of an agent interacting with a Gym environment.

    The agent will interact with the given env environment using the q_network to map
    states to Q values and using a greedy policy to choose its actions (i.e it will
    choose the actions that yield the maximum Q values).

    The video will be saved to a file with the given filename. The video format must be
    specified in the filename by providing a file extension (.mp4, .gif, etc..). If you 
    want to embed the video in a Jupyter notebook using the embed_mp4 function, then the
    video must be saved as an MP4 file. 

    Args:
        filename (string):
            The path to the file to which the video will be saved. The video format will
            be selected based on the filename. Therefore, the video format must be
            specified in the filename by providing a file extension (i.e.
            "./videos/xxx.mp4"). To see a list of supported formats see the
            imageio documentation: https://imageio.readthedocs.io/en/v2.8.0/formats.html
        env (Gym Environment): 
            The Gym environment the agent will interact with.
        q_network (tf.keras.Sequential):
            A TensorFlow Keras Sequential model that maps states to Q values.
        fps (int):
            The number of frames per second. Specifies the frame rate of the output
            video. The default frame rate is 30 frames per second.  
    """

    with imageio.get_writer(filename, fps=fps, codec='libx264') as video:
        done = False
        state, _ = env.reset()
        frame = env.render()
        video.append_data(frame)

        while not done:
            state_input = np.expand_dims(state, axis=0)
            q_values = q_network(state_input)
            action = np.argmax(q_values.numpy()[0])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frame = env.render()
            video.append_data(frame)

            state = next_state
