
import tensorflow as tf
import logging
from collections import deque, namedtuple
import numpy as np
import gymnasium as gym
from config import *
import utils

tf.random.set_seed(SEED)
logging.getLogger().setLevel(logging.ERROR)

Experience = namedtuple("Experience", field_names=[
                        "state", "action", "reward", "next_state", "done"])

def build_q_network(input_shape, num_actions):
    """
    Creates a Q-network with 2 hidden layers.
    """
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_actions)
    ])

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
        return int(np.argmax(q_values.numpy()))
    return np.random.randint(num_actions)

@tf.function(jit_compile=True)
def agent_learn(experiences, gamma, target_q_network, optimizer, q_network):
    """
    Updates the weights of the Q networks.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.

    """

    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = utils.compute_loss(
            experiences, gamma, q_network, target_q_network)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)

    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # update the weights of target q_network
    utils.update_target_network(q_network, target_q_network)


def main():
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    q_network = build_q_network(obs_shape, num_actions)
    target_q_network = build_q_network(obs_shape, num_actions)
    target_q_network.set_weights(q_network.get_weights())

    optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)

    memory_buffer = deque(maxlen=MEMORY_SIZE)
    epsilon = 1.0
    recent_points = deque(maxlen=NUM_P_AV)

    for episode in range(1, NUM_OF_EPISODES + 1):

        # Reset the environment to the initial state and get the initial state
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0.0

        for t in range(MAX_NUM_OF_STEPS):

            # From the current state S choose an action A using an ε-greedy policy
            # state needs to be the right shape for the q_network
            state_input = tf.expand_dims(state, 0)  # TF-friendly
            q_values = q_network(state_input)
            action = get_action(q_values, epsilon)
            # Take action A and receive reward R and the next state S'
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # Store experience tuple (S,A,R,S') in the memory buffer.
            # We store the done variable as well for convenience.
            memory_buffer.append(Experience(
                state, action, reward, next_state, done))

            if utils.check_update_conditions(t, memory_buffer):
                experiences = utils.get_experiences(memory_buffer)
                agent_learn(experiences, GAMMA, target_q_network, optimizer, q_network)

            state = next_state
            total_reward += reward

            if done:
                break

        recent_points.append(total_reward)
        avg_points = sum(recent_points) / len(recent_points)
        epsilon = utils.get_new_eps(epsilon)

        print(f"\rEpisode {episode} | Average (last {NUM_P_AV}): {avg_points:.2f}",
                end="\n" if episode % NUM_P_AV == 0 else "")

        if avg_points >= CUTTOFF_AVG:
            print(f"\n\nEnvironment solved in {episode} episodes!")
            q_network.save("./data/lunar_lander_model.h5")
            break

    filename = "./data/lunar_lander.mp4"
    utils.create_video(filename, env, q_network)

if __name__ == "__main__":
    main()