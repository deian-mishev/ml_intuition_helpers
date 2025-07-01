
import tensorflow as tf
import logging
from collections import deque, namedtuple
import numpy as np
import gymnasium as gym
from config import *
import utils
import imageio

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


def main():
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    # q_network = tf.keras.models.load_model('./data/lunar_lander_model.h5')
    q_network = build_q_network(obs_shape, num_actions)
    target_q_network = build_q_network(obs_shape, num_actions)
    target_q_network.set_weights(q_network.get_weights())

    optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)

    memory_buffer = deque(maxlen=MEMORY_SIZE)
    epsilon = 1.0
    recent_points = deque(maxlen=NUM_P_AV)
    prev_avg_points = float('-inf')

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
            action = utils.get_action(q_values, epsilon)
            # Take action A and receive reward R and the next state S'
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # Store experience tuple (S,A,R,S') in the memory buffer.
            # We store the done variable as well for convenience.
            memory_buffer.append(Experience(
                state, action, reward, next_state, done))

            if utils.check_update_conditions(t, memory_buffer):
                experiences = utils.get_experiences_smart(
                    memory_buffer, env.observation_space, env.action_space)
                utils.agent_learn(experiences, GAMMA,
                                  target_q_network, optimizer, q_network)

            state = next_state
            total_reward += reward

            if done:
                break

        recent_points.append(total_reward)
        new_avg = sum(recent_points) / len(recent_points)
        epsilon = utils.get_new_eps(epsilon, prev_avg_points < new_avg)

        print(f"\rEpisode {episode} | Epsilon {epsilon} | Average (last {NUM_P_AV}): {new_avg:.2f}",
              end="\n" if episode % NUM_P_AV == 0 else "")

        if new_avg >= CUTTOFF_AVG:
            print(f"\n\nEnvironment solved in {episode} episodes!")
            q_network.save("./data/lunar_lander_model.h5")
            target_q_network.save("./data/lunar_lander_weights.h5")
            break

        prev_avg_points = new_avg

    filename = "./data/lunar_lander.mp4"
    create_video(filename, env, q_network)


def create_video(filename, env, q_network, fps=30):
    with imageio.get_writer(filename, fps=fps, codec='libx264') as video:
        done = False

        # Handle reset for Gym and Gymnasium APIs
        reset_result = env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            state, _ = reset_result
        else:
            state = reset_result

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

    return filename


if __name__ == "__main__":
    main()
