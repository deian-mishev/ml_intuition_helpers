import tensorflow as tf

from collections import deque, namedtuple
import numpy as np
import gymnasium as gym
from vizdoom import gymnasium_wrapper
from config import *
import utils
import random
import imageio
random.seed(SEED)

tf.random.set_seed(SEED)
Experience = namedtuple("Experience", field_names=[
                        "state", "action", "reward", "next_state", "done"])


def build_q_network(input_shape, num_actions):
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(
            32, (8, 8), strides=4, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.Conv2D(
            64, (4, 4), strides=2, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.Conv2D(
            64, (3, 3), strides=1, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu',
                              kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(num_actions)
    ])


def main():
    env = gym.make("VizdoomCorridor-v0", render_mode="human")
    env.unwrapped.game.set_doom_skill(2)
    state, _ = env.reset()
    state = state['screen']
    obs_shape = state.shape
    num_actions = env.action_space.n

    q_network = tf.keras.models.load_model("./data/viz_doom_v2.h5")
    target_q_network = tf.keras.models.load_model(
        "./data/viz_doom_weights_v2.h5")
    # q_network = build_q_network(obs_shape, num_actions)
    # target_q_network = build_q_network(obs_shape, num_actions)
    # target_q_network.set_weights(q_network.get_weights())
    optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)
    recent_points = deque(maxlen=NUM_P_AV)
    memory_buffer = deque(maxlen=MEMORY_SIZE)
    epsilon = E_MAX
    for episode in range(1, NUM_OF_EPISODES - 1):

        state, _ = env.reset()
        state = state['screen']
        total_reward = 0.0

        for t in range(MAX_NUM_OF_STEPS):
            state_input = tf.expand_dims(state, 0)
            q_values = q_network(state_input)

            action = utils.get_action(q_values, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state['screen']
            done = terminated or truncated
            if reward < 0:
                reward *= PENALTY_FACTOR

            memory_buffer.append(Experience(
                state, action, reward, next_state, done))

            if utils.check_update_conditions(t, memory_buffer):
                experiences = utils.get_experiences(
                    memory_buffer, MINIBATCH_SIZE)
                utils.agent_learn(experiences, GAMMA, target_q_network,
                                  optimizer, q_network)

            state = next_state
            total_reward += reward

            if done:
                break

        recent_points.append(total_reward)
        new_avg = sum(recent_points) / len(recent_points)
        epsilon = utils.get_new_eps(epsilon)

        print(f"\rEpisode {episode} | Epsilon {epsilon} | Last {total_reward} | Average (last {NUM_P_AV}): {new_avg:.2f}",
              end="\n" if episode % NUM_P_AV == 0 else "")

        if new_avg >= CUTTOFF_AVG:
            print(f"\n\nEnvironment solved in {episode} episodes!")
            q_network.save("./data/viz_doom_v3.h5")
            target_q_network.save("./data/viz_doom_weights_v3.h5")
            break
    env.close()
    create_video(FILE_FEEDBACK, q_network)


def create_video(filename, q_network, fps=30):
    env = gym.make("VizdoomCorridor-v0", render_mode="rgb_array")
    env.unwrapped.game.set_doom_skill(2)
    with imageio.get_writer(filename, fps=fps, codec='libx264') as video:
        done = False

        # Reset the environment
        reset_result, _ = env.reset()
        state = reset_result['screen']

        frame = env.render()
        video.append_data(frame)

        while not done:
            # Convert RGB to grayscale and add batch dimension
            state_input = tf.expand_dims(state, axis=0)

            # Predict and choose greedy action
            q_values = q_network(state_input)
            action = np.argmax(q_values.numpy()[0])

            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Add current frame to video
            frame = env.render()
            video.append_data(frame)

            # Move to next state
            state = next_state['screen']
    env.close()
    return filename


if __name__ == "__main__":
    main()
