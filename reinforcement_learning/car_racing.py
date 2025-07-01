import keyboard
import time
import tensorflow as tf
import logging
import imageio
from collections import deque, namedtuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import *
import utils
import random
random.seed(SEED)

tf.random.set_seed(SEED)
logging.getLogger().setLevel(logging.ERROR)

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

def preprocess_frame_np(frame):
    gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
    return gray.astype(np.float32)[..., None]

def manual_drive(env, buffer, num_steps):
    state, _ = env.reset()
    state = preprocess_frame_np(state)
    time.sleep(1)
    utils.focus_env_window()
    start_time = time.time()
    pressed_key = False
    
    print(f"\nManual driving mode: use arrow keys in 10 seconds from now, press ESC to stop.")
    while time.time() - start_time < 10:
        if keyboard.is_pressed('up') or keyboard.is_pressed('down') or \
           keyboard.is_pressed('left') or keyboard.is_pressed('right'):
            pressed_key = True
            break
        time.sleep(0.1)

    if not pressed_key:
        return False

    done = False
    steps = 0

    ACTION_DICT = {
        'none': 0,
        'right': 1,
        'left': 2,
        'gas': 3,
        'break': 4
    }
    while not done and steps < num_steps:
        action = ACTION_DICT['gas']
        if keyboard.is_pressed('down'):
            action = ACTION_DICT['break']
        elif keyboard.is_pressed('left'):
            action = ACTION_DICT['left']
        elif keyboard.is_pressed('right'):
            action = ACTION_DICT['right']
        elif keyboard.is_pressed('esc'):
            break

        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = preprocess_frame_np(next_state)
        done = terminated or truncated

        buffer.append(Experience(state, action, reward, next_state, done))

        state = next_state
        steps += 1

    return True


def main():
    env = gym.make("CarRacing-v3", render_mode="rgb_array",
                   lap_complete_percent=0.95, domain_randomize=False, continuous=False)

    obs_shape = spaces.Box(
        low=0, high=255, shape=(96, 96, 1), dtype=np.uint8)
    state, _ = env.reset()
    obs_shape = obs_shape.shape
    num_actions = env.action_space.n - 1

    q_network = tf.keras.models.load_model("./data/car_racing_model_v3.h5")
    target_q_network = tf.keras.models.load_model(
        "./data/car_racing_weights_v3.h5")
    # q_network = build_q_network(obs_shape, num_actions)
    # target_q_network = build_q_network(obs_shape, num_actions)
    # target_q_network.set_weights(q_network.get_weights())

    optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)

    memory_buffer = deque(maxlen=MEMORY_SIZE)
    epsilon = E_MAX
    recent_points = deque(maxlen=NUM_P_AV)
    prev_avg_points = float('-inf')
    for episode in range(1, NUM_OF_EPISODES - 1):

        state, _ = env.reset()
        state = preprocess_frame_np(state)
        total_reward = 0.0

        for t in range(MAX_NUM_OF_STEPS):
            state_input = tf.expand_dims(state, 0)
            q_values = q_network(state_input)

            action = utils.get_action(q_values, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if reward < 0:
                reward *= PENALTY_FACTOR
                
            next_state = preprocess_frame_np(next_state)
            memory_buffer.append(Experience(
                state, action, reward, next_state, done))

            if utils.check_update_conditions(t, memory_buffer):
                experiences = utils.get_experiences(memory_buffer, MINIBATCH_SIZE)
                utils.agent_learn(experiences, GAMMA, target_q_network,
                            optimizer, q_network)

            state = next_state
            total_reward += reward

            if done:
                break
                
        recent_points.append(total_reward)
        new_avg = sum(recent_points) / len(recent_points)
        epsilon = utils.get_new_eps(epsilon, prev_avg_points < new_avg)

        if prev_avg_points > new_avg and episode % NUM_S_M == 0:
            env_human = gym.make("CarRacing-v3", render_mode="human",
                                 lap_complete_percent=0.95, domain_randomize=False, continuous=False)
            temp = deque(maxlen=MAX_NUM_OF_STEPS*2)
            if manual_drive(env_human, temp, MAX_NUM_OF_STEPS*2):
                env_human.close()
                experiences = utils.get_experiences(temp, len(temp))
                utils.agent_learn(experiences, GAMMA, target_q_network,
                            optimizer, q_network)
                memory_buffer.extend(temp)
            else:
                env_human.close()
                create_video(FILE_FEEDBACK, env, q_network)

        print(f"\rEpisode {episode} | Epsilon {epsilon} | Average (last {NUM_P_AV}): {new_avg:.2f}",
              end="\n" if episode % NUM_P_AV == 0 else "")

        if new_avg >= CUTTOFF_AVG:
            print(f"\n\nEnvironment solved in {episode} episodes!")
            q_network.save("./data/car_racing_model_v4.h5")
            target_q_network.save("./data/car_racing_weights_v4.h5")
            break

        prev_avg_points = new_avg

    create_video(FILE_FEEDBACK, env, q_network)


def create_video(filename, env, q_network, fps=30):

    with imageio.get_writer(filename, fps=fps, codec='libx264') as video:
        done = False

        # Reset the environment
        reset_result = env.reset()
        state = reset_result[0] if isinstance(
            reset_result, tuple) else reset_result

        frame = env.render()
        video.append_data(frame)

        while not done:
            # Convert RGB to grayscale and add batch dimension
            gray_state = tf.image.rgb_to_grayscale(state)
            state_input = tf.expand_dims(gray_state, axis=0)

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
            state = next_state

    return filename


if __name__ == "__main__":
    main()
