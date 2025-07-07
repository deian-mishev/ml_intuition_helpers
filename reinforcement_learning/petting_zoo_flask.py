import eventlet
eventlet.monkey_patch()

from collections import deque, namedtuple
from dataclasses import dataclass, field
import os
import threading
import utils
from config import *
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
from pettingzoo.atari import boxing_v2, space_invaders_v2, tennis_v3, double_dunk_v3, ice_hockey_v2, mario_bros_v3, pong_v3, wizard_of_wor_v3
from flask_socketio import SocketIO, emit, disconnect
from flask import Flask, jsonify, request
from typing import Callable, Optional

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
global_lock = threading.Lock()
client_sessions = {}

epsilon = 1.0
input_timeout = 0.01


@dataclass
class EnvironmentConfig:
    KEY_MAP: dict
    model_path: str
    weights_path: str
    num_actions: int
    observation_space: tuple[int, ...]
    q_network: Optional[tf.keras.Model] = None
    target_q_network: Optional[tf.keras.Model] = None
    env: Optional[Callable] = field(default=None)
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None

@dataclass
class SessionState:
    env_config: EnvironmentConfig
    env: object
    state: object
    agent_iter: Optional[object] = None
    current_agent: Optional[object] = None
    thread: threading.Thread = None
    thread_stop_event: threading.Event = None
    nemesis_total_reward: int = field(default=0)
    action_queue: deque = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)
    memory_buffer: deque = field(
        default_factory=lambda: deque(maxlen=MEMORY_SIZE))


atari_pro = "./data/keras/atari_pro.keras"
atari_pro_weights = "./data/keras/atari_pro_weights.keras"

environments = {
    'Boxing': EnvironmentConfig(
        KEY_MAP={
            "0": 0, "s": 1, "8": 2, "6": 3, "4": 4, "2": 5,
            "9": 6, "7": 7, "3": 8, "1": 9, "w": 10, "d": 11,
            "a": 12, "x": 13, "e": 14, "q": 15, "c": 16, "z": 17
        },
        model_path="./data/keras/boxing_v2.keras",
        weights_path="./data/keras/boxing_v2_weights.keras",
        env=lambda render_mode="rgb_array": boxing_v2.env(
            render_mode=render_mode),
        num_actions=18,
        observation_space=(210, 160, 3)
    ),
    "Tennis": EnvironmentConfig(
        KEY_MAP={
            "8": 2, "6": 3, "4": 4,
            "2": 5, "9": 6, "7": 7,
            "3": 8, "1": 9, "w": 10,
            "d": 11, "a": 12, "x": 13,
            "e": 14, "q": 15, "c": 16,
            "z": 17, "s": 1
        },
        model_path="./data/keras/tennis_v3.keras",
        weights_path="./data/keras/tennis_v3_weights.keras",
        env=lambda render_mode="rgb_array": tennis_v3.env(
            render_mode=render_mode),
        num_actions=18,
        observation_space=(210, 160, 3)
    ),
    'Wizard of War': EnvironmentConfig(
        KEY_MAP={
            "8": 1, "6": 2, "4": 3,
            "2": 4, "9": 5, "7": 6,
            "3": 7, "1": 8, "s": 0
        },
        model_path="./data/keras/wizard_of_wor_v3.keras",
        weights_path="./data/keras/wizard_of_wor_v3_weights.keras",
        env=lambda render_mode="rgb_array": wizard_of_wor_v3.env(
            render_mode=render_mode),
        num_actions=9,
        observation_space=(210, 160, 3)
    ),
    'Mario Bros': EnvironmentConfig(
        KEY_MAP={
            "8": 2, "6": 3, "4": 4,
            "2": 5, "9": 6, "7": 7,
            "3": 8, "1": 9, "w": 10,
            "d": 11, "a": 12, "x": 13,
            "e": 14, "q": 15, "c": 16,
            "z": 17, "s": 1
        },
        model_path="./data/keras/mario_bros_v3.keras",
        weights_path="./data/keras/mario_bros_v3_weights.keras",
        env=lambda render_mode="rgb_array": mario_bros_v3.env(
            render_mode=render_mode),
        num_actions=18,
        observation_space=(210, 160, 3)
    ),
    'Ice Hockey': EnvironmentConfig(
        KEY_MAP={
            "8": 2, "6": 3, "4": 4,
            "2": 5, "9": 6, "7": 7,
            "3": 8, "1": 9, "w": 10,
            "d": 11, "a": 12, "x": 13,
            "e": 14, "q": 15, "c": 16,
            "z": 17, "s": 1
        },
        model_path="./data/keras/ice_hockey_v2.keras",
        weights_path="./data/keras/ice_hockey_v2_weights.keras",
        env=lambda render_mode="rgb_array": ice_hockey_v2.env(
            render_mode=render_mode),
        num_actions=18,
        observation_space=(210, 160, 3)
    ),
    'Double Dunk': EnvironmentConfig(
        KEY_MAP={
            "8": 2, "6": 3, "4": 4,
            "2": 5, "9": 6, "7": 7,
            "3": 8, "1": 9, "w": 10,
            "d": 11, "a": 12, "x": 13,
            "e": 14, "q": 15, "c": 16,
            "z": 17, "s": 1
        },
        model_path="./data/keras/double_dunk_v3.keras",
        weights_path="./data/keras/double_dunk_v3_weights.keras",
        env=lambda render_mode="rgb_array": double_dunk_v3.env(
            render_mode=render_mode),
        num_actions=18,
        observation_space=(210, 160, 3)
    ),
    'Pong': EnvironmentConfig(
        KEY_MAP={
            "4": 3, "6": 2, "s": 1
        },
        model_path="./data/keras/pong_v3_model.keras",
        weights_path="./data/keras/pong_v3_weights.keras",
        env=lambda render_mode="rgb_array": pong_v3.env(
            render_mode=render_mode),
        num_actions=6,
        observation_space=(210, 160, 3)
    ),
    "Space Invaders": EnvironmentConfig(
        KEY_MAP={
            "8": 2, "6": 4, "4": 3, "2": 5, "s": 1
        },
        model_path="./data/keras/space_invaders_v2.keras",
        weights_path="./data/keras/space_invaders_v2_weights.keras",
        env=lambda render_mode="rgb_array": space_invaders_v2.env(
            render_mode=render_mode),
        num_actions=6,
        observation_space=(210, 160, 3)
    )
}

Experience = namedtuple("Experience", field_names=[
                        "state", "action", "reward", "next_state", "done"])


def build_q_network(input_shape, num_actions):
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


def train_step(session: SessionState):
    if len(session.memory_buffer) < MINIBATCH_SIZE:
        return
    experiences = utils.get_experiences(session.memory_buffer, MINIBATCH_SIZE)
    utils.agent_learn(experiences, GAMMA, session.env_config.target_q_network,
                      session.env_config.optimizer, session.env_config.q_network)

def initialize_optimizer_vars(optimizer, model):
    zero_grads = [tf.zeros_like(v) for v in model.trainable_variables]
    optimizer.apply_gradients(zip(zero_grads, model.trainable_variables))

def render_frame(session: SessionState):
    frame = session.env.render()
    img = Image.fromarray(frame)
    buf = BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def frame_emit_loop(sid):
    session: SessionState = client_sessions.get(sid)
    if not session:
        print(f"No session for sid {sid}, terminating frame_emit_loop.")
        return

    while not session.thread_stop_event.is_set():
        try:
            obs, reward, terminated, truncated, info = session.env.last()
            done = terminated or truncated
            
            if session.current_agent == HUMAN_AGENT_NAME:
                session.nemesis_total_reward += reward
                with session.lock:
                    if session.action_queue:
                        action = session.action_queue.popleft()
                    else:
                        action = 0
            else:
                with global_lock:
                    q_values = session.env_config.q_network(tf.expand_dims(session.state, 0))
                action = utils.get_action(q_values, epsilon, num_actions)
            
            session.memory_buffer.append(Experience(
                session.state, action, reward, obs, done))

            if done:
                session.env.reset()
                session.agent_iter = iter(session.env.agent_iter())
                session.current_agent = next(session.agent_iter)
                socketio.emit('episode_end', {
                    'message': 'Episode ended'}, room=sid)
                break

            session.env.step(action)
            session.current_agent = next(session.agent_iter)

            frame = render_frame(session)
            socketio.emit('frame', frame, room=sid)
            session.state = obs
        except Exception as e:
            print("Error emitting frame:", e)

        socketio.sleep(input_timeout)


@app.route('/preconnect', methods=['GET'])
def preconnect():
    environments_list = list(environments.keys())
    ai_players = ["regular", "atari_pro"]

    return jsonify({
        "environments": environments_list,
        "ai_players": ai_players
    })


@socketio.on('connect')
def on_connect():
    global num_actions
    sid = request.sid
    env_name = request.args.get("env")
    ai_player = request.args.get("ai_player")
    print(
        f"Client {sid} connected with env={env_name}, againg player_id={ai_player}")

    env_config: EnvironmentConfig = environments[env_name]
    env = env_config.env()

    if ai_player == 'atari_pro':
        env_config.model_path = atari_pro
        env_config.weights_path = atari_pro_weights
    
    env.reset()
    agent_iter = iter(env.agent_iter())
    current_agent = next(agent_iter)
    state, _, _, _, _ = env.last()

    with global_lock:
        if env_config.q_network is None:
            num_actions = env_config.num_actions
            if os.path.exists(env_config.model_path) and os.path.exists(env_config.weights_path):
                print("Loading existing models...")
                env_config.q_network = tf.keras.models.load_model(env_config.model_path)
                env_config.target_q_network = tf.keras.models.load_model(
                    env_config.weights_path)
                env_config.optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)
            else:
                print("Initializing new models...")
                obs_shape = env_config.observation_space
                env_config.q_network = build_q_network(obs_shape, num_actions)
                env_config.target_q_network = build_q_network(obs_shape, num_actions)
                env_config.optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)
                dummy_input = tf.zeros((1, *obs_shape), dtype=tf.float32)
                _ = env_config.q_network(dummy_input)
                _ = env_config.target_q_network(dummy_input)
                env_config.target_q_network.set_weights(env_config.q_network.get_weights())
            zero_grads = [tf.zeros_like(v) for v in env_config.q_network.trainable_variables]
            env_config.optimizer.apply_gradients(zip(zero_grads, env_config.q_network.trainable_variables))

    thread_stop_event = threading.Event()

    session_state = SessionState(
        env_config=env_config,
        env=env,
        state=state,
        thread_stop_event=thread_stop_event,
    )
    session_state.agent_iter = agent_iter
    session_state.current_agent = current_agent

    client_sessions[sid] = session_state
    thread = threading.Thread(target=frame_emit_loop, args=(sid,))
    thread.daemon = True
    session_state.thread = thread
    thread.start()


@socketio.on('input')
def on_input(keys: list[str]):
    sid = request.sid
    session = client_sessions.get(sid)
    if session is None:
        return

    with session.lock:
        for key in keys:
            action = session.env_config.KEY_MAP.get(key)
            if action is not None:
                session.action_queue.append(action)
            else:
                session.action_queue.append(0)


@socketio.on('disconnect')
def on_disconnect():
    global epsilon
    sid = request.sid
    print(f"Client {sid} disconnected")

    session: SessionState = client_sessions.pop(sid, None)
    if session:
        session.thread_stop_event.set()
        if session.thread:
            session.thread.join(timeout=1.0)
            session.thread = None
        try:
            session.env.close()
        except Exception as e:
            print(f"Error closing env for session {sid}: {e}")

        print(f"Session {sid} cleaned up.")
    try:
        with global_lock:
            print(f"Teaching model after {sid}. Nemesis scored: {session.nemesis_total_reward}")
            for _ in range(10):
                train_step(session)

            epsilon = utils.get_new_eps(epsilon)
            session.env_config.q_network.save(session.env_config.model_path)
            session.env_config.target_q_network.save(session.env_config.weights_path)
    except Exception as e:
        print(f"Error updating model for session {sid}: {e}")


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
