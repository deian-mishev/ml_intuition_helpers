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
from pettingzoo.atari import boxing_v2, space_invaders_v2, tennis_v3
from flask_socketio import SocketIO, emit, disconnect
from flask import Flask, jsonify, request
from typing import Callable, Optional

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
global_lock = threading.Lock()
optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)

client_sessions = {}

q_network = None
target_q_network = None
num_actions = 0

epsilon = 1.0
input_timeout = 0.01


@dataclass
class EnvironmentConfig:
    KEY_MAP: dict
    model_path: str
    weights_path: str
    env: Optional[Callable] = field(default=None)


atari_pro = "./data/atari_pro.keras"
atari_pro_weights = "./data/atari_pro_weights.keras"

environments = {
    'boxing_v2_config': EnvironmentConfig(
        KEY_MAP={
            "0": 0, "s": 1, "8": 2, "6": 3, "4": 4, "2": 5,
            "9": 6, "7": 7, "3": 8, "1": 9, "w": 10, "d": 11,
            "a": 12, "x": 13, "e": 14, "q": 15, "c": 16, "z": 17
        },
        model_path="./data/boxing_model.keras",
        weights_path="./data/boxing_weights.keras",
        env=lambda render_mode="rgb_array": boxing_v2.env(
            render_mode=render_mode)
    ),
    "space_invaders_v2": EnvironmentConfig(
        KEY_MAP={
            "8": 2, "6": 4, "4": 3, "2": 5, "s": 1
        },
        model_path="./data/space_invaders_v2.keras",
        weights_path="./data/space_invaders_v2_weights.keras",
        env=lambda render_mode="rgb_array": space_invaders_v2.env(
            render_mode=render_mode)
    ),
    "tennis_v3": EnvironmentConfig(
        KEY_MAP={
            "8": 2, "6": 3, "4": 4,
            "2": 5, "9": 6, "7": 7,
            "3": 8, "1": 9, "w": 10,
            "d": 11, "a": 12, "x": 13,
            "e": 14, "q": 15, "c": 16,
            "z": 17, "s": 1
        },
        model_path="./data/tennis_v3.keras",
        weights_path="./data/tennis_v3_weights.keras",
        env=lambda render_mode="rgb_array": tennis_v3.env(
            render_mode=render_mode)
    )
}


@dataclass
class SessionState:
    env_config: EnvironmentConfig
    env: object
    agent_iter: object
    current_agent: object
    state: object
    thread: threading.Thread = None
    thread_stop_event: threading.Event = None
    action_queue: deque = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)
    memory_buffer: deque = field(
        default_factory=lambda: deque(maxlen=MEMORY_SIZE))


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
    utils.agent_learn(experiences, GAMMA, target_q_network,
                      optimizer, q_network)


def render_frame(env):
    frame = env.render()
    img = Image.fromarray(frame)
    buf = BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def frame_emit_loop(sid):
    session = client_sessions.get(sid)
    if not session:
        print(f"No session for sid {sid}, terminating frame_emit_loop.")
        return

    while not session.thread_stop_event.is_set():
        try:
            if session.current_agent == HUMAN_AGENT_NAME:
                with session.lock:
                    if session.action_queue:
                        action = session.action_queue.popleft()
                    else:
                        action = 0
            else:
                with global_lock:
                    q_values = q_network(tf.expand_dims(session.state, 0))
                action = utils.get_action(q_values, epsilon, num_actions)

            obs, reward, terminated, truncated, info = session.env.last()
            done = terminated or truncated

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

            frame = render_frame(session.env)
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
    global num_actions, q_network, target_q_network
    sid = request.sid
    env_name = request.args.get("env")
    ai_player = request.args.get("ai_player")
    print(
        f"Client {sid} connected with env={env_name}, againg player_id={ai_player}")

    env_config: EnvironmentConfig = environments[env_name]
    env = env_config.env()

    if ai_player == 'atari_pro':
        env_config.model_path = environments['atari_pro']
        env_config.weights_path = environments['atari_pro_weights']

    env.reset()
    agent_iter = iter(env.agent_iter())
    current_agent = next(agent_iter)
    state, _, _, _, _ = env.last()

    if q_network is None:
        with global_lock:
            num_actions = env.action_space(current_agent).n
            if os.path.exists(env_config.model_path) and os.path.exists(env_config.weights_path):
                print("Loading existing models...")
                q_network = tf.keras.models.load_model(env_config.model_path)
                target_q_network = tf.keras.models.load_model(
                    env_config.weights_path)
            else:
                print("Initializing new models...")
                obs_shape = env.observation_space(current_agent).shape
                q_network = build_q_network(obs_shape, num_actions)
                target_q_network = build_q_network(obs_shape, num_actions)
                dummy_input = tf.zeros((1, *obs_shape), dtype=tf.float32)
                _ = q_network(dummy_input)
                _ = target_q_network(dummy_input)
                target_q_network.set_weights(q_network.get_weights())

    thread_stop_event = threading.Event()

    session_state = SessionState(
        env=env,
        agent_iter=agent_iter,
        current_agent=current_agent,
        state=state,
        thread_stop_event=thread_stop_event,
        env_config=env_config
    )
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

    session = client_sessions.pop(sid, None)
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
            print(f"Teaching model after {sid}.")
            for _ in range(10):
                train_step(session)

            epsilon = utils.get_new_eps(epsilon)
            q_network.save(session.env_config.model_path)
            target_q_network.save(session.env_config.weights_path)
    except Exception as e:
        print(f"Error updating model for session {sid}: {e}")


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
