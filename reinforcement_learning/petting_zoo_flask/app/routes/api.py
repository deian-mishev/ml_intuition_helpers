import os
from flask_socketio import disconnect
import tensorflow as tf
from flask import request, jsonify, session

from app import app, socketio, global_lock, client_sessions, client_sessions_lock
from app.services.ml_service import ml_service
from app.services.session_runner import SessionRunner
from app.config.session_state import SessionState
from app.config.ml_env_config import EnvironmentConfig, ENVIRONMENTS
from app.config.env_config import *
from app.config.oauth2_config import login_required, roles_required
from flask import render_template

AI_PLAYERS = ["regular", "atari_pro"]


@app.route('/preconnect', methods=['GET'])
@login_required
@roles_required('User')
def preconnect():
    environments_list = list(ENVIRONMENTS.keys())

    return jsonify({
        "environments": environments_list,
        "ai_players": AI_PLAYERS
    })


@socketio.on('connect')
def on_connect():
    user = session.get('user')
    roles = session.get('roles', [])

    if not user or 'User' not in roles:
        disconnect()
        return

    sid = request.sid
    env_name = request.args.get("env")
    ai_player = request.args.get("ai_player")
    if env_name not in ENVIRONMENTS or ai_player not in AI_PLAYERS:
        disconnect()
        return

    app.logger.info(
        f"{sid}: {user['name']} connected with roles {roles} in env={env_name}, against '{ai_player}'")

    env_config: EnvironmentConfig = ENVIRONMENTS[env_name]
    env = env_config.env()

    if ai_player == 'atari_pro':
        env_config.model_path = ATARI_PRO
        env_config.weights_path = ATARI_PRO_WEIGHTS

    env.reset()
    agent_iter = iter(env.agent_iter())
    current_agent = next(agent_iter)
    state, _, _, _, _ = env.last()

    with global_lock:
        if env_config.q_network is None:
            num_actions = env_config.num_actions
            if os.path.exists(env_config.model_path) and os.path.exists(env_config.weights_path):
                app.logger.info(
                    f"{sid}: Loading existing models ${env_config.model_path}")
                env_config.q_network = tf.keras.models.load_model(
                    env_config.model_path)
                env_config.target_q_network = tf.keras.models.load_model(
                    env_config.weights_path)
                env_config.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=ALPHA)
            else:
                app.logger.info(f"{sid}: Initializing new models...")
                obs_shape = env_config.observation_space
                env_config.q_network = ml_service.build_q_network(
                    obs_shape, num_actions)
                env_config.target_q_network = ml_service.build_q_network(
                    obs_shape, num_actions)
                env_config.optimizer = tf.keras.optimizers.Adam(
                    learning_rate=ALPHA)
                dummy_input = tf.zeros((1, *obs_shape), dtype=tf.float32)
                _ = env_config.q_network(dummy_input)
                _ = env_config.target_q_network(dummy_input)
                env_config.target_q_network.set_weights(
                    env_config.q_network.get_weights())
            zero_grads = [tf.zeros_like(
                v) for v in env_config.q_network.trainable_variables]
            env_config.optimizer.apply_gradients(
                zip(zero_grads, env_config.q_network.trainable_variables))

    session_state = SessionState(
        user=user,
        env_config=env_config,
        env=env,
        state=state
    )
    session_state.agent_iter = agent_iter
    session_state.current_agent = current_agent

    with client_sessions_lock:
        client_sessions[sid] = session_state

    session_state.runner = SessionRunner(
        sid, session_state, socketio, global_lock)
    session_state.runner.start()


@socketio.on('input')
def on_input(keys: list[str]):
    sid = request.sid
    with client_sessions_lock:
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
    sid = request.sid
    app.logger.info(f"{sid}: User disconnected...")
    with client_sessions_lock:
        session = client_sessions.pop(sid, None)

    if session:
        if session.runner:
            session.runner.stop()
        try:
            session.env.close()
        except Exception as e:
            app.logger.error(f"{sid}: Error closing env for session: {e}")

        app.logger.info(f"{sid}: Session cleaned up.")
    try:
        with global_lock:
            app.logger.info(
                f"{sid}: Teaching model, nemesis scored: {session.nemesis_total_reward}")
            for _ in range(10):
                ml_service.train_step(session)

            session.env_config.epsilon = ml_service.get_new_eps(
                session.env_config.epsilon)
            session.env_config.q_network.save(session.env_config.model_path)
            session.env_config.target_q_network.save(
                session.env_config.weights_path)
    except Exception as e:
        app.logger.error(f"{sid}: Error updating model for session: {e}")


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
@login_required
def index(path):
    return render_template("index.html", width=AGENT_VID_WIDTH, height=AGENT_VID_HEIGHT)
