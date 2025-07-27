import math

from app import app, client_sessions, client_sessions_lock
from app.services.ml_service import ml_service
from app.services.experience_store import experience_service
from app.config.session_state import PlayerType
from app.config.ml_env_config import ENVIRONMENTS


def cleanup_session(sid, stop_greenlet: bool = True):
    with client_sessions_lock:
        session = client_sessions.pop(sid, None)

    if not session:
        app.logger.info(f"{sid}: No session found to clean up.")
        return

    if stop_greenlet and session.runner:
        session.runner.stop()

    try:
        session.env.close()
    except Exception as e:
        app.logger.error(f"{sid}: Error closing env for session: {e}")

    app.logger.info(f"{sid}: Session cleaned up.")

    best_score_agent = None
    best_score = -math.inf
    for agent_name, agent_state in session.agents.items():
        if best_score < agent_state.total_reward:
            best_score_agent = agent_state
            best_score = agent_state.total_reward
        app.logger.info(
            f"{sid}: {agent_name.capitalize()} total score: {agent_state.total_reward}")
        # socketio.start_background_task(
        #     experience_service.insert_experience_batch,
        #     env_name=session.env_config.name,
        #     experiences=list(best_score_agent.memory_buffer),
        #     sid=sid
        # )
    try:
        with session.env_config.lock:
            app.logger.info(
                f"{sid}: Teaching model, best agent scored: {best_score_agent.total_reward}")
            for agent_name, agent_state in session.agents.items():
                for _ in range(10):
                    ml_service.train_step(session, agent_state)

            session.env_config.epsilon = ml_service.get_new_eps(
                session.env_config.epsilon)
            session.q_network.save(session.env_config.model_path)
            session.target_q_network.save_weights(
                session.env_config.weights_path)
    except Exception as e:
        app.logger.error(f"{sid}: Error updating model for session: {e}")


def get_available_environments_and_nemesis():
    return [{"name": name, "agents": cfg.agents} for name, cfg in ENVIRONMENTS.items()], [pt.value for pt in PlayerType]
