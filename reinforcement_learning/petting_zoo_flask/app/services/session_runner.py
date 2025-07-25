import eventlet
import tensorflow as tf
from flask_socketio import SocketIO
from app import app
from app.config.session_state import PlayerType, SessionState, Experience
from app.services.rendering_service import render_frame
from app.services.ml_service import ml_service
from app.config.env_config import INPUT_TIMEOUT, EPSILON


class SessionRunner:
    def __init__(self, sid: str, session: SessionState, socketio: SocketIO):
        self.sid = sid
        self.session = session
        self.socketio = socketio
        self._stop_event = eventlet.event.Event()
        self._running_greenlet = None

    def start(self):
        self._running_greenlet = eventlet.spawn(self._run_loop)

    def stop(self):
        self._stop_event.send()
        if self._running_greenlet:
            self._running_greenlet.wait()

    def _should_stop(self):
        return self._stop_event.ready()

    def _run_loop(self):

        while not self._should_stop():
            try:
                with self.session.lock:
                    agent_in_turn = self.session.current_agent

                    obs, reward, terminated, truncated, _ = self.session.env.last()
                    done = terminated or truncated
                    obs = ml_service.preprocess_state(self.session.env_config.observation_space, obs)

                    if agent_in_turn.current_experience is not None:
                        agent_in_turn.current_experience.next_state = obs
                        agent_in_turn.memory_buffer.append(agent_in_turn.current_experience)

                    agent_in_turn.total_reward += reward
                    human_action = (agent_in_turn.type == PlayerType.HUMAN)
                    old_obs = obs

                if human_action:
                    action = self.session.next_human_action
                else:
                    q_values = self.session.q_network(tf.expand_dims(old_obs, 0))
                    action = ml_service.get_action(
                        q_values, EPSILON, self.session.env_config.num_actions)

                with self.session.lock:
                    agent_in_turn.current_experience = Experience(
                        state=obs,
                        action=action,
                        reward=reward,
                        done=done
                    )

                    if done:
                        for agent in self.session.agents.values():
                            if agent.current_experience:
                                agent.current_experience.next_state = obs
                                agent.memory_buffer.append(agent.current_experience)
                                agent.current_experience = None

                        self.session.env.reset()
                        self.session.agent_iter = iter(
                            self.session.env.agent_iter())
                        self.session.current_agent = self.session.agents[next(
                            self.session.agent_iter)]
                        self.socketio.emit(
                            'episode_end', {'message': 'Episode ended'}, room=self.sid)
                        return

                    self.session.env.step(action)
                    self.session.current_agent = self.session.agents[next(self.session.agent_iter)]

                frame = render_frame(self.session)
                self.socketio.emit('frame', frame, room=self.sid)

            except Exception as e:
                app.logger.error(f"{self.sid}: Error emitting frame: {e}")

            self.socketio.sleep(INPUT_TIMEOUT)
