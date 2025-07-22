import eventlet
import tensorflow as tf
from flask_socketio import SocketIO
from app import app
from app.config.session_state import SessionState, Experience
from app.services.rendering_service import render_frame
from app.services.ml_service import ml_service
from app.config.env_config import HUMAN_AGENT_NAME, INPUT_TIMEOUT, EPSILON
from app.services.experience_store import experience_service

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
                obs, reward, terminated, truncated, info = self.session.env.last()
                done = terminated or truncated

                if self.session.current_agent == HUMAN_AGENT_NAME:
                    self.session.nemesis_total_reward += reward
                    with self.session.lock:
                        action = self.session.current_action
                else:
                    with self.session.lock:
                        q_values = self.session.q_network(tf.expand_dims(self.session.state, 0))
                    action = ml_service.get_action(q_values, EPSILON, self.session.env_config.num_actions)

                experience_service.insert_experience(
                    env_name=self.session.env_config.name,
                    experience=Experience(
                        self.session.state, action, reward, obs,
                        done)
                )

                if done:
                    self.session.env.reset()
                    self.session.agent_iter = iter(self.session.env.agent_iter())
                    self.session.current_agent = next(self.session.agent_iter)
                    self.socketio.emit('episode_end', {'message': 'Episode ended'}, room=self.sid)
                    break

                self.session.env.step(action)
                self.session.current_agent = next(self.session.agent_iter)

                frame = render_frame(self.session)
                self.socketio.emit('frame', frame, room=self.sid)
                self.session.state = obs
            except Exception as e:
                app.logger.error(f"{self.sid}: Error emitting frame: {e}")

            self.socketio.sleep(INPUT_TIMEOUT)
