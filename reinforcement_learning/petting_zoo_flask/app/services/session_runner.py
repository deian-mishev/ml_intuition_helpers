import threading
import tensorflow as tf
from flask_socketio import SocketIO
from app.config.session_state import SessionState, Experience
from app.services.rendering_service import render_frame
from app.services.ml_service import ml_service
from app.config.scalars import HUMAN_AGENT_NAME, INPUT_TIMEOUT, EPSILON

class SessionRunner:
    def __init__(self, sid: str, session: SessionState, socketio: SocketIO, global_lock: threading.Lock):
        self.sid = sid
        self.session = session
        self.socketio = socketio
        self.global_lock = global_lock
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run_loop)
        self.thread.daemon = True

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _should_stop(self):
        return self._stop_event.is_set()

    def _run_loop(self):
        while not self._should_stop():
            try:
                obs, reward, terminated, truncated, info = self.session.env.last()
                done = terminated or truncated

                if self.session.current_agent == HUMAN_AGENT_NAME:
                    self.session.nemesis_total_reward += reward
                    with self.session.lock:
                        action = self.session.action_queue.popleft() if self.session.action_queue else 0
                else:
                    with self.global_lock:
                        q_values = self.session.env_config.q_network(tf.expand_dims(self.session.state, 0))
                    action = ml_service.get_action(q_values, EPSILON, self.session.env_config.num_actions)

                self.session.memory_buffer.append(Experience(
                    self.session.state, action, reward, obs, done))

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
                print(f"Error emitting frame for {self.sid}:", e)

            self.socketio.sleep(INPUT_TIMEOUT)
