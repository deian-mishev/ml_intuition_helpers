from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional
import threading
from collections import deque, namedtuple
from app.config.env_config import MEMORY_SIZE
from app.config.ml_env_config import EnvironmentConfig

if TYPE_CHECKING:
    from app.services.session_runner import SessionRunner


@dataclass
class SessionState:
    env_config: EnvironmentConfig
    env: object
    state: object
    user: dict = None
    agent_iter: Optional[object] = None
    current_agent: Optional[object] = None
    runner: Optional["SessionRunner"] = None
    nemesis_total_reward: int = field(default=0)
    action_queue: deque = field(default_factory=deque)
    lock: threading.Lock = field(default_factory=threading.Lock)
    memory_buffer: deque = field(
        default_factory=lambda: deque(maxlen=MEMORY_SIZE))


Experience = namedtuple("Experience", field_names=[
                        "state", "action", "reward", "next_state", "done"])
