from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Deque, Optional
from collections import deque, namedtuple

import eventlet
from app.config.ml_env_config import EnvironmentConfig
import tensorflow as tf

if TYPE_CHECKING:
    from app.services.session_runner import SessionRunner

Experience = namedtuple("Experience", field_names=[ "env_name",
                        "state", "action", "reward", "next_state", "done", "timestamp"])

@dataclass
class SessionState:
    env_config: EnvironmentConfig
    env: object
    state: object
    user: dict = None
    agent_iter: Optional[object] = None
    current_agent: Optional[object] = None
    runner: Optional["SessionRunner"] = None
    q_network: Optional[tf.keras.Model] = None
    target_q_network: Optional[tf.keras.Model] = None
    nemesis_total_reward: int = field(default=0)
    current_action: int = 0
    lock: eventlet.semaphore.Semaphore = field(default_factory=eventlet.semaphore.Semaphore)
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None