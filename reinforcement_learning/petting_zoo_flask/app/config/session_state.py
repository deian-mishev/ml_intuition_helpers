from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Deque, Dict, Optional
from collections import deque

import eventlet
from app.config.env_config import MEMORY_SIZE
from app.config.ml_env_config import EnvironmentConfig
import tensorflow as tf

if TYPE_CHECKING:
    from app.services.session_runner import SessionRunner

@dataclass
class Experience:
    state: any
    action: int = None
    reward: float = None
    next_state: any = None
    done: bool = None

class PlayerType(str, Enum):
    COMPUTER = "computer"
    HUMAN = "human"
  
@dataclass
class PlayerState:
    type: PlayerType = None
    current_experience: Optional[Experience] = None
    total_reward: int = field(default=0)
    memory_buffer: Deque[Experience] = field(default_factory=lambda: deque(maxlen=MEMORY_SIZE))

@dataclass
class SessionState:
    env: object
    env_config: EnvironmentConfig
    runner: Optional["SessionRunner"] = None
    q_network: Optional[tf.keras.Model] = None
    target_q_network: Optional[tf.keras.Model] = None
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None
    agent_iter: Optional[object] = None
    current_agent: Optional[PlayerState] = None
    agents: Dict[str, PlayerState] = field(default_factory=dict)
    next_human_action: int = 0
    lock: eventlet.semaphore.Semaphore = field(default_factory=eventlet.semaphore.Semaphore)
