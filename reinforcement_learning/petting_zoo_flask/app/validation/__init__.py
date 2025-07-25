from flask import request
from app.config.ml_env_config import ENVIRONMENTS
from app.config.session_state import PlayerType


def validate_players(env_name, allowed_types, players, human_player=PlayerType.HUMAN.value):
    if not players:
        return False, "Missing players"

    if isinstance(players, str):
        try:
            import json
            players = json.loads(players)
        except ValueError:
            return False, "Invalid players format"

    if not isinstance(players, dict):
        return False, "Players must be a dictionary"

    if env_name not in ENVIRONMENTS:
        return False, "Invalid environment"

    env_cfg = ENVIRONMENTS[env_name]

    if len(players) != len(env_cfg.agents):
        return False, "Player count mismatch"

    human_seen = False
    for agent in env_cfg.agents:
        if agent not in players:
            return False, f"Missing agent: {agent}"
        agent_type = players[agent]
        if agent_type not in allowed_types:
            return False, f"Invalid type for {agent}: {agent_type}"
        if agent_type == human_player:
            if human_seen:
                return False, "More than one human player not supported atm"
            human_seen = True

    return True, players
