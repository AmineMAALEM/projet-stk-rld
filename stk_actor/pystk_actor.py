from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import torch

from .actors import Actor

#: The base environment name
env_name = "supertuxkart/flattened_multidiscrete-v0"
#: Player name
player_name = "PPO_Master"

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    # We only flatten. We will handle the Stacking (4 frames) inside the Actor.
    return [
        lambda env: FlattenObservation(env)
    ]

def get_actor(
    state: dict | None,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
) -> Agent:
    actor = Actor(observation_space, action_space)
    if state is not None:
        actor.load_state_dict(state, strict=False)
    return actor