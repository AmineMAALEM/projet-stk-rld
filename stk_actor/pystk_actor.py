from bbrl.agents import Agent
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from .actors import Actor

env_name = "supertuxkart/flattened_multidiscrete-v0"
player_name = "VROOM VROOM"

def get_wrappers():
    return [lambda env: FlattenObservation(env)]

def get_actor(state, observation_space, action_space):
    actor = Actor(observation_space, action_space)
    if state is not None:
        actor.load_state_dict(state, strict=False)
    return actor