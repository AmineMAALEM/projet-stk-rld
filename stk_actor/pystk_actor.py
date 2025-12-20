from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym
import torch
import numpy as np

from .actors import Actor
from .wrappers import (
    FeatureEngineeringWrapper, 
    ActionConversionWrapper, # Renommé pour clarté
    FlattenWrapper,          # NOUVEAU
    FrameStackingWrapper, 
    NormalizeWrapper
)

env_name = "supertuxkart/simple-v0"
player_name = "Team_PPO_Expert"

norm_stats = {}

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    wrappers = [
        # 1. Features (ObsWrapper)
        lambda env: FeatureEngineeringWrapper(env),
        
        # 2. Action (ActionWrapper) - Uniquement conversion 0-14 -> Dict
        lambda env: ActionConversionWrapper(env),
        
        # 3. Flatten (ObsWrapper) - Uniquement Dict -> Array
        lambda env: FlattenWrapper(env),
        
        # 4. Stacking (ObsWrapper)
        lambda env: FrameStackingWrapper(env, n_stack=4),
    ]
    
    # 5. Normalisation (ObsWrapper)
    if "mean" in norm_stats:
        wrappers.append(lambda env: NormalizeWrapper(
            env, 
            mean=norm_stats["mean"], 
            var=norm_stats["var"], 
            epsilon=norm_stats["epsilon"],
            clip_obs=norm_stats.get("clip", 10.0)
        ))
    
    return wrappers

def get_actor(
    state: dict | None,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
) -> Agent:
    
    if state is None:
        return Actor(observation_space, action_space)

    if "norm_mean" in state:
        norm_stats["mean"] = state["norm_mean"]
        norm_stats["var"] = state["norm_var"]
        norm_stats["epsilon"] = state["norm_epsilon"]
        norm_stats["clip"] = state.get("norm_clip", 10.0)
    
    actor = Actor(observation_space, action_space)
    
    if "model_state_dict" in state:
        actor.load_state_dict(state["model_state_dict"])
    else:
        actor.load_state_dict(state)

    return actor
# version"1"
# from bbrl.agents import Agent
# import gymnasium as gym
# from gymnasium.wrappers import FlattenObservation
# from .actors import Actor

# env_name = "supertuxkart/flattened_multidiscrete-v0"
# player_name = "VROOM VROOM"

# def get_wrappers():
#     return [lambda env: FlattenObservation(env)]

# def get_actor(state, observation_space, action_space):
#     actor = Actor(observation_space, action_space)
#     if state is not None:
#         actor.load_state_dict(state, strict=False)
#     return actor