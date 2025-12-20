from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym
import torch
import numpy as np

# Imports relatifs
from .actors import Actor
# On importe les NOUVEAUX noms définis dans wrappers.py
from .wrappers import (
    FeatureEngineeringWrapper, 
    ActionConversionWrapper, 
    FlattenWrapper, 
    FrameStackingWrapper
)

# On utilise simple-v0 (dictionnaire)
env_name = "supertuxkart/simple-v0"

player_name = "Vroom Vroom"

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    return [
        # 1. Features
        lambda env: FeatureEngineeringWrapper(env),
        # 2. Action (Conversion)
        lambda env: ActionConversionWrapper(env),
        # 3. Flatten (Avec Whitelist)
        lambda env: FlattenWrapper(env),
        # 4. Stacking
        lambda env: FrameStackingWrapper(env, n_stack=4),
    ]

def get_actor(
    state: dict | None,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
) -> Agent:
    
    actor = Actor(observation_space, action_space)

    if state is None:
        return actor

    # 1. Chargement des Poids
    if "model_state_dict" in state:
        actor.load_state_dict(state["model_state_dict"])
    else:
        actor.load_state_dict(state)
        
    # 2. Chargement des Stats Normalisation (Vers l'acteur directement)
    if "norm_mean" in state:
        mean = state["norm_mean"]
        var = state["norm_var"]
        
        # Sécurité : Conversion Numpy -> Tensor si nécessaire
        if not isinstance(mean, torch.Tensor):
            mean = torch.as_tensor(mean).float()
        if not isinstance(var, torch.Tensor):
            var = torch.as_tensor(var).float()
            
        actor.set_normalization_stats(
            mean=mean,
            var=var,
            epsilon=state.get("norm_epsilon", 1e-8),
            clip=state.get("norm_clip", 10.0)
        )

    return actor