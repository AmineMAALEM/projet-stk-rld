from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, FrameStack
import torch

# Import our Actor class
from .actors import Actor

#: The base environment name
env_name = "supertuxkart/flattened_multidiscrete-v0"

#: Player name (CHANGE THIS)
player_name = "PPO_Master" 

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """
    Returns a list of wrappers to mimic the training environment.
    Training used: Flatten -> FrameStack(4) -> Flatten (implicit in SB3 MLP)
    """
    return [
        # 1. Ensure observation is flat (matches make_env in training)
        lambda env: FlattenObservation(env),
        
        # 2. Stack 4 frames (matches VecFrameStack(n_stack=4))
        # This creates an observation of shape (4, N)
        lambda env: FrameStack(env, num_stack=4),
        
        # 3. Flatten the stack into a single vector (4*N)
        # This makes it compatible with the Linear layer of the MLP
        lambda env: FlattenObservation(env)
    ]

def get_actor(
    state: dict | None,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
) -> Agent:
    # Create the Actor
    # The observation_space here will already be the Stacked+Flattened one
    # thanks to get_wrappers() being called before this.
    actor = Actor(observation_space, action_space)

    # Load weights
    if state is not None:
        actor.load_state_dict(state, strict=False)

    return actor