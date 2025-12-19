import gymnasium as gym
from bbrl.agents import Agent
import torch
import torch.nn as nn

class Actor(Agent):
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        # 1. Input Size
        # With the wrappers in pystk_actor, this should be 4 * original_size
        if isinstance(observation_space, gym.spaces.Box):
            input_size = observation_space.shape[0]
        else:
            input_size = 10 # Fallback

        # 2. Output Size (Actions)
        if isinstance(action_space, gym.spaces.MultiDiscrete):
            self.nvec = action_space.nvec
            self.output_dims = self.nvec.tolist()
            output_size = int(sum(self.output_dims))
            self.is_multi = True
        elif isinstance(action_space, gym.spaces.Discrete):
            output_size = action_space.n
            self.is_multi = False
        else:
            output_size = 4
            self.is_multi = False

        # 3. Normalization Buffers
        # The .pth file will fill these with the values from VecNormalize
        self.register_buffer("obs_mean", torch.zeros(input_size))
        self.register_buffer("obs_var", torch.ones(input_size))
        self.epsilon = 1e-8

        # 4. The Network (Matches SB3 PPO MlpPolicy)
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_size),
        )

    def forward(self, t: int, **kwargs):
        # Get stacked observation
        obs = self.get(("env/env_obs", t))
        
        # Normalize
        # (obs - mean) / sqrt(var + eps)
        # Note: self.obs_mean is loaded from the .pth file
        obs_norm = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon)

        # Forward Pass
        scores = self.net(obs_norm)
        
        # Action Selection
        if self.is_multi:
            pointer = 0
            actions = []
            for dim in self.output_dims:
                sub_scores = scores[:, pointer : pointer + dim]
                sub_action = sub_scores.argmax(dim=1)
                actions.append(sub_action)
                pointer += dim
            final_action = torch.stack(actions, dim=1)
        else:
            final_action = scores.argmax(dim=1)

        self.set(("action", t), final_action)

# Dummy classes
class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int): super().__init__(env)
    def action(self, action): return action
class ArgmaxActor(Agent):
    def forward(self, t): pass
class SamplingActor(Agent):
    def __init__(self, action_space): super().__init__()
    def forward(self, t): pass