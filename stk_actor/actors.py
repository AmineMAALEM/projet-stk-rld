import gymnasium as gym
from bbrl.agents import Agent
import torch
import torch.nn as nn

class Actor(Agent):
    def __init__(self, observation_space, action_space):
        super().__init__()
        # 616 entrées (4 images de 154 pixels)
        self.net_input_size = 616
        self.num_stack = 4
        self.history = None 

        if isinstance(action_space, gym.spaces.MultiDiscrete):
            self.output_dims = action_space.nvec.tolist()
            output_size = int(sum(self.output_dims))
            self.is_multi = True
        else:
            output_size = 4
            self.is_multi = False

        self.register_buffer("obs_mean", torch.zeros(self.net_input_size))
        self.register_buffer("obs_var", torch.ones(self.net_input_size))
        self.epsilon = 1e-8

        self.net = nn.Sequential(
            nn.Linear(self.net_input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_size),
        )

    def forward(self, t: int, **kwargs):
        # 1. Input (154,)
        obs = self.get(("env/env_obs", t))
        if not torch.is_tensor(obs): obs = torch.tensor(obs)
        if obs.dim() == 1: obs = obs.unsqueeze(0)
        
        # 2. Stacking Manuel
        batch_size = obs.shape[0]
        if t == 0 or self.history is None or self.history.shape[0] != batch_size:
            self.history = obs.unsqueeze(1).repeat(1, self.num_stack, 1)
        else:
            self.history = torch.roll(self.history, shifts=-1, dims=1)
            self.history[:, -1, :] = obs

        # 3. Aplatir vers 616
        obs_stacked = self.history.view(batch_size, -1)
        if obs_stacked.device != self.obs_mean.device:
            obs_stacked = obs_stacked.to(self.obs_mean.device)

        # 4. Normaliser
        obs_norm = (obs_stacked - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon)
        
        # 5. Prédire
        scores = self.net(obs_norm)
        
        if self.is_multi:
            pointer = 0
            actions = []
            for dim in self.output_dims:
                actions.append(scores[:, pointer : pointer + dim].argmax(dim=1))
                pointer += dim
            final_action = torch.stack(actions, dim=1)
        else:
            final_action = scores.argmax(dim=1)

        self.set(("action", t), final_action)