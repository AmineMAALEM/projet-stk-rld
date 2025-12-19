import gymnasium as gym
from bbrl.agents import Agent
import torch
import torch.nn as nn

class Actor(Agent):
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        # --- 1. Define Sizes ---
        # The Network expects 616 (4 frames x 154)
        self.net_input_size = 616
        self.num_stack = 4
        
        # Calculate the size of ONE frame
        # If we divide 616 by 4, we get 154.
        self.single_frame_size = self.net_input_size // self.num_stack

        # --- 2. Internal Memory for Stacking ---
        # We store the history of frames here
        self.history = None 

        # --- 3. Output Size ---
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

        # --- 4. Normalization Buffers ---
        # Loaded from pystk_actor.pth (Size 616)
        self.register_buffer("obs_mean", torch.zeros(self.net_input_size))
        self.register_buffer("obs_var", torch.ones(self.net_input_size))
        self.epsilon = 1e-8

        # --- 5. The Network ---
        self.net = nn.Sequential(
            nn.Linear(self.net_input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_size),
        )

    def forward(self, t: int, **kwargs):
        # 1. Get the current single frame from the environment
        # Expected size: (Batch, 154) or just (154)
        obs = self.get(("env/env_obs", t))
        
        # Ensure it's a tensor
        if not torch.is_tensor(obs):
            obs = torch.tensor(obs)
            
        # Handle Batch Dimension: 
        # If input is 1D (154,), make it 2D (1, 154)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            
        batch_size = obs.shape[0]

        # 2. Manage History (Frame Stacking)
        # If t=0 or history is empty, initialize it by repeating the first frame
        if t == 0 or self.history is None or self.history.shape[0] != batch_size:
            # Create stack: (Batch, 4, 154)
            self.history = obs.unsqueeze(1).repeat(1, self.num_stack, 1)
        else:
            # Shift history: Drop oldest, add newest
            # self.history is (Batch, 4, 154)
            # Roll to the left
            self.history = torch.roll(self.history, shifts=-1, dims=1)
            # Update last frame
            self.history[:, -1, :] = obs

        # 3. Flatten History
        # Transform (Batch, 4, 154) -> (Batch, 616)
        obs_stacked = self.history.view(batch_size, -1)
        
        # Verify device (ensure history is on same device as weights)
        if obs_stacked.device != self.obs_mean.device:
            obs_stacked = obs_stacked.to(self.obs_mean.device)

        # 4. Normalize (Using the 616 stats)
        obs_norm = (obs_stacked - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon)

        # 5. Predict
        scores = self.net(obs_norm)
        
        # 6. Action Logic
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

# Keep dummy classes to avoid import errors
class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int): super().__init__(env)
    def action(self, action): return action
class ArgmaxActor(Agent):
    def forward(self, t): pass
class SamplingActor(Agent):
    def __init__(self, action_space): super().__init__()
    def forward(self, t): pass