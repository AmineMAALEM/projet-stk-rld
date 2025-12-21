import gymnasium as gym
from bbrl.agents import Agent
import torch
import torch.nn as nn

class Actor(Agent):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__()
        obs_dim = 448 
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_var", torch.ones(obs_dim))
        self.epsilon = 1e-8
        self.clip_obs = 10.0
        self.printed = False

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.action_net = nn.Linear(64, 15)

    def forward(self, t: int):
        obs = self.get(("env/env_obs", t))
        
        # DEBUG pour voir ce qu'on reçoit
        if not self.printed:
            print(f"[ACTOR] Obs type: {type(obs)}")
            if isinstance(obs, torch.Tensor):
                print(f"[ACTOR] Obs shape: {obs.shape}")
            self.printed = True

        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, device=self.obs_mean.device, dtype=torch.float32)
        
        if obs.ndim == 1: obs = obs.unsqueeze(0)
        elif obs.ndim > 2: obs = obs.reshape(-1, 448)

        obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon)
        obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)
        
        logits = self.action_net(self.net(obs))
        action = torch.argmax(logits, dim=1)
        self.set(("action", t), action)

    def load_state_dict(self, state_dict):
        new_sd = {}
        for k, v in state_dict.items():
            if "mlp_extractor.policy_net.0" in k: new_sd["net.0.weight" if "weight" in k else "net.0.bias"] = v
            elif "mlp_extractor.policy_net.2" in k: new_sd["net.2.weight" if "weight" in k else "net.2.bias"] = v
            elif "action_net" in k: new_sd[k] = v
        super().load_state_dict(new_sd, strict=False)

    def set_normalization_stats(self, mean, var, epsilon, clip):
        self.obs_mean.copy_(mean[:448])
        self.obs_var.copy_(var[:448])
        self.epsilon, self.clip_obs = epsilon, clip
# import gymnasium as gym
# from bbrl.agents import Agent
# import torch
# import torch.nn as nn

# class Actor(Agent):
#     def __init__(self, observation_space: gym.Space, action_space: gym.Space):
#         super().__init__()
        
#         obs_dim = 448 

#         self.register_buffer("obs_mean", torch.zeros(obs_dim))
#         self.register_buffer("obs_var", torch.ones(obs_dim))
#         self.epsilon = 1e-8
#         self.clip_obs = 10.0

#         self.net = nn.Sequential(
#             nn.Linear(obs_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh()
#         )
#         self.action_net = nn.Linear(64, 15)

#     def forward(self, t: int):
#         obs = self.get(("env/env_obs", t))
#         if not isinstance(obs, torch.Tensor):
#             obs = torch.tensor(obs, device=self.obs_mean.device, dtype=torch.float32)
        
#         # --- PATCH ULTIME ---
#         # Si malgré tout on reçoit 112, on duplique pour faire 448
#         if obs.shape[-1] == 112:
#             print("⚠️ WARNING: Received 112 dims. Stacking manually in Actor.")
#             obs = obs.repeat(1, 4) if obs.ndim > 1 else obs.repeat(4)
#         # --------------------

#         if obs.ndim == 1: obs = obs.unsqueeze(0)
#         elif obs.ndim > 2: obs = obs.reshape(-1, 448)

#         obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon)
#         obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)
        
#         logits = self.action_net(self.net(obs))
#         action = torch.argmax(logits, dim=1)
        
#         self.set(("action", t), action)

#     def load_state_dict(self, state_dict):
#         new_sd = {}
#         for k, v in state_dict.items():
#             if "mlp_extractor.policy_net.0" in k: new_sd["net.0.weight" if "weight" in k else "net.0.bias"] = v
#             elif "mlp_extractor.policy_net.2" in k: new_sd["net.2.weight" if "weight" in k else "net.2.bias"] = v
#             elif "action_net" in k: new_sd[k] = v
#         super().load_state_dict(new_sd, strict=False)

#     def set_normalization_stats(self, mean, var, epsilon, clip):
#         self.obs_mean.copy_(mean[:448])
#         self.obs_var.copy_(var[:448])
#         self.epsilon, self.clip_obs = epsilon, clip
# # import gymnasium as gym
# # from bbrl.agents import Agent
# # import torch
# # import torch.nn as nn

# # class Actor(Agent):
# #     def __init__(self, observation_space: gym.Space, action_space: gym.Space):
# #         super().__init__()
        
# #         obs_dim = 448 

# #         self.register_buffer("obs_mean", torch.zeros(obs_dim))
# #         self.register_buffer("obs_var", torch.ones(obs_dim))
# #         self.epsilon = 1e-8
# #         self.clip_obs = 10.0

# #         self.net = nn.Sequential(
# #             nn.Linear(obs_dim, 64),
# #             nn.Tanh(),
# #             nn.Linear(64, 64),
# #             nn.Tanh()
# #         )
        
# #         action_dim = 15
# #         if hasattr(action_space, 'n'):
# #             action_dim = action_space.n
            
# #         self.action_net = nn.Linear(64, action_dim)

# #     def forward(self, t: int):
# #         obs = self.get(("env/env_obs", t))
        
# #         if not isinstance(obs, torch.Tensor):
# #             obs = torch.tensor(obs, device=self.obs_mean.device, dtype=torch.float32)
        
# #         norm_obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon)
# #         norm_obs = torch.clamp(norm_obs, -self.clip_obs, self.clip_obs)
        
# #         features = self.net(norm_obs)
# #         action_logits = self.action_net(features)
# #         action = torch.argmax(action_logits, dim=1)
        
# #         self.set(("action", t), action)

# #     def load_state_dict(self, state_dict):
# #         new_state_dict = {}
# #         for key, value in state_dict.items():
# #             if "mlp_extractor.policy_net.0" in key:
# #                 new_key = key.replace("mlp_extractor.policy_net.0", "net.0")
# #             elif "mlp_extractor.policy_net.2" in key:
# #                 new_key = key.replace("mlp_extractor.policy_net.2", "net.2")
# #             elif "action_net" in key:
# #                 new_key = key
# #             else:
# #                 continue
# #             new_state_dict[new_key] = value
# #         super().load_state_dict(new_state_dict, strict=False)

# #     def set_normalization_stats(self, mean, var, epsilon, clip):
# #         limit = min(mean.shape[0], self.obs_mean.shape[0])
# #         self.obs_mean[:limit] = mean[:limit]
# #         self.obs_var[:limit] = var[:limit]
# #         self.epsilon = epsilon
# #         self.clip_obs = clip
# # # import gymnasium as gym
# # # from bbrl.agents import Agent
# # # import torch
# # # import torch.nn as nn

# # # class Actor(Agent):
# # #     def __init__(self, observation_space: gym.Space, action_space: gym.Space):
# # #         super().__init__()
        
# # #         # On attend 448 (112 * 4)
# # #         obs_dim = 448 

# # #         # Buffers Normalisation
# # #         self.register_buffer("obs_mean", torch.zeros(obs_dim))
# # #         self.register_buffer("obs_var", torch.ones(obs_dim))
# # #         self.epsilon = 1e-8
# # #         self.clip_obs = 10.0

# # #         # Réseau
# # #         self.net = nn.Sequential(
# # #             nn.Linear(obs_dim, 64),
# # #             nn.Tanh(),
# # #             nn.Linear(64, 64),
# # #             nn.Tanh()
# # #         )
        
# # #         action_dim = 15
# # #         if hasattr(action_space, 'n'):
# # #             action_dim = action_space.n
            
# # #         self.action_net = nn.Linear(64, action_dim)

# # #     def forward(self, t: int):
# # #         obs = self.get(("env/env_obs", t))
        
# # #         if not isinstance(obs, torch.Tensor):
# # #             obs = torch.tensor(obs, device=self.obs_mean.device, dtype=torch.float32)
        
# # #         # Important : Gestion du Batch dimension (B, 448) vs (448)
# # #         # Si l'obs arrive sans dimension de batch, on unsqueeze ?
# # #         # BBRL gère généralement le batch, mais au cas où.
        
# # #         # Normalisation
# # #         norm_obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon)
# # #         norm_obs = torch.clamp(norm_obs, -self.clip_obs, self.clip_obs)
        
# # #         features = self.net(norm_obs)
# # #         action_logits = self.action_net(features)
# # #         action = torch.argmax(action_logits, dim=1)
        
# # #         self.set(("action", t), action)

# # #     def load_state_dict(self, state_dict):
# # #         new_state_dict = {}
# # #         for key, value in state_dict.items():
# # #             if "mlp_extractor.policy_net.0" in key:
# # #                 new_key = key.replace("mlp_extractor.policy_net.0", "net.0")
# # #             elif "mlp_extractor.policy_net.2" in key:
# # #                 new_key = key.replace("mlp_extractor.policy_net.2", "net.2")
# # #             elif "action_net" in key:
# # #                 new_key = key
# # #             else:
# # #                 continue
# # #             new_state_dict[new_key] = value
# # #         super().load_state_dict(new_state_dict, strict=False)

# # #     def set_normalization_stats(self, mean, var, epsilon, clip):
# # #         # On copie en s'assurant de la taille
# # #         # Le mean venant du .pth devrait faire 448 si exporté après un training local
# # #         limit = min(mean.shape[0], self.obs_mean.shape[0])
# # #         self.obs_mean[:limit] = mean[:limit]
# # #         self.obs_var[:limit] = var[:limit]
# # #         self.epsilon = epsilon
# # #         self.clip_obs = clip