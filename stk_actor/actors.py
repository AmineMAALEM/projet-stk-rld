import torch
import torch.nn as nn
from bbrl.agents import Agent
import gymnasium as gym

class PPOInferenceActor(Agent):
    def __init__(self, state: dict):
        super().__init__()
        
        # 1. Récupération des stats de normalisation depuis le dictionnaire
        self.register_buffer("obs_mean", state["norm_mean"])
        self.register_buffer("obs_var", state["norm_var"])
        self.register_buffer("epsilon", state["norm_epsilon"])
        self.register_buffer("clip_value", state["norm_clip"])
        
        # 2. Reconstruction de l'architecture PPO (MlpPolicy défaut SB3)
        # L'observation_space après le FrameStacking est de taille 4 x (taille après flatten)
        # Ton flatten donne environ 105 features, donc 4*105 = 420.
        # On récupère la taille exacte depuis la moyenne de normalisation.
        input_dim = self.obs_mean.shape[0]
        action_dim = 15 # D'après ton DiscreteActionWrapper
        
        # Réseau partagé (feature extractor)
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        # Tête de l'acteur (action)
        self.action_net = nn.Linear(64, action_dim) 
        
        # 3. Chargement des poids depuis le dictionnaire
        weights = state["model_state_dict"]
        
        # On charge les poids dans notre architecture PyTorch
        self.shared_net[0].load_state_dict({
            'weight': weights['mlp_extractor.policy_net.0.weight'],
            'bias': weights['mlp_extractor.policy_net.0.bias']
        })
        self.shared_net[2].load_state_dict({
            'weight': weights['mlp_extractor.policy_net.2.weight'],
            'bias': weights['mlp_extractor.policy_net.2.bias']
        })
        self.action_net.load_state_dict({
            'weight': weights['action_net.weight'],
            'bias': weights['action_net.bias']
        })

    def forward(self, t: int, **kwargs):
        # Récupère l'observation du workspace BBRL
        obs = self.get(("env/env_obs", t))
        
        # 1. Normalisation MANUELLE (comme le fait VecNormalize)
        obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon)
        obs = torch.clamp(obs, -self.clip_value, self.clip_value)
        
        # 2. Passage dans le réseau de neurones
        features = self.shared_net(obs)
        action_logits = self.action_net(features)
        
        # 3. Stocke les logits pour l'ArgmaxActor
        self.set(("action_logits", t), action_logits)

class ArgmaxActor(Agent):
    """Sélectionne l'action avec le plus grand score (logit)"""
    def __init__(self):
        super().__init__()

    def forward(self, t: int, **kwargs):
        action_logits = self.get(("action_logits", t))
        action = torch.argmax(action_logits, dim=-1)
        self.set(("action", t), action)
# import gymnasium as gym
# from bbrl.agents import Agent
# import torch
# import torch.nn as nn

# class Actor(Agent):
#     def __init__(self, observation_space: gym.Space, action_space: gym.Space):
#         super().__init__()
        
#         # On attend 448 (112 * 4)
#         obs_dim = 448 

#         # Buffers Normalisation
#         self.register_buffer("obs_mean", torch.zeros(obs_dim))
#         self.register_buffer("obs_var", torch.ones(obs_dim))
#         self.epsilon = 1e-8
#         self.clip_obs = 10.0

#         # Réseau
#         self.net = nn.Sequential(
#             nn.Linear(obs_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh()
#         )
        
#         action_dim = 15
#         if hasattr(action_space, 'n'):
#             action_dim = action_space.n
            
#         self.action_net = nn.Linear(64, action_dim)

#     def forward(self, t: int):
#         obs = self.get(("env/env_obs", t))
        
#         if not isinstance(obs, torch.Tensor):
#             obs = torch.tensor(obs, device=self.obs_mean.device, dtype=torch.float32)
        
#         # Important : Gestion du Batch dimension (B, 448) vs (448)
#         # Si l'obs arrive sans dimension de batch, on unsqueeze ?
#         # BBRL gère généralement le batch, mais au cas où.
        
#         # Normalisation
#         norm_obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon)
#         norm_obs = torch.clamp(norm_obs, -self.clip_obs, self.clip_obs)
        
#         features = self.net(norm_obs)
#         action_logits = self.action_net(features)
#         action = torch.argmax(action_logits, dim=1)
        
#         self.set(("action", t), action)

#     def load_state_dict(self, state_dict):
#         new_state_dict = {}
#         for key, value in state_dict.items():
#             if "mlp_extractor.policy_net.0" in key:
#                 new_key = key.replace("mlp_extractor.policy_net.0", "net.0")
#             elif "mlp_extractor.policy_net.2" in key:
#                 new_key = key.replace("mlp_extractor.policy_net.2", "net.2")
#             elif "action_net" in key:
#                 new_key = key
#             else:
#                 continue
#             new_state_dict[new_key] = value
#         super().load_state_dict(new_state_dict, strict=False)

#     def set_normalization_stats(self, mean, var, epsilon, clip):
#         # On copie en s'assurant de la taille
#         # Le mean venant du .pth devrait faire 448 si exporté après un training local
#         limit = min(mean.shape[0], self.obs_mean.shape[0])
#         self.obs_mean[:limit] = mean[:limit]
#         self.obs_var[:limit] = var[:limit]
#         self.epsilon = epsilon
#         self.clip_obs = clip