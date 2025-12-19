import gymnasium as gym
from bbrl.agents import Agent
import torch
import torch.nn as nn

class Actor(Agent):
    """
    Cet acteur reproduit l'architecture exacte d'un PPO (MlpPolicy) de Stable Baselines 3
    pour l'inférence (pilotage) uniquement.
    
    Architecture standard SB3 :
    Input -> [Linear(64), Tanh, Linear(64), Tanh] -> Linear(ActionDim)
    """
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__()
        
        # 1. Calcul de la dimension d'entrée
        # L'observation_space ici est celui APRÈS tous les wrappers (Stacking, Flatten, etc.)
        # Donc obs_dim sera grand (ex: ~400+)
        if isinstance(observation_space, gym.spaces.Box):
            obs_dim = observation_space.shape[0]
        else:
            raise ValueError(f"Espace d'observation non supporté : {observation_space}")

        # 2. Calcul de la dimension de sortie
        # On sécurise à 15 car c'est le nombre d'actions défini dans ton DiscreteActionWrapper
        # et utilisé lors de l'entraînement.
        action_dim = 15
        if hasattr(action_space, 'n'):
            action_dim = action_space.n

        # 3. Définition du Réseau (Feature Extractor + Policy Net)
        # Correspond à 'mlp_extractor.policy_net' dans SB3
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        # 4. Tête d'action (Logits)
        # Correspond à 'action_net' dans SB3
        self.action_net = nn.Linear(64, action_dim)

    def forward(self, t: int):
        """
        Calcule l'action au pas de temps t.
        """
        # Récupère l'observation courante depuis l'espace de travail BBRL
        # La clé "env/env_obs" est standard.
        # L'observation est déjà normalisée et clippée par le NormalizeWrapper.
        obs = self.get(("env/env_obs", t))
        
        # Passage dans le réseau de neurones
        features = self.net(obs)
        action_logits = self.action_net(features)
        
        # Sélection de l'action :
        # On prend l'indice de la plus grande valeur (Argmax) pour un comportement déterministe.
        action = torch.argmax(action_logits, dim=1)
        
        # Enregistre l'action dans l'espace de travail pour que l'environnement l'exécute
        self.set(("action", t), action)

    def load_state_dict(self, state_dict):
        """
        Fonction personnalisée pour charger les poids extraits de Stable Baselines 3.
        Elle fait correspondre les noms des couches de SB3 avec celles de notre classe Actor.
        """
        new_state_dict = {}
        
        for key, value in state_dict.items():
            # --- Mapping du Feature Extractor ---
            # Dans SB3, c'est stocké sous 'mlp_extractor.policy_net.X.weight'
            # Nous le mappons vers 'net.X.weight'
            if "mlp_extractor.policy_net.0" in key:
                new_key = key.replace("mlp_extractor.policy_net.0", "net.0")
            elif "mlp_extractor.policy_net.2" in key:
                # Note : l'index est 2 car l'index 1 est la fonction d'activation Tanh
                new_key = key.replace("mlp_extractor.policy_net.2", "net.2")
            
            # --- Mapping de la Tête d'Action ---
            # Dans SB3, c'est 'action_net.weight'
            # Nous le gardons tel quel ou le mappons vers 'action_net'
            elif "action_net" in key:
                new_key = key
            
            # --- Ignorer le reste ---
            # On ignore 'value_net', 'mlp_extractor.value_net', etc.
            # car l'acteur n'a pas besoin de la fonction de valeur pour piloter.
            else:
                continue
                
            new_state_dict[new_key] = value
            
        # Chargement des poids avec strict=False
        # C'est nécessaire car 'new_state_dict' ne contient pas tout ce qu'il y avait 
        # dans le fichier d'origine, mais contient exactement ce dont notre Acteur a besoin.
        super().load_state_dict(new_state_dict, strict=False)
        


# Version"1"
# import gymnasium as gym
# from bbrl.agents import Agent
# import torch
# import torch.nn as nn

# class Actor(Agent):
#     def __init__(self, observation_space, action_space):
#         super().__init__()
        
#         # --- 1. CONFIGURATION STRICTE (FrameStack 4) ---
#         # Ton modèle PPO attend 616 entrées. On force cette valeur.
#         self.net_input_size = 616
#         self.num_stack = 4
#         self.history = None 

#         # --- 2. SORTIES (ACTIONS) ---
#         # On s'assure de suivre l'ordre exact de l'environnement
#         if isinstance(action_space, gym.spaces.MultiDiscrete):
#             self.output_dims = action_space.nvec.tolist()
#             output_size = int(sum(self.output_dims))
#             self.is_multi = True
#         else:
#             # Fallback (Au cas où l'env change)
#             output_size = 4
#             self.is_multi = False

#         # --- 3. NORMALISATION (Importante !) ---
#         # Ces buffers seront remplis par le load_state_dict du .pth
#         self.register_buffer("obs_mean", torch.zeros(self.net_input_size))
#         self.register_buffer("obs_var", torch.ones(self.net_input_size))
#         self.epsilon = 1e-8

#         # --- 4. LE RÉSEAU NEURAL ---
#         self.net = nn.Sequential(
#             nn.Linear(self.net_input_size, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#             nn.Linear(64, output_size),
#         )

#     def forward(self, t: int, **kwargs):
#         # 1. Récupération de l'observation brute (154,)
#         obs = self.get(("env/env_obs", t))
        
#         # Sécurité : Conversion en Tensor si nécessaire
#         if not torch.is_tensor(obs):
#             obs = torch.tensor(obs, device=self.obs_mean.device, dtype=torch.float32)
#         else:
#             obs = obs.to(self.obs_mean.device, dtype=torch.float32)

#         # Gestion du Batch : (154,) -> (1, 154)
#         if obs.dim() == 1:
#             obs = obs.unsqueeze(0)
            
#         batch_size = obs.shape[0]

#         # 2. FRAME STACKING (Le cœur du problème)
#         # Si c'est le début (t=0) OU si la taille de batch change (nouvel épisode)
#         if t == 0 or self.history is None or self.history.shape[0] != batch_size:
#             # On répète la première image 4 fois pour remplir la mémoire
#             self.history = obs.unsqueeze(1).repeat(1, self.num_stack, 1)
#         else:
#             # Sinon, on décale tout vers la gauche et on ajoute la nouvelle image à la fin
#             self.history = torch.roll(self.history, shifts=-1, dims=1)
#             self.history[:, -1, :] = obs

#         # 3. Aplatir l'historique : (Batch, 4, 154) -> (Batch, 616)
#         obs_stacked = self.history.view(batch_size, -1)

#         # 4. NORMALISATION + CLIPPING (CORRECTION CRITIQUE !)
#         # SB3 fait toujours un clip entre -10 et 10. Sans ça, le réseau explose.
#         obs_norm = (obs_stacked - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon)
#         obs_norm = torch.clamp(obs_norm, -10.0, 10.0) # <--- LA LIGNE QUI SAUVE LA VIE

#         # 5. Prédiction du réseau (Logits)
#         scores = self.net(obs_norm)
        
#         # 6. Décodage de l'action
#         if self.is_multi:
#             pointer = 0
#             actions = []
#             # On découpe le vecteur de sortie selon les dimensions (Steer, Accel, etc.)
#             for dim in self.output_dims:
#                 sub_scores = scores[:, pointer : pointer + dim]
#                 actions.append(sub_scores.argmax(dim=1))
#                 pointer += dim
#             final_action = torch.stack(actions, dim=1)
#         else:
#             final_action = scores.argmax(dim=1)

#         # 7. Envoi de l'action
#         self.set(("action", t), final_action)