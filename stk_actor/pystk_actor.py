from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym
import torch
import numpy as np

try:
    from .actors import Actor
    # from .wrappers import MyActionWrapper, MyObservationWrapper
    from .wrappers import ActionConversionWrapper, UltraWrapper
except ImportError:
    from actors import Actor
    # from wrappers import MyActionWrapper, MyObservationWrapper,
    from wrappers import ActionConversionWrapper, UltraWrapper


env_name = "supertuxkart/simple-v0"
player_name = "Vroom Vroom"

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    # L'ordre dans la liste [W1, W2] signifie W2(W1(env)).
    return [
        # 1. D'abord on traite les observations (au plus près de l'env)
        # lambda env: MyObservationWrapper(env, n_stack=4),
        lambda env: UltraWrapper(env, n_stack=4),


        
        # 2. Ensuite on traite les actions (au plus près de l'agent)
        lambda env: ActionConversionWrapper(env)
    ]

def get_actor(
    state: dict | None,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
) -> Agent:
    
    actor = Actor(observation_space, action_space)
    if state is None: return actor

    if "model_state_dict" in state:
        actor.load_state_dict(state["model_state_dict"])
    else:
        actor.load_state_dict(state)
        
    if "norm_mean" in state:
        m, v = state["norm_mean"], state["norm_var"]
        if not isinstance(m, torch.Tensor): m = torch.as_tensor(m).float()
        if not isinstance(v, torch.Tensor): v = torch.as_tensor(v).float()
        actor.set_normalization_stats(m, v, state.get("norm_epsilon", 1e-8), state.get("norm_clip", 10.0))

    return actor
# from typing import List, Callable
# from bbrl.agents import Agents, Agent
# import gymnasium as gym
# import torch
# import numpy as np

# try:
#     from .actors import Actor
#     from .wrappers import UltraWrapper, ActionConversionWrapper
# except ImportError:
#     from actors import Actor
#     from wrappers import UltraWrapper, ActionConversionWrapper

# env_name = "supertuxkart/multi-full-v0"
# player_name = "Team_PPO_Final"

# def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
#     return [
#         # Action d'abord (couche extérieure pour le serveur)
#         lambda env: ActionConversionWrapper(env),
#         # Observation ensuite (contient Polar, Constant, Features, Flatten, Stack)
#         lambda env: UltraWrapper(env, n_stack=4)
#     ]

# def get_actor(
#     state: dict | None,
#     observation_space: gym.spaces.Space,
#     action_space: gym.spaces.Space,
# ) -> Agent:
    
#     actor = Actor(observation_space, action_space)
#     if state is None: return actor

#     if "model_state_dict" in state:
#         actor.load_state_dict(state["model_state_dict"])
#     else:
#         actor.load_state_dict(state)
        
#     if "norm_mean" in state:
#         m, v = state["norm_mean"], state["norm_var"]
#         if not isinstance(m, torch.Tensor): m = torch.as_tensor(m).float()
#         if not isinstance(v, torch.Tensor): v = torch.as_tensor(v).float()
#         actor.set_normalization_stats(m, v, state.get("norm_epsilon", 1e-8), state.get("norm_clip", 10.0))

#     return actor
# # from typing import List, Callable
# # from bbrl.agents import Agents, Agent
# # import gymnasium as gym
# # import torch
# # import numpy as np

# # from .actors import Actor
# # from .wrappers import UltraWrapper, ActionConversionWrapper, ConstantSizedObservationsNew
# # from pystk2_gymnasium.stk_wrappers import  PolarObservations, ConstantSizedObservations

# # env_name = "supertuxkart/multi-full-v0"
# # player_name = "Vroom Vroom"

# # def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
# #     return [
# #         lambda env: ConstantSizedObservations(env, state_items=5,state_karts=5,state_paths=5),
# #         lambda env: PolarObservations(env),

        
# #         # Action wrapper en premier (intérieur) ou deuxième peu importe ici car distinct
        
        
# #         lambda env: ActionConversionWrapper(env),
# #         # UltraWrapper fait tout le reste
# #         lambda env: UltraWrapper(env)
# #     ]

# # def get_actor(
# #     state: dict | None,
# #     observation_space: gym.spaces.Space,
# #     action_space: gym.spaces.Space,
# # ) -> Agent:
    
# #     actor = Actor(observation_space, action_space)

# #     if state is None:
# #         return actor

# #     if "model_state_dict" in state:
# #         actor.load_state_dict(state["model_state_dict"])
# #     else:
# #         actor.load_state_dict(state)
        
# #     if "norm_mean" in state:
# #         mean = state["norm_mean"]
# #         var = state["norm_var"]
# #         if not isinstance(mean, torch.Tensor):
# #             mean = torch.as_tensor(mean).float()
# #         if not isinstance(var, torch.Tensor):
# #             var = torch.as_tensor(var).float()
# #         actor.set_normalization_stats(
# #             mean=mean,
# #             var=var,
# #             epsilon=state.get("norm_epsilon", 1e-8),
# #             clip=state.get("norm_clip", 10.0)
# #         )

# #     return actor
# # from typing import List, Callable
# # from bbrl.agents import Agents, Agent
# # import gymnasium as gym
# # import torch
# # import numpy as np

# # from .actors import Actor
# # from .wrappers import (
# #     FeatureEngineeringWrapper, 
# #     ActionConversionWrapper, 
# #     FlattenWrapper, 
# #     FrameStackingWrapper
# # )

# # # Indication pour l'affichage, mais le serveur force multi-full-v0
# # env_name = "supertuxkart/simple-v0"
# # player_name = "Team_PPO_Rocket"

# # def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
# #     return [
# #         lambda env: FeatureEngineeringWrapper(env),
# #         lambda env: ActionConversionWrapper(env),
# #         lambda env: FlattenWrapper(env),
# #         lambda env: FrameStackingWrapper(env, n_stack=4),
# #     ]

# # def get_actor(
# #     state: dict | None,
# #     observation_space: gym.spaces.Space,
# #     action_space: gym.spaces.Space,
# # ) -> Agent:
    
# #     actor = Actor(observation_space, action_space)

# #     if state is None:
# #         return actor

# #     if "model_state_dict" in state:
# #         actor.load_state_dict(state["model_state_dict"])
# #     else:
# #         actor.load_state_dict(state)
        
# #     if "norm_mean" in state:
# #         mean = state["norm_mean"]
# #         var = state["norm_var"]
        
# #         # Sécurité Numpy -> Tensor
# #         if not isinstance(mean, torch.Tensor):
# #             mean = torch.as_tensor(mean).float()
# #         if not isinstance(var, torch.Tensor):
# #             var = torch.as_tensor(var).float()
            
# #         actor.set_normalization_stats(
# #             mean=mean,
# #             var=var,
# #             epsilon=state.get("norm_epsilon", 1e-8),
# #             clip=state.get("norm_clip", 10.0)
# #         )

# #     return actor
# # # from typing import List, Callable
# # # from bbrl.agents import Agents, Agent
# # # import gymnasium as gym
# # # import torch
# # # import numpy as np

# # # # Imports relatifs
# # # from .actors import Actor
# # # # On importe les NOUVEAUX noms définis dans wrappers.py
# # # from .wrappers import (
# # #     FeatureEngineeringWrapper, 
# # #     ActionConversionWrapper, 
# # #     FlattenWrapper, 
# # #     FrameStackingWrapper
# # # )

# # # # On utilise simple-v0 (dictionnaire)
# # # env_name = "supertuxkart/simple-v0"

# # # player_name = "Vroom Vroom"

# # # def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
# # #     return [
# # #         # 1. Features
# # #         lambda env: FeatureEngineeringWrapper(env),
# # #         # 2. Action (Conversion)
# # #         lambda env: ActionConversionWrapper(env),
# # #         # 3. Flatten (Avec Whitelist)
# # #         lambda env: FlattenWrapper(env),
# # #         # 4. Stacking
# # #         lambda env: FrameStackingWrapper(env, n_stack=4),
# # #     ]

# # # def get_actor(
# # #     state: dict | None,
# # #     observation_space: gym.spaces.Space,
# # #     action_space: gym.spaces.Space,
# # # ) -> Agent:
    
# # #     actor = Actor(observation_space, action_space)

# # #     if state is None:
# # #         return actor

# # #     # 1. Chargement des Poids
# # #     if "model_state_dict" in state:
# # #         actor.load_state_dict(state["model_state_dict"])
# # #     else:
# # #         actor.load_state_dict(state)
        
# # #     # 2. Chargement des Stats Normalisation (Vers l'acteur directement)
# # #     if "norm_mean" in state:
# # #         mean = state["norm_mean"]
# # #         var = state["norm_var"]
        
# # #         # Sécurité : Conversion Numpy -> Tensor si nécessaire
# # #         if not isinstance(mean, torch.Tensor):
# # #             mean = torch.as_tensor(mean).float()
# # #         if not isinstance(var, torch.Tensor):
# # #             var = torch.as_tensor(var).float()
            
# # #         actor.set_normalization_stats(
# # #             mean=mean,
# # #             var=var,
# # #             epsilon=state.get("norm_epsilon", 1e-8),
# # #             clip=state.get("norm_clip", 10.0)
# # #         )

# # #     return actor