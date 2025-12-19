from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym
import torch
import numpy as np

# =============================================================================
# IMPORTS RELATIFS (CRITIQUE : Ne pas changer)
# =============================================================================
from .actors import Actor
from .wrappers import (
    FeatureEngineeringWrapper, 
    DiscreteActionWrapper, 
    FrameStackingWrapper, 
    NormalizeWrapper
)

# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================

# ⚠️ IMPORTANT : On utilise 'simple-v0' car 'FeatureEngineeringWrapper'
# a besoin du dictionnaire brut (velocity, center_path...) pour fonctionner.
# 'flattened_multidiscrete-v0' écraserait ces données.
env_name = "supertuxkart/simple-v0"

# 🏷️ Mets ici le nom de ton équipe qui s'affichera au-dessus du Kart
player_name = "Vroom Vroom"

# Variable globale pour stocker temporairement les stats de normalisation
# chargées depuis le fichier .pth, afin de les passer aux wrappers.
norm_stats = {}

# =============================================================================
# DÉFINITION DES WRAPPERS
# =============================================================================
def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """
    Retourne la liste des wrappers à appliquer à l'environnement.
    L'ordre d'application effectif est celui de la liste.
    """
    wrappers = [
        # 1. Calcul des Features (Transforme l'obs brute en obs enrichie)
        lambda env: FeatureEngineeringWrapper(env),
        
        # 2. Gestion Action Discrète & Aplatissement (Dict -> Array)
        lambda env: DiscreteActionWrapper(env),
        
        # 3. Mémoire à court terme (Empile 4 frames)
        lambda env: FrameStackingWrapper(env, n_stack=4),
    ]
    
    # 4. Normalisation (Seulement si les stats ont été chargées)
    if "mean" in norm_stats:
        wrappers.append(lambda env: NormalizeWrapper(
            env, 
            mean=norm_stats["mean"], 
            var=norm_stats["var"], 
            epsilon=norm_stats["epsilon"],
            clip_obs=norm_stats.get("clip", 10.0) # Par défaut 10.0 si pas trouvé
        ))
    
    return wrappers

# =============================================================================
# CHARGEMENT DE L'ACTEUR
# =============================================================================
def get_actor(
    state: dict | None,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
) -> Agent:
    """
    Crée l'agent BBRL et charge les poids.
    Cette fonction est appelée par le script d'évaluation avec le contenu de pystk_actor.pth
    """
    
    # Cas 1 : Aucun fichier chargé (ex: initialisation random)
    if state is None:
        return Actor(observation_space, action_space)

    # Cas 2 : Fichier chargé -> Extraction des données
    
    # --- A. Extraction des stats de Normalisation ---
    # On remplit la variable globale pour que get_wrappers() puisse l'utiliser
    if "norm_mean" in state:
        norm_stats["mean"] = state["norm_mean"]
        norm_stats["var"] = state["norm_var"]
        norm_stats["epsilon"] = state["norm_epsilon"]
        # On récupère le clip s'il existe, sinon 10.0 par défaut
        norm_stats["clip"] = state.get("norm_clip", 10.0)
    
    # --- B. Instanciation de l'Acteur ---
    # Note : observation_space ici correspond à l'espace APRÈS l'application
    # de tous les wrappers (donc Stacked + Normalized)
    actor = Actor(observation_space, action_space)
    
    # --- C. Chargement des Poids du Réseau ---
    if "model_state_dict" in state:
        # Cas où on a utilisé notre script d'export propre (Recommandé)
        actor.load_state_dict(state["model_state_dict"])
    else:
        # Fallback : Cas où le fichier contient directement le state_dict
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