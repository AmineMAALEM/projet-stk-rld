import gymnasium as gym
import torch
import pickle
import os
import numpy as np
from collections import deque 

from stable_baselines3.common.policies import ActorCriticPolicy

# --- CONFIGURATION ---
env_name = "supertuxkart/flattened_multidiscrete-v0" 
player_name = "PPO_Default_64"

def get_actor(state, observation_space, action_space):
    # 1. Architecture PPO PAR DÉFAUT
    # C'est ici qu'on change : [64, 64] au lieu de [256, 256]
    actor = ActorCriticPolicy(
        observation_space,
        action_space,
        lr_schedule=lambda _: 0.0,
        net_arch=dict(pi=[64, 64], vf=[64, 64]) 
    )
    
    # 2. Chargement des poids
    try:
        actor.load_state_dict(state)
        actor.set_training_mode(False)
    except Exception as e:
        print(f"❌ ERREUR DE CHARGEMENT : {e}")
        print("💡 Conseil : Si l'erreur parle de 'size mismatch', vérifie si net_arch doit être [256, 256].")

    # 3. Normalisation
    base_path = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(base_path, "normalization.pkl")
    
    vec_norm = None
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            vec_norm = pickle.load(f)
        print("✅ Normalisation chargée.")

    # 4. Mémoire (FrameStack 4)
    frame_stack = deque(maxlen=4)

    def policy(observation):
        # A. Aplatir (Flatten)
        obs_flat = observation
        if isinstance(observation, dict):
            # On prend toutes les valeurs du dictionnaire et on les met à la suite
            # C'est une approximation robuste pour s'assurer d'avoir un vecteur plat
            vals = []
            for k in sorted(observation.keys()): # Sorted pour garantir l'ordre
                v = np.array(observation[k]).flatten()
                vals.append(v)
            obs_flat = np.concatenate(vals)
        
        # B. Remplir la mémoire (Stacking)
        # Si la pile est vide (début de course), on copie l'image 4 fois
        if len(frame_stack) == 0:
            for _ in range(4):
                frame_stack.append(obs_flat)
        else:
            frame_stack.append(obs_flat)
        
        # On colle les 4 images pour avoir la même "forme" que pendant l'entrainement
        stacked_obs = np.concatenate(frame_stack)

        # C. Appliquer la Normalisation
        if vec_norm is not None:
            mean = vec_norm.obs_rms.mean
            var = vec_norm.obs_rms.var
            epsilon = 1e-8
            
            # On vérifie que la taille correspond avant de diviser
            if stacked_obs.shape == mean.shape:
                stacked_obs = (stacked_obs - mean) / np.sqrt(var + epsilon)

        # D. Action
        obs_tensor = torch.tensor(stacked_obs).float().unsqueeze(0)
        with torch.no_grad():
            action, _ = actor.predict(obs_tensor, deterministic=True)
        
        return action[0]

    return policy