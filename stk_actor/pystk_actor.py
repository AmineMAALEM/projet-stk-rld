import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from typing import List, Callable
from stable_baselines3.common.policies import ActorCriticPolicy

# --- CONFIGURATION ---
env_name = "supertuxkart/flattened_multidiscrete-v0"
player_name = "PPO_Agent_SB3"

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    return [lambda env: gym.wrappers.FlattenObservation(env)]

class SB3Agent(nn.Module):
    def __init__(self, actor, vec_norm_path):
        super().__init__()
        self.actor = actor
        self.n_stack = 4
        self.history = None
        
        if os.path.exists(vec_norm_path):
            try:
                with open(vec_norm_path, "rb") as f:
                    vn = pickle.load(f)
                    self.obs_mean = vn.obs_rms.mean
                    self.obs_var = vn.obs_rms.var
                    self.epsilon = 1e-8
                print("[INFO] Normalisation chargée.")
            except Exception as e:
                print(f"[WARN] Erreur stats: {e}")

    def forward(self, observation, t=0, **kwargs):
        obs_data = observation.get("env/env_obs", t)
        obs_np = obs_data.detach().cpu().numpy().flatten() if torch.is_tensor(obs_data) else np.array(obs_data).flatten()
        
        if self.history is None:
            self.history = np.tile(obs_np, (self.n_stack, 1))
        else:
            self.history = np.roll(self.history, shift=-1, axis=0)
            self.history[-1] = obs_np
        
        flat_stacked = self.history.reshape(1, -1)

        if hasattr(self, 'obs_mean'):
            final_obs = (flat_stacked - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
            final_obs = np.clip(final_obs, -10, 10)
        else:
            final_obs = flat_stacked

        with torch.no_grad():
            raw_action, _ = self.actor.predict(final_obs, deterministic=True)

        # Extraction propre des scalaires (Règle le TypeError/ValueError)
        # raw_action peut être [ [a, b, c, ...] ] donc on aplatit
        acts = np.array(raw_action).flatten()

        # --- SESSION DE TEST : REMAPPING ---
        # Ordre STK : [Steer, Accel, Brake, Drift, Nitro, Item, Rescue]
        final_mapped_action = np.zeros(7, dtype=np.int64)
        
        # Test 1: On assigne les 7 valeurs prédites dans l'ordre par défaut
        for i in range(min(len(acts), 7)):
            final_mapped_action[i] = int(acts[i])

        # --- MODE DIAGNOSTIC : DECOMMENTE UNE LIGNE POUR FORCER UNE ACTION ---
        # final_mapped_action[1] = 1  # Force l'accélération à fond pour tester la direction
        # final_mapped_action[0] = 0  # Force à aller tout droit pour tester l'accélération

        if t % 500 == 0:
            print(f"t={t} | Prédit (aplatit): {acts} | Final: {final_mapped_action}")

        act_tensor = torch.tensor(final_mapped_action, dtype=torch.long)
        for key in ["action", "action/discrete"]:
            if key not in observation.variables:
                observation.set_full(key, torch.zeros((10000, 1, 7), dtype=torch.long))
            try:
                observation.variables[key].tensor[t, 0] = act_tensor
            except: pass
        return act_tensor

def get_actor(state, observation_space, action_space):
    fixed_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(616,), dtype=np.float32)
    actor_policy = ActorCriticPolicy(fixed_obs_space, action_space, lr_schedule=lambda _: 0.0, net_arch=dict(pi=[64, 64], vf=[64, 64]))
    if state is not None: actor_policy.load_state_dict(state)
    pkl_path = os.path.join(os.path.dirname(__file__), "normalization.pkl")
    return SB3Agent(actor_policy, pkl_path)