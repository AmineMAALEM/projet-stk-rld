import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

# =============================================================================
# 1. FEATURE ENGINEERING WRAPPER (Les Yeux & Capteurs)
# =============================================================================
class FeatureEngineeringWrapper(gym.Wrapper):
    """
    Calcule des indicateurs physiques et stratégiques (trajectoire, risque, items)
    et les ajoute au dictionnaire d'observation.
    """
    def __init__(self, env):
        super().__init__(env)
        # On ne modifie pas l'observation_space ici car on renvoie toujours un Dict,
        # mais enrichi. C'est le DiscreteActionWrapper qui va aplatir tout ça.
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._compute_features(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._compute_features(obs)
        return obs, info

    def _compute_features(self, obs):
        # --- 1. BASES (Vitesse & Position) ---
        velocity_vec = obs['velocity']
        speed = np.linalg.norm(velocity_vec)
        
        center_vec = obs['center_path'] 
        dist_center = np.linalg.norm(center_vec)
        # Angle par rapport au centre de la piste
        angle_center = np.arctan2(center_vec[0], center_vec[2])

        # --- 2. FUTURE RISK (Risque de sortie de piste) ---
        raw_widths = obs.get('paths_width', np.array([10.0]))
        safe_width = np.min(raw_widths) 
        limit_dist = safe_width / 2.0
        
        current_lateral_pos = -center_vec[0]
        lateral_velocity = velocity_vec[0]
        TIME_HORIZON = 0.5
        future_lateral_pos = current_lateral_pos + (lateral_velocity * TIME_HORIZON)
        
        if limit_dist > 0:
            future_risk = abs(future_lateral_pos) / limit_dist
        else:
            future_risk = 0.0

        # --- 3. LOOKAHEAD (Anticipation virage) ---
        paths_end = obs.get('paths_end', [])
        lookahead_angle = 0.0
        
        # On vise un point futur (le 3ème segment si dispo)
        if len(paths_end) > 3:
            target_vec = paths_end[2] 
            lookahead_angle = np.arctan2(target_vec[0], target_vec[2])
        
        curve_intensity = lookahead_angle - angle_center

        # --- 4. PHYSIQUE (Dérapage & Saut) ---
        skeed = float(obs.get('skeed_factor', 0.0))
        is_airborne = float(obs.get('jumping', 0))

        # --- 5. ITEMS (Détection bonus) ---
        items_pos = obs.get('items_position', [])
        item_angle = 0.0
        has_item_in_sight = 0.0
        
        if len(items_pos) > 0:
            dists = np.linalg.norm(items_pos, axis=1)
            min_idx = np.argmin(dists)
            closest_item = items_pos[min_idx]
            closest_dist = dists[min_idx]
            
            # Item devant (Z > 0) et proche (< 20m)
            if closest_item[2] > 0 and closest_dist < 20.0:
                item_angle = np.arctan2(closest_item[0], closest_item[2])
                has_item_in_sight = 1.0

        # --- ASSEMBLAGE DES FEATURES ---
        # On ajoute ces valeurs au dictionnaire existant
        # (Conversion explicite en array float32 pour éviter erreurs PyTorch)
        obs['feat_speed'] = np.array([speed], dtype=np.float32)
        obs['feat_dist_center'] = np.array([dist_center], dtype=np.float32)
        obs['feat_angle_center'] = np.array([angle_center], dtype=np.float32)
        obs['feat_future_risk'] = np.array([future_risk], dtype=np.float32)
        obs['feat_lookahead_angle'] = np.array([lookahead_angle], dtype=np.float32)
        obs['feat_curve_intensity'] = np.array([curve_intensity], dtype=np.float32)
        obs['feat_skeed'] = np.array([skeed], dtype=np.float32)
        obs['feat_air'] = np.array([is_airborne], dtype=np.float32)
        obs['feat_item_angle'] = np.array([item_angle], dtype=np.float32)
        obs['feat_item_detected'] = np.array([has_item_in_sight], dtype=np.float32)
        
        is_off_track = 1.0 if dist_center > limit_dist else 0.0
        obs['feat_off_track'] = np.array([is_off_track], dtype=np.float32)

        return obs


# =============================================================================
# 2. DISCRETE ACTION WRAPPER (Conversion Action + Aplatissement Obs)
# =============================================================================
class DiscreteActionWrapper(gym.Wrapper):
    """
    1. Convertit l'index d'action (0-14) en dictionnaire d'action STK.
    2. Aplatit le dictionnaire d'observation en un vecteur unique (Box).
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Calcul dynamique de la taille de l'observation aplatie
        dummy_obs, _ = self.env.reset()
        flat_size = self._flatten_obs(dummy_obs).shape[0]
        
        # Redéfinition de l'espace d'observation (C'est maintenant un vecteur)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_size,), dtype=np.float32
        )

        # Liste des 15 actions définies dans ton entraînement
        # (Steer, Accel, Brake, Drift, Fire, Nitro, Rescue)
        self.actions_list = [
            (0.0, 1.0, 0, 0, 0, 0, 0),   # 0
            (-0.6, 1.0, 0, 0, 0, 0, 0),  # 1
            (0.6, 1.0, 0, 0, 0, 0, 0),   # 2
            (-1.0, 1.0, 0, 1, 0, 0, 0),  # 3
            (1.0, 1.0, 0, 1, 0, 0, 0),   # 4
            (-0.5, 1.0, 0, 1, 0, 0, 0),  # 5
            (0.5, 1.0, 0, 1, 0, 0, 0),   # 6
            (0.0, 0.3, 0, 0, 0, 0, 0),   # 7
            (-0.6, 0.3, 0, 0, 0, 0, 0),  # 8
            (0.6, 0.3, 0, 0, 0, 0, 0),   # 9
            (0.0, 0.0, 1, 0, 0, 0, 0),   # 10
            (-1.0, 0.0, 1, 0, 0, 0, 0),  # 11
            (1.0, 0.0, 1, 0, 0, 0, 0),   # 12
            (0.0, 1.0, 0, 0, 1, 0, 0),   # 13
            (0.0, 1.0, 0, 0, 0, 1, 0),   # 14
        ]
        self.action_space = spaces.Discrete(len(self.actions_list))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info

    def step(self, action_idx):
        # Sécurité : si l'action vient d'un tenseur PyTorch, on extrait la valeur
        if hasattr(action_idx, "item"):
            action_idx = int(action_idx.item())
        else:
            action_idx = int(action_idx)
            
        vals = self.actions_list[action_idx]
        
        game_action = {
            'steer': np.array([vals[0]], dtype=np.float32),
            'acceleration': np.array([vals[1]], dtype=np.float32),
            'brake': vals[2],
            'drift': vals[3],
            'fire': vals[4],
            'nitro': vals[5],
            'rescue': vals[6]
        }
        
        obs, reward, terminated, truncated, info = self.env.step(game_action)
        return self._flatten_obs(obs), reward, terminated, truncated, info

    def _flatten_obs(self, obs):
        """Aplatit le dictionnaire en un seul vecteur numpy float32."""
        flat_list = []
        # Le tri des clés est CRITIQUE pour que l'ordre reste constant
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, (int, float, np.number)):
                flat_list.append([val])
            elif isinstance(val, np.ndarray):
                flat_list.append(val.flatten())
            # On ignore les listes python simples non gérées ici, 
            # mais FeatureEngineering ne renvoie que des arrays/scalaires.
                
        return np.concatenate(flat_list).astype(np.float32)


# =============================================================================
# 3. FRAME STACKING WRAPPER (Mémoire court terme)
# =============================================================================
class FrameStackingWrapper(gym.Wrapper):
    """
    Empile n_stack observations consécutives.
    """
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)
        
        # L'entrée est déjà aplatie par DiscreteActionWrapper
        original_shape = env.observation_space.shape[0]
        new_shape = original_shape * n_stack
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_shape,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Au reset, on remplit le buffer avec la même image initiale
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return self._get_stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked_obs(), reward, terminated, truncated, info

    def _get_stacked_obs(self):
        # Concaténation des frames stockées
        return np.concatenate(list(self.frames)).astype(np.float32)


# =============================================================================
# 4. NORMALIZATION WRAPPER (Avec CLIPPING)
# =============================================================================
class NormalizeWrapper(gym.ObservationWrapper):
    """
    Simule VecNormalize de SB3 (inférence uniquement).
    Applique (obs - mean) / sqrt(var + eps)
    Puis CLIPPING (Très important car tu avais clip_obs=10).
    """
    def __init__(self, env, mean, var, epsilon=1e-8, clip_obs=10.0):
        super().__init__(env)
        self.mean = mean
        self.var = var
        self.epsilon = epsilon
        self.clip_obs = clip_obs
        # On ne touche pas à l'observation_space (taille identique)

    def observation(self, observation):
        # 1. Normalisation
        norm_obs = (observation - self.mean) / np.sqrt(self.var + self.epsilon)
        
        # 2. Clipping (Empêche les valeurs extrêmes de casser le réseau)
        return np.clip(norm_obs, -self.clip_obs, self.clip_obs)