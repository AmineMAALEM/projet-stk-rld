import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

# =============================================================================
# 1. FEATURE ENGINEERING WRAPPER
# =============================================================================
class FeatureEngineeringWrapper(gym.Wrapper):
    """
    Ajoute 11 features scalaires au dictionnaire d'observation.
    """
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._compute_features(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._compute_features(obs)
        return obs, info

    def _compute_features(self, obs):
        # 1. Extraction (Attention aux types np.float32 pour PyTorch)
        velocity_vec = obs['velocity']
        speed = np.linalg.norm(velocity_vec)
        
        center_vec = obs['center_path'] 
        dist_center = np.linalg.norm(center_vec)
        angle_center = np.arctan2(center_vec[0], center_vec[2])

        # 2. Risk
        raw_widths = obs.get('paths_width', np.array([10.0]))
        safe_width = np.min(raw_widths) 
        limit_dist = safe_width / 2.0
        
        current_lateral_pos = -center_vec[0]
        lateral_velocity = velocity_vec[0]
        future_lateral_pos = current_lateral_pos + (lateral_velocity * 0.5)
        
        future_risk = 0.0
        if limit_dist > 0:
            future_risk = abs(future_lateral_pos) / limit_dist

        # 3. Lookahead
        paths_end = obs.get('paths_end', [])
        lookahead_angle = 0.0
        if len(paths_end) > 3:
            target_vec = paths_end[2] 
            lookahead_angle = np.arctan2(target_vec[0], target_vec[2])
        
        curve_intensity = lookahead_angle - angle_center

        # 4. Physique & Items
        skeed = float(obs.get('skeed_factor', 0.0))
        is_airborne = float(obs.get('jumping', 0))

        items_pos = obs.get('items_position', [])
        item_angle = 0.0
        has_item_in_sight = 0.0
        
        if len(items_pos) > 0:
            dists = np.linalg.norm(items_pos, axis=1)
            min_idx = np.argmin(dists)
            closest_item = items_pos[min_idx]
            if closest_item[2] > 0 and dists[min_idx] < 20.0:
                item_angle = np.arctan2(closest_item[0], closest_item[2])
                has_item_in_sight = 1.0

        # 5. Injection des 11 features
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
# 2. DISCRETE ACTION WRAPPER (CORRIGÉ: PAS DE RESET DANS INIT)
# =============================================================================
class DiscreteActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # --- CORRECTION CRITIQUE ---
        # On calcule la taille théorique sans appeler reset()
        # car le serveur initialise un "FakeEnv" qui crash si on reset.
        
        base_size = 0
        base_space = self.env.observation_space
        
        # 1. On calcule la taille de l'espace 'simple-v0' original
        if isinstance(base_space, spaces.Dict):
            for key in base_space.spaces:
                space = base_space.spaces[key]
                if isinstance(space, spaces.Box):
                    base_size += int(np.prod(space.shape))
                elif isinstance(space, spaces.Discrete):
                    base_size += 1
                # On ignore les autres types complexes s'il y en a
        elif isinstance(base_space, spaces.Box):
             base_size = int(np.prod(base_space.shape))
             
        # 2. On ajoute la taille des 11 features ajoutées par FeatureEngineeringWrapper
        # (speed, dist, angle, risk, lookahead, curve, skeed, air, item_angle, item_detect, off_track)
        added_features_size = 11
        
        total_flat_size = base_size + added_features_size
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_flat_size,), dtype=np.float32
        )

        # 15 Actions
        self.actions_list = [
            (0.0, 1.0, 0, 0, 0, 0, 0), (-0.6, 1.0, 0, 0, 0, 0, 0), (0.6, 1.0, 0, 0, 0, 0, 0),
            (-1.0, 1.0, 0, 1, 0, 0, 0), (1.0, 1.0, 0, 1, 0, 0, 0), (-0.5, 1.0, 0, 1, 0, 0, 0),
            (0.5, 1.0, 0, 1, 0, 0, 0), (0.0, 0.3, 0, 0, 0, 0, 0), (-0.6, 0.3, 0, 0, 0, 0, 0),
            (0.6, 0.3, 0, 0, 0, 0, 0), (0.0, 0.0, 1, 0, 0, 0, 0), (-1.0, 0.0, 1, 0, 0, 0, 0),
            (1.0, 0.0, 1, 0, 0, 0, 0), (0.0, 1.0, 0, 0, 1, 0, 0), (0.0, 1.0, 0, 0, 0, 1, 0),
        ]
        self.action_space = spaces.Discrete(len(self.actions_list))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info

    def step(self, action_idx):
        if hasattr(action_idx, "item"): action_idx = int(action_idx.item())
        else: action_idx = int(action_idx)
            
        vals = self.actions_list[action_idx]
        game_action = {
            'steer': np.array([vals[0]], dtype=np.float32),
            'acceleration': np.array([vals[1]], dtype=np.float32),
            'brake': vals[2], 'drift': vals[3], 'fire': vals[4], 'nitro': vals[5], 'rescue': vals[6]
        }
        obs, reward, terminated, truncated, info = self.env.step(game_action)
        return self._flatten_obs(obs), reward, terminated, truncated, info

    def _flatten_obs(self, obs):
        flat_list = []
        # Le tri est important pour que l'ordre des features soit constant
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, (int, float, np.number)): flat_list.append([val])
            elif isinstance(val, np.ndarray): flat_list.append(val.flatten())
        return np.concatenate(flat_list).astype(np.float32)


# =============================================================================
# 3. FRAME STACKING WRAPPER
# =============================================================================
class FrameStackingWrapper(gym.Wrapper):
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)
        
        original_shape = env.observation_space.shape[0]
        new_shape = original_shape * n_stack
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_shape,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_stack): self.frames.append(obs)
        return self._get_stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked_obs(), reward, terminated, truncated, info

    def _get_stacked_obs(self):
        return np.concatenate(list(self.frames)).astype(np.float32)


# =============================================================================
# 4. NORMALIZATION WRAPPER
# =============================================================================
class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, mean, var, epsilon=1e-8, clip_obs=10.0):
        super().__init__(env)
        self.mean = mean
        self.var = var
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def observation(self, observation):
        norm_obs = (observation - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(norm_obs, -self.clip_obs, self.clip_obs)