import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

# =============================================================================
# 1. FEATURE ENGINEERING (ObservationWrapper)
# =============================================================================
class FeatureEngineeringWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        return self._compute_features(obs)

    def _compute_features(self, obs):
        # Extraction
        velocity_vec = obs['velocity']
        speed = np.linalg.norm(velocity_vec)
        center_vec = obs['center_path'] 
        dist_center = np.linalg.norm(center_vec)
        angle_center = np.arctan2(center_vec[0], center_vec[2])

        # Risk
        raw_widths = obs.get('paths_width', np.array([10.0]))
        safe_width = np.min(raw_widths) 
        limit_dist = safe_width / 2.0
        future_risk = 0.0
        if limit_dist > 0:
            current_lateral_pos = -center_vec[0]
            future_pos = current_lateral_pos + (velocity_vec[0] * 0.5)
            future_risk = abs(future_pos) / limit_dist

        # Lookahead
        paths_end = obs.get('paths_end', [])
        lookahead_angle = 0.0
        if len(paths_end) > 3:
            target = paths_end[2]
            lookahead_angle = np.arctan2(target[0], target[2])
        curve_intensity = lookahead_angle - angle_center

        # Physique & Items
        skeed = float(obs.get('skeed_factor', 0.0))
        is_airborne = float(obs.get('jumping', 0))
        
        items = obs.get('items_position', [])
        item_angle = 0.0
        has_item = 0.0
        if len(items) > 0:
            dists = np.linalg.norm(items, axis=1)
            min_idx = np.argmin(dists)
            if items[min_idx][2] > 0 and dists[min_idx] < 20.0:
                item_angle = np.arctan2(items[min_idx][0], items[min_idx][2])
                has_item = 1.0

        # Injection
        obs['feat_speed'] = np.array([speed], dtype=np.float32)
        obs['feat_dist_center'] = np.array([dist_center], dtype=np.float32)
        obs['feat_angle_center'] = np.array([angle_center], dtype=np.float32)
        obs['feat_future_risk'] = np.array([future_risk], dtype=np.float32)
        obs['feat_lookahead_angle'] = np.array([lookahead_angle], dtype=np.float32)
        obs['feat_curve_intensity'] = np.array([curve_intensity], dtype=np.float32)
        obs['feat_skeed'] = np.array([skeed], dtype=np.float32)
        obs['feat_air'] = np.array([is_airborne], dtype=np.float32)
        obs['feat_item_angle'] = np.array([item_angle], dtype=np.float32)
        obs['feat_item_detected'] = np.array([has_item], dtype=np.float32)
        obs['feat_off_track'] = np.array([1.0 if dist_center > limit_dist else 0.0], dtype=np.float32)

        return obs

# =============================================================================
# 2. ACTION CONVERSION (ActionWrapper)
# =============================================================================
class ActionConversionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 15 actions
        self.actions_list = [
            (0.0, 1.0, 0, 0, 0, 0, 0), (-0.6, 1.0, 0, 0, 0, 0, 0), (0.6, 1.0, 0, 0, 0, 0, 0),
            (-1.0, 1.0, 0, 1, 0, 0, 0), (1.0, 1.0, 0, 1, 0, 0, 0), (-0.5, 1.0, 0, 1, 0, 0, 0),
            (0.5, 1.0, 0, 1, 0, 0, 0), (0.0, 0.3, 0, 0, 0, 0, 0), (-0.6, 0.3, 0, 0, 0, 0, 0),
            (0.6, 0.3, 0, 0, 0, 0, 0), (0.0, 0.0, 1, 0, 0, 0, 0), (-1.0, 0.0, 1, 0, 0, 0, 0),
            (1.0, 0.0, 1, 0, 0, 0, 0), (0.0, 1.0, 0, 0, 1, 0, 0), (0.0, 1.0, 0, 0, 0, 1, 0),
        ]
        self.action_space = spaces.Discrete(len(self.actions_list))

    def action(self, action_idx):
        if hasattr(action_idx, "item"): action_idx = int(action_idx.item())
        else: action_idx = int(action_idx)
        vals = self.actions_list[action_idx]
        return {
            'steer': np.array([vals[0]], dtype=np.float32),
            'acceleration': np.array([vals[1]], dtype=np.float32),
            'brake': vals[2], 'drift': vals[3], 'fire': vals[4], 'nitro': vals[5], 'rescue': vals[6]
        }

# =============================================================================
# 3. FLATTEN WRAPPER (WHITELIST ROBUSTE)
# =============================================================================
class FlattenWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # LISTE STRICTE : On ne garde QUE ce que tu as utilisé en local (112 dims).
        # Les clés du serveur non listées ici seront ignorées.
        self.keys_whitelist = sorted([
            'attachment', 'attachment_time_left', 'aux_ticks', 'center_path', 
            'center_path_distance', 'distance_down_track', 'energy', 'front', 
            'items_position', 'items_type', 'jumping', 'karts_position', 
            'max_steer_angle', 'paths_distance', 'paths_end', 'paths_start', 
            'paths_width', 'phase', 'powerup', 'shield_time', 'skeed_factor', 
            'velocity',
            # Tes Features
            'feat_speed', 'feat_dist_center', 'feat_angle_center', 'feat_future_risk',
            'feat_lookahead_angle', 'feat_curve_intensity', 'feat_skeed', 'feat_air',
            'feat_item_angle', 'feat_item_detected', 'feat_off_track'
        ])
        
        self.target_size = 112
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.target_size,), dtype=np.float32
        )

    def observation(self, obs):
        flat_list = []
        # On itère sur la Whitelist : Si le serveur a des clés en plus, on ne les voit pas.
        for key in self.keys_whitelist:
            if key in obs:
                val = obs[key]
                if isinstance(val, (int, float, np.number)): 
                    flat_list.append([val])
                elif isinstance(val, np.ndarray): 
                    flat_list.append(val.flatten())
            # Si une clé manque (peu probable), on ignore (pas de crash, juste vecteur plus court)
            # Mais simple-v0 est stable sur les clés de base.
        
        flat_obs = np.concatenate(flat_list).astype(np.float32)
        
        # Sécurité ultime sur la taille (Padding si nécessaire)
        if len(flat_obs) < self.target_size:
             padding = np.zeros(self.target_size - len(flat_obs), dtype=np.float32)
             flat_obs = np.concatenate([flat_obs, padding])
             
        return flat_obs

# =============================================================================
# 4. FRAME STACKING
# =============================================================================
class FrameStackingWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)
        
        original_shape = 112
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(original_shape * n_stack,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.n_stack): 
            self.frames.append(obs)
        return self._get_stacked_obs(), info

    def observation(self, obs):
        self.frames.append(obs)
        return self._get_stacked_obs()

    def _get_stacked_obs(self):
        return np.concatenate(list(self.frames)).astype(np.float32)