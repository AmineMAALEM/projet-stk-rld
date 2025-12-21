import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations, PolarObservations

class ActionConversionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
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

class UltraWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_stack=4):
        # Wrappers pystk internes
        env = ConstantSizedObservations(env, state_items=5, state_karts=5, state_paths=5)
        env = PolarObservations(env)
        
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = deque(maxlen=self.n_stack)
        
        self.keys_whitelist = sorted([
            'attachment', 'attachment_time_left', 'aux_ticks', 'center_path', 
            'center_path_distance', 'distance_down_track', 'energy', 'front', 
            'items_position', 'items_type', 'jumping', 'karts_position', 
            'max_steer_angle', 'paths_distance', 'paths_end', 'paths_start', 
            'paths_width', 'phase', 'powerup', 'shield_time', 'skeed_factor', 
            'velocity',
            'feat_speed', 'feat_dist_center', 'feat_angle_center', 'feat_future_risk',
            'feat_lookahead_angle', 'feat_curve_intensity', 'feat_skeed', 'feat_air',
            'feat_item_angle', 'feat_item_detected', 'feat_off_track'
        ])
        
        self.single_frame_size = 112
        self.target_size = self.single_frame_size * n_stack
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.target_size,), dtype=np.float32
        )

    def _compute_features_and_flatten(self, obs):
        # ---------------------------------------------------------
        # 🛡️ ZONE DE SANITIZATION (LE CORRECTIF CRITIQUE) 🛡️
        # ---------------------------------------------------------
        # Si on a plus de karts que l'entraînement (souvent 2), on met les autres à 0.
        # karts_position est shape (5, 3).
        # On garde l'index 0 (nous) et 1 (le premier rival).
        # On efface les index 2, 3, 4.
        if 'karts_position' in obs:
            # On s'assure que c'est un array modifiable
            karts = np.array(obs['karts_position'], copy=True)
            # On masque tout à partir du 3ème kart (index 2)
            # Cela simule un environnement à 2 karts, comme à l'entraînement.
            karts[2:] = 0.0 
            obs['karts_position'] = karts

        # Idem pour les items si l'entraînement avait peu d'items, mais c'est moins grave.
        # ---------------------------------------------------------

        # Feature Engineering (Standard)
        velocity_vec = obs.get('velocity', np.zeros(3))
        speed = np.linalg.norm(velocity_vec)
        center_vec = obs.get('center_path', np.zeros(3))
        dist_center = np.linalg.norm(center_vec)
        angle_center = np.arctan2(center_vec[0], center_vec[2]) if center_vec.shape[0] >= 3 and center_vec[2] != 0 else 0.0

        raw_widths = obs.get('paths_width', np.array([10.0]))
        safe_width = np.min(raw_widths) if raw_widths.size > 0 else 10.0
        limit_dist = safe_width / 2.0
        
        future_risk = abs((-center_vec[0] + velocity_vec[0] * 0.5) / limit_dist) if limit_dist > 0 else 0.0

        paths_end = obs.get('paths_end', [])
        lookahead_angle = np.arctan2(paths_end[2][0], paths_end[2][2]) if len(paths_end) > 3 else 0.0
        curve_intensity = lookahead_angle - angle_center

        skeed = float(obs.get('skeed_factor', 0.0))
        is_airborne = float(obs.get('jumping', 0))
        
        # Items logic
        items = obs.get('items_position', [])
        ia, idet = 0.0, 0.0
        if len(items) > 0:
            dists = np.linalg.norm(items, axis=1)
            for i in range(len(items)):
                if items[i][2] > 0 and dists[i] < 20.0:
                    ia, idet = np.arctan2(items[i][0], items[i][2]), 1.0
                    break

        obs['feat_speed'] = np.array([speed], dtype=np.float32)
        obs['feat_dist_center'] = np.array([dist_center], dtype=np.float32)
        obs['feat_angle_center'] = np.array([angle_center], dtype=np.float32)
        obs['feat_future_risk'] = np.array([future_risk], dtype=np.float32)
        obs['feat_lookahead_angle'] = np.array([lookahead_angle], dtype=np.float32)
        obs['feat_curve_intensity'] = np.array([curve_intensity], dtype=np.float32)
        obs['feat_skeed'] = np.array([skeed], dtype=np.float32)
        obs['feat_air'] = np.array([is_airborne], dtype=np.float32)
        obs['feat_item_angle'] = np.array([ia], dtype=np.float32)
        obs['feat_item_detected'] = np.array([idet], dtype=np.float32)
        obs['feat_off_track'] = np.array([1.0 if dist_center > limit_dist else 0.0], dtype=np.float32)

        # Flatten Whitelist
        flat = []
        for k in self.keys_whitelist:
            val = obs.get(k, np.zeros(1))
            if isinstance(val, (int, float, np.number)): flat.append([val])
            else: flat.append(val.flatten())
        
        processed = np.concatenate(flat).astype(np.float32)
        
        # Padding 112
        if len(processed) < 112: 
            processed = np.concatenate([processed, np.zeros(112-len(processed))])
        else: 
            processed = processed[:112]
            
        return processed

    def observation(self, obs):
        processed = self._compute_features_and_flatten(obs)
        if len(self.frames) == 0:
            for _ in range(self.n_stack): self.frames.append(processed)
        else:
            self.frames.append(processed)
        return np.concatenate(list(self.frames)).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        processed = self._compute_features_and_flatten(obs)
        for _ in range(self.n_stack): self.frames.append(processed)
        return np.concatenate(list(self.frames)).astype(np.float32), info
# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# from collections import deque
# from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations, PolarObservations

# # =============================================================================
# # 1. ACTION WRAPPER
# # =============================================================================
# class ActionConversionWrapper(gym.ActionWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.actions_list = [
#             (0.0, 1.0, 0, 0, 0, 0, 0), (-0.6, 1.0, 0, 0, 0, 0, 0), (0.6, 1.0, 0, 0, 0, 0, 0),
#             (-1.0, 1.0, 0, 1, 0, 0, 0), (1.0, 1.0, 0, 1, 0, 0, 0), (-0.5, 1.0, 0, 1, 0, 0, 0),
#             (0.5, 1.0, 0, 1, 0, 0, 0), (0.0, 0.3, 0, 0, 0, 0, 0), (-0.6, 0.3, 0, 0, 0, 0, 0),
#             (0.6, 0.3, 0, 0, 0, 0, 0), (0.0, 0.0, 1, 0, 0, 0, 0), (-1.0, 0.0, 1, 0, 0, 0, 0),
#             (1.0, 0.0, 1, 0, 0, 0, 0), (0.0, 1.0, 0, 0, 1, 0, 0), (0.0, 1.0, 0, 0, 0, 1, 0),
#         ]
#         self.action_space = spaces.Discrete(len(self.actions_list))

#     def action(self, action_idx):
#         if hasattr(action_idx, "item"): action_idx = int(action_idx.item())
#         else: action_idx = int(action_idx)
#         vals = self.actions_list[action_idx]
#         return {
#             'steer': np.array([vals[0]], dtype=np.float32),
#             'acceleration': np.array([vals[1]], dtype=np.float32),
#             'brake': vals[2], 'drift': vals[3], 'fire': vals[4], 'nitro': vals[5], 'rescue': vals[6]
#         }

# # =============================================================================
# # 2. ULTRA WRAPPER (CORRIGÉ POUR TUPLES)
# # =============================================================================
# class UltraWrapper(gym.ObservationWrapper):
#     def __init__(self, env, n_stack=4):
#         # Initialisation interne des wrappers pystk
#         env = ConstantSizedObservations(env, state_items=5, state_karts=5, state_paths=5)
#         env = PolarObservations(env)
        
#         super().__init__(env)
#         self.n_stack = n_stack
#         self.frames = deque(maxlen=self.n_stack)
        
#         self.keys_whitelist = sorted([
#             'attachment', 'attachment_time_left', 'aux_ticks', 'center_path', 
#             'center_path_distance', 'distance_down_track', 'energy', 'front', 
#             'items_position', 'items_type', 'jumping', 'karts_position', 
#             'max_steer_angle', 'paths_distance', 'paths_end', 'paths_start', 
#             'paths_width', 'phase', 'powerup', 'shield_time', 'skeed_factor', 
#             'velocity',
#             'feat_speed', 'feat_dist_center', 'feat_angle_center', 'feat_future_risk',
#             'feat_lookahead_angle', 'feat_curve_intensity', 'feat_skeed', 'feat_air',
#             'feat_item_angle', 'feat_item_detected', 'feat_off_track'
#         ])
        
#         self.single_frame_size = 112
#         self.target_size = self.single_frame_size * n_stack
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(self.target_size,), dtype=np.float32
#         )

#     def _process_frame(self, obs):
#         # --- Feature Engineering ---
#         # Note: on convertit tout en array pour éviter les erreurs de type (Tuple)
#         velocity_vec = np.array(obs.get('velocity', [0,0,0]), dtype=np.float32)
#         speed = np.linalg.norm(velocity_vec)
        
#         center_vec = np.array(obs.get('center_path', [0,0,0]), dtype=np.float32)
#         dist_center = np.linalg.norm(center_vec)
#         angle_center = np.arctan2(center_vec[0], center_vec[2]) if center_vec.shape[0] >= 3 and center_vec[2] != 0 else 0.0

#         raw_widths = obs.get('paths_width', [])
#         # Ta correction : on utilise len() car ça peut être un tuple/list
#         safe_width = np.min(raw_widths) if len(raw_widths) > 0 else 10.0
#         limit_dist = safe_width / 2.0
        
#         future_risk = 0.0
#         if limit_dist > 0:
#             current_lateral_pos = -center_vec[0]
#             future_pos = current_lateral_pos + (velocity_vec[0] * 0.5)
#             future_risk = abs(future_pos) / limit_dist

#         paths_end = obs.get('paths_end', [])
#         lookahead_angle = 0.0
#         if len(paths_end) > 3:
#             # Conversion explicite en array pour accès par index
#             target = np.array(paths_end[2], dtype=np.float32)
#             lookahead_angle = np.arctan2(target[0], target[2])
#         curve_intensity = lookahead_angle - angle_center

#         skeed = float(obs.get('skeed_factor', 0.0))
#         is_airborne = float(obs.get('jumping', 0))
        
#         # Items logic
#         item_angle = 0.0
#         has_item = 0.0
#         items = obs.get('items_position', [])
        
#         if len(items) > 0:
#             items_arr = np.array(items, dtype=np.float32)
#             dists = np.linalg.norm(items_arr, axis=1)
#             for i in range(len(items_arr)):
#                 if items_arr[i][2] > 0 and dists[i] < 20.0:
#                     item_angle = np.arctan2(items_arr[i][0], items_arr[i][2])
#                     has_item = 1.0
#                     break

#         obs['feat_speed'] = np.array([speed], dtype=np.float32)
#         obs['feat_dist_center'] = np.array([dist_center], dtype=np.float32)
#         obs['feat_angle_center'] = np.array([angle_center], dtype=np.float32)
#         obs['feat_future_risk'] = np.array([future_risk], dtype=np.float32)
#         obs['feat_lookahead_angle'] = np.array([lookahead_angle], dtype=np.float32)
#         obs['feat_curve_intensity'] = np.array([curve_intensity], dtype=np.float32)
#         obs['feat_skeed'] = np.array([skeed], dtype=np.float32)
#         obs['feat_air'] = np.array([is_airborne], dtype=np.float32)
#         obs['feat_item_angle'] = np.array([item_angle], dtype=np.float32)
#         obs['feat_item_detected'] = np.array([has_item], dtype=np.float32)
#         obs['feat_off_track'] = np.array([1.0 if dist_center > limit_dist else 0.0], dtype=np.float32)

#         # --- Flattening Whitelist (CORRECTION CRITIQUE TUPLE) ---
#         flat = []
#         for k in self.keys_whitelist:
#             val = obs.get(k, np.zeros(1))
            
#             # ON FORCE LA CONVERSION EN NUMPY ARRAY
#             # Cela gère : int, float, list, tuple, ndarray d'un coup
#             val_arr = np.array(val, dtype=np.float32)
            
#             if val_arr.ndim == 0:
#                 flat.append([val_arr])
#             else:
#                 flat.append(val_arr.flatten())
        
#         processed = np.concatenate(flat).astype(np.float32)
        
#         # Padding
#         if len(processed) < 112: 
#             processed = np.concatenate([processed, np.zeros(112-len(processed))])
#         else: 
#             processed = processed[:112]
            
#         return processed

#     def observation(self, obs):
#         processed = self._process_frame(obs)
#         if len(self.frames) == 0:
#             for _ in range(self.n_stack): self.frames.append(processed)
#         else:
#             self.frames.append(processed)
#         return np.concatenate(list(self.frames)).astype(np.float32)

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         self.frames.clear()
        
#         # Appel direct à observation pour initialiser la stack
#         # (observation() gère le cas len==0 en remplissant tout)
#         return self.observation(obs), info
# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# from collections import deque
# from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations, PolarObservations

# # =============================================================================
# # 1. ACTION WRAPPER (Strictement Action)
# # =============================================================================
# class MyActionWrapper(gym.ActionWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.actions_list = [
#             (0.0, 1.0, 0, 0, 0, 0, 0), (-0.6, 1.0, 0, 0, 0, 0, 0), (0.6, 1.0, 0, 0, 0, 0, 0),
#             (-1.0, 1.0, 0, 1, 0, 0, 0), (1.0, 1.0, 0, 1, 0, 0, 0), (-0.5, 1.0, 0, 1, 0, 0, 0),
#             (0.5, 1.0, 0, 1, 0, 0, 0), (0.0, 0.3, 0, 0, 0, 0, 0), (-0.6, 0.3, 0, 0, 0, 0, 0),
#             (0.6, 0.3, 0, 0, 0, 0, 0), (0.0, 0.0, 1, 0, 0, 0, 0), (-1.0, 0.0, 1, 0, 0, 0, 0),
#             (1.0, 0.0, 1, 0, 0, 0, 0), (0.0, 1.0, 0, 0, 1, 0, 0), (0.0, 1.0, 0, 0, 0, 1, 0),
#         ]
#         self.action_space = spaces.Discrete(len(self.actions_list))

#     def action(self, action_idx):
#         if hasattr(action_idx, "item"): action_idx = int(action_idx.item())
#         else: action_idx = int(action_idx)
#         vals = self.actions_list[action_idx]
#         return {
#             'steer': np.array([vals[0]], dtype=np.float32),
#             'acceleration': np.array([vals[1]], dtype=np.float32),
#             'brake': vals[2], 'drift': vals[3], 'fire': vals[4], 'nitro': vals[5], 'rescue': vals[6]
#         }

# # =============================================================================
# # 2. OBSERVATION WRAPPER (Strictement Obs + Pystk logic)
# # =============================================================================
# class MyObservationWrapper(gym.ObservationWrapper):
#     def __init__(self, env, n_stack=4):
#         # STRATÉGIE CLÉ : On applique les wrappers Pystk ICI, en interne.
#         # Comme ça, 'env' devient un environnement qui sort déjà du Polaire/Constant.
#         env = ConstantSizedObservations(env, state_items=5, state_karts=5, state_paths=5)
#         env = PolarObservations(env)
        
#         super().__init__(env)
        
#         self.n_stack = n_stack
#         self.frames = deque(maxlen=self.n_stack)
        
#         # Whitelist 112 dims
#         self.keys_whitelist = sorted([
#             'attachment', 'attachment_time_left', 'aux_ticks', 'center_path', 
#             'center_path_distance', 'distance_down_track', 'energy', 'front', 
#             'items_position', 'items_type', 'jumping', 'karts_position', 
#             'max_steer_angle', 'paths_distance', 'paths_end', 'paths_start', 
#             'paths_width', 'phase', 'powerup', 'shield_time', 'skeed_factor', 
#             'velocity',
#             'feat_speed', 'feat_dist_center', 'feat_angle_center', 'feat_future_risk',
#             'feat_lookahead_angle', 'feat_curve_intensity', 'feat_skeed', 'feat_air',
#             'feat_item_angle', 'feat_item_detected', 'feat_off_track'
#         ])
        
#         self.single_frame_size = 112
#         self.target_size = self.single_frame_size * n_stack # 448
        
#         # C'est ÇA qui dit à BBRL "Arrête de chercher un dictionnaire, je suis un tableau"
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(self.target_size,), dtype=np.float32
#         )

#     def _process(self, obs):
#         # 1. Feature Engineering (Sur données déjà Polaires)
#         velocity_vec = obs.get('velocity', np.zeros(3))
#         speed = np.linalg.norm(velocity_vec)
#         center_vec = obs.get('center_path', np.zeros(3))
#         dist_center = np.linalg.norm(center_vec)
#         # En polaire, center_path est relatif, l'angle est implicite.
#         # Mais calculons le quand même si possible.
#         angle_center = np.arctan2(center_vec[0], center_vec[2]) if center_vec.shape[0] >= 3 and center_vec[2] != 0 else 0.0

#         raw_widths = obs.get('paths_width', np.array([10.0]))
#         safe_width = np.min(raw_widths) if raw_widths.size > 0 else 10.0
#         limit_dist = safe_width / 2.0
        
#         future_risk = 0.0
#         if limit_dist > 0:
#             current_lateral_pos = -center_vec[0]
#             future_pos = current_lateral_pos + (velocity_vec[0] * 0.5)
#             future_risk = abs(future_pos) / limit_dist

#         paths_end = obs.get('paths_end', [])
#         lookahead_angle = 0.0
#         if len(paths_end) > 3:
#             lookahead_angle = np.arctan2(paths_end[2][0], paths_end[2][2])
#         curve_intensity = lookahead_angle - angle_center

#         skeed = float(obs.get('skeed_factor', 0.0))
#         is_airborne = float(obs.get('jumping', 0))
        
#         items = obs.get('items_position', [])
#         ia, idet = 0.0, 0.0
#         if len(items) > 0:
#             dists = np.linalg.norm(items, axis=1)
#             # Find closest valid item
#             for i in range(len(items)):
#                 if items[i][2] > 0 and dists[i] < 20.0:
#                     ia, idet = np.arctan2(items[i][0], items[i][2]), 1.0
#                     break

#         obs['feat_speed'] = np.array([speed], dtype=np.float32)
#         obs['feat_dist_center'] = np.array([dist_center], dtype=np.float32)
#         obs['feat_angle_center'] = np.array([angle_center], dtype=np.float32)
#         obs['feat_future_risk'] = np.array([future_risk], dtype=np.float32)
#         obs['feat_lookahead_angle'] = np.array([lookahead_angle], dtype=np.float32)
#         obs['feat_curve_intensity'] = np.array([curve_intensity], dtype=np.float32)
#         obs['feat_skeed'] = np.array([skeed], dtype=np.float32)
#         obs['feat_air'] = np.array([is_airborne], dtype=np.float32)
#         obs['feat_item_angle'] = np.array([ia], dtype=np.float32)
#         obs['feat_item_detected'] = np.array([idet], dtype=np.float32)
#         obs['feat_off_track'] = np.array([1.0 if dist_center > limit_dist else 0.0], dtype=np.float32)

#         # 2. Flatten Whitelist
#         flat = []
#         for k in self.keys_whitelist:
#             val = obs.get(k, np.zeros(1))
#             if isinstance(val, (int, float, np.number)): flat.append([val])
#             else: flat.append(val.flatten())
        
#         processed = np.concatenate(flat).astype(np.float32)
        
#         # 3. Padding Force 112
#         if len(processed) < 112: 
#             processed = np.concatenate([processed, np.zeros(112-len(processed))])
#         else: 
#             processed = processed[:112]
            
#         return processed

#     def observation(self, obs):
#         processed = self._process(obs)
#         if len(self.frames) == 0:
#             for _ in range(self.n_stack): self.frames.append(processed)
#         else:
#             self.frames.append(processed)
#         return np.concatenate(list(self.frames)).astype(np.float32)

#     def reset(self, **kwargs):
#         # Au reset, on vide et on remplit 4 fois
#         obs, info = self.env.reset(**kwargs)
#         self.frames.clear()
        
#         processed = self._process(obs)
#         for _ in range(self.n_stack):
#             self.frames.append(processed)
            
#         return np.concatenate(list(self.frames)).astype(np.float32), info

# # import gymnasium as gym
# # from gymnasium import spaces
# # import numpy as np
# # from collections import deque
# # from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations, PolarObservations

# # class ActionConversionWrapper(gym.ActionWrapper):
# #     def __init__(self, env):
# #         super().__init__(env)
# #         self.actions_list = [
# #             (0.0, 1.0, 0, 0, 0, 0, 0), (-0.6, 1.0, 0, 0, 0, 0, 0), (0.6, 1.0, 0, 0, 0, 0, 0),
# #             (-1.0, 1.0, 0, 1, 0, 0, 0), (1.0, 1.0, 0, 1, 0, 0, 0), (-0.5, 1.0, 0, 1, 0, 0, 0),
# #             (0.5, 1.0, 0, 1, 0, 0, 0), (0.0, 0.3, 0, 0, 0, 0, 0), (-0.6, 0.3, 0, 0, 0, 0, 0),
# #             (0.6, 0.3, 0, 0, 0, 0, 0), (0.0, 0.0, 1, 0, 0, 0, 0), (-1.0, 0.0, 1, 0, 0, 0, 0),
# #             (1.0, 0.0, 1, 0, 0, 0, 0), (0.0, 1.0, 0, 0, 1, 0, 0), (0.0, 1.0, 0, 0, 0, 1, 0),
# #         ]
# #         self.action_space = spaces.Discrete(len(self.actions_list))

# #     def action(self, action_idx):
# #         if hasattr(action_idx, "item"): action_idx = int(action_idx.item())
# #         else: action_idx = int(action_idx)
# #         vals = self.actions_list[action_idx]
# #         return {
# #             'steer': np.array([vals[0]], dtype=np.float32),
# #             'acceleration': np.array([vals[1]], dtype=np.float32),
# #             'brake': vals[2], 'drift': vals[3], 'fire': vals[4], 'nitro': vals[5], 'rescue': vals[6]
# #         }

# # class UltraWrapper(gym.ObservationWrapper):
# #     def __init__(self, env, n_stack=4):
# #         # 1. On enveloppe l'environnement reçu avec les wrappers pystk
# #         # Cela transforme l'input multi-full en format simple-like (Polaire + Taille fixe)
# #         env = ConstantSizedObservations(env, state_items=5, state_karts=5, state_paths=5)
# #         env = PolarObservations(env)
        
# #         super().__init__(env)
# #         self.n_stack = n_stack
# #         self.frames = deque(maxlen=self.n_stack)
        
# #         self.keys_whitelist = sorted([
# #             'attachment', 'attachment_time_left', 'aux_ticks', 'center_path', 
# #             'center_path_distance', 'distance_down_track', 'energy', 'front', 
# #             'items_position', 'items_type', 'jumping', 'karts_position', 
# #             'max_steer_angle', 'paths_distance', 'paths_end', 'paths_start', 
# #             'paths_width', 'phase', 'powerup', 'shield_time', 'skeed_factor', 
# #             'velocity',
# #             'feat_speed', 'feat_dist_center', 'feat_angle_center', 'feat_future_risk',
# #             'feat_lookahead_angle', 'feat_curve_intensity', 'feat_skeed', 'feat_air',
# #             'feat_item_angle', 'feat_item_detected', 'feat_off_track'
# #         ])
        
# #         self.single_frame_size = 112
# #         self.target_size = self.single_frame_size * n_stack
# #         self.observation_space = spaces.Box(
# #             low=-np.inf, high=np.inf, shape=(self.target_size,), dtype=np.float32
# #         )

# #     def _compute_features_and_flatten(self, obs):
# #         # --- Feature Engineering ---
# #         velocity_vec = obs.get('velocity', np.zeros(3))
# #         speed = np.linalg.norm(velocity_vec)
# #         center_vec = obs.get('center_path', np.zeros(3))
# #         dist_center = np.linalg.norm(center_vec)
# #         angle_center = np.arctan2(center_vec[0], center_vec[2]) if center_vec.shape[0] >= 3 and center_vec[2] != 0 else 0.0

# #         raw_widths = obs.get('paths_width', np.array([10.0]))
# #         safe_width = np.min(raw_widths) if raw_widths.size > 0 else 10.0
# #         limit_dist = safe_width / 2.0
        
# #         future_risk = 0.0
# #         if limit_dist > 0:
# #             current_lateral_pos = -center_vec[0]
# #             future_pos = current_lateral_pos + (velocity_vec[0] * 0.5)
# #             future_risk = abs(future_pos) / limit_dist

# #         paths_end = obs.get('paths_end', [])
# #         lookahead_angle = 0.0
# #         if len(paths_end) > 3:
# #             target = paths_end[2]
# #             lookahead_angle = np.arctan2(target[0], target[2])
# #         curve_intensity = lookahead_angle - angle_center

# #         skeed = float(obs.get('skeed_factor', 0.0))
# #         is_airborne = float(obs.get('jumping', 0))
        
# #         # --- CORRECTION VARIABLE UNDEFINED ---
# #         # On initialise AVANT le if pour être sûr qu'elles existent
# #         item_angle = 0.0
# #         has_item = 0.0
        
# #         items = obs.get('items_position', [])
# #         if len(items) > 0:
# #             dists = np.linalg.norm(items, axis=1)
# #             # On cherche l'item valide le plus proche
# #             for i in range(len(items)):
# #                 # Si item devant et proche
# #                 if items[i][2] > 0 and dists[i] < 20.0:
# #                     item_angle = np.arctan2(items[i][0], items[i][2])
# #                     has_item = 1.0
# #                     break # On prend le premier trouvé
                    
# #         # Injection
# #         obs['feat_speed'] = np.array([speed], dtype=np.float32)
# #         obs['feat_dist_center'] = np.array([dist_center], dtype=np.float32)
# #         obs['feat_angle_center'] = np.array([angle_center], dtype=np.float32)
# #         obs['feat_future_risk'] = np.array([future_risk], dtype=np.float32)
# #         obs['feat_lookahead_angle'] = np.array([lookahead_angle], dtype=np.float32)
# #         obs['feat_curve_intensity'] = np.array([curve_intensity], dtype=np.float32)
# #         obs['feat_skeed'] = np.array([skeed], dtype=np.float32)
# #         obs['feat_air'] = np.array([is_airborne], dtype=np.float32)
        
# #         # Utilisation des variables corrigées
# #         obs['feat_item_angle'] = np.array([item_angle], dtype=np.float32)
# #         obs['feat_item_detected'] = np.array([has_item], dtype=np.float32)
        
# #         obs['feat_off_track'] = np.array([1.0 if dist_center > limit_dist else 0.0], dtype=np.float32)

# #         # --- Flattening Whitelist ---
# #         flat_list = []
# #         for key in self.keys_whitelist:
# #             val = obs.get(key, np.zeros(1))
# #             if isinstance(val, (int, float, np.number)): flat_list.append([val])
# #             elif isinstance(val, np.ndarray): flat_list.append(val.flatten())
        
# #         processed = np.concatenate(flat_list).astype(np.float32)
        
# #         # Padding
# #         if len(processed) < 112: 
# #             processed = np.concatenate([processed, np.zeros(112-len(processed))])
# #         else: 
# #             processed = processed[:112]
            
# #         return processed

# #     def observation(self, obs):
# #         """
# #         Logique de Stacking Automatique.
# #         Si la mémoire est vide (ce qui arrive au tout début après un reset), on la remplit.
# #         """
# #         processed_obs = self._compute_features_and_flatten(obs)
        
# #         if len(self.frames) == 0:
# #             # CAS RESET : On remplit tout le buffer avec la première image
# #             for _ in range(self.n_stack):
# #                 self.frames.append(processed_obs)
# #         else:
# #             # CAS STEP : On ajoute juste la nouvelle image
# #             self.frames.append(processed_obs)
            
# #         return np.concatenate(list(self.frames)).astype(np.float32)

# #     def reset(self, **kwargs):
# #         # On vide la mémoire pour forcer le remplissage dans observation()
# #         self.frames.clear()
        
# #         # L'appel à super().reset() va déclencher self.observation() automatiquement
# #         # grâce à l'implémentation de Gym ObservationWrapper
# #         return super().reset(**kwargs)
# # import gymnasium as gym
# # from gymnasium import spaces
# # import numpy as np
# # from collections import deque

# # class UltraWrapper(gym.ObservationWrapper):
# #     """
# #     Ce wrapper fait TOUT :
# #     1. Calcule les features.
# #     2. Filtre et Aplatit (Whitelist -> 112 dims).
# #     3. Empile 4 frames (Stacking -> 448 dims).
# #     """
# #     def __init__(self, env):
# #         super().__init__(env)
        
# #         self.n_stack = 4
# #         self.frames = deque(maxlen=self.n_stack)
        
# #         # --- CONFIGURATION WHITELIST (112 dims) ---
# #         self.keys_whitelist = sorted([
# #             'attachment', 'attachment_time_left', 'aux_ticks', 'center_path', 
# #             'center_path_distance', 'distance_down_track', 'energy', 'front', 
# #             'items_position', 'items_type', 'jumping', 'karts_position', 
# #             'max_steer_angle', 'paths_distance', 'paths_end', 'paths_start', 
# #             'paths_width', 'phase', 'powerup', 'shield_time', 'skeed_factor', 
# #             'velocity',
# #             'feat_speed', 'feat_dist_center', 'feat_angle_center', 'feat_future_risk',
# #             'feat_lookahead_angle', 'feat_curve_intensity', 'feat_skeed', 'feat_air',
# #             'feat_item_angle', 'feat_item_detected', 'feat_off_track'
# #         ])
# #         self.single_frame_size = 112
        
# #         # --- OUTPUT SPACE (448 dims) ---
# #         self.target_size = self.single_frame_size * self.n_stack
# #         self.observation_space = spaces.Box(
# #             low=-np.inf, high=np.inf, shape=(self.target_size,), dtype=np.float32
# #         )

# #     def observation(self, obs):
# #         """
# #         Cette méthode est appelée par Gym. 
# #         Pour respecter le contrat ObservationWrapper, elle doit retourner l'observation FINALE (448).
# #         Donc elle doit gérer le stacking ici-même.
# #         """
# #         # 1. Calcul des Features (Feature Engineering)
# #         processed_frame = self._compute_features_and_flatten(obs)
        
# #         # 2. Gestion du Stacking
# #         # Astuce : Si c'est un reset (deque vide), on remplit tout.
# #         # Sinon (step), on ajoute juste une frame.
# #         if len(self.frames) == 0:
# #             for _ in range(self.n_stack):
# #                 self.frames.append(processed_frame)
# #         else:
# #             self.frames.append(processed_frame)
            
# #         # 3. Retour du Stack (448)
# #         return np.concatenate(list(self.frames)).astype(np.float32)

# #     def reset(self, **kwargs):
# #         # On vide la mémoire avant le reset
# #         self.frames.clear()
# #         obs, info = self.env.reset(**kwargs)
# #         # Gym appellera self.observation(obs) automatiquement ou on le fait ici
# #         return self.observation(obs), info

# #     def _compute_features_and_flatten(self, obs):
# #         # --- A. Feature Engineering (Copie de ton code) ---
# #         velocity_vec = obs.get('velocity', np.zeros(3))
# #         speed = np.linalg.norm(velocity_vec)
# #         center_vec = obs.get('center_path', np.zeros(3))
# #         dist_center = np.linalg.norm(center_vec)
# #         angle_center = 0.0
# #         if center_vec.shape[0] >= 3 and center_vec[2] != 0:
# #             angle_center = np.arctan2(center_vec[0], center_vec[2])

# #         raw_widths = obs.get('paths_width', np.array([10.0]))
# #         safe_width = np.min(raw_widths) if len(raw_widths) > 0 else 10.0
# #         limit_dist = safe_width / 2.0
        
# #         future_risk = 0.0
# #         if limit_dist > 0:
# #             current_lateral_pos = -center_vec[0]
# #             future_pos = current_lateral_pos + (velocity_vec[0] * 0.5)
# #             future_risk = abs(future_pos) / limit_dist

# #         paths_end = obs.get('paths_end', [])
# #         lookahead_angle = 0.0
# #         if len(paths_end) > 3:
# #             target = paths_end[2]
# #             lookahead_angle = np.arctan2(target[0], target[2])
# #         curve_intensity = lookahead_angle - angle_center

# #         skeed = float(obs.get('skeed_factor', 0.0))
# #         is_airborne = float(obs.get('jumping', 0))
        
# #         items = obs.get('items_position', [])
# #         item_angle = 0.0
# #         has_item = 0.0
# #         if len(items) > 0:
# #             dists = np.linalg.norm(items, axis=1)
# #             min_idx = np.argmin(dists)
# #             if items[min_idx][2] > 0 and dists[min_idx] < 20.0:
# #                 item_angle = np.arctan2(items[min_idx][0], items[min_idx][2])
# #                 has_item = 1.0

# #         # On modifie le dict (ou une copie)
# #         obs['feat_speed'] = np.array([speed], dtype=np.float32)
# #         obs['feat_dist_center'] = np.array([dist_center], dtype=np.float32)
# #         obs['feat_angle_center'] = np.array([angle_center], dtype=np.float32)
# #         obs['feat_future_risk'] = np.array([future_risk], dtype=np.float32)
# #         obs['feat_lookahead_angle'] = np.array([lookahead_angle], dtype=np.float32)
# #         obs['feat_curve_intensity'] = np.array([curve_intensity], dtype=np.float32)
# #         obs['feat_skeed'] = np.array([skeed], dtype=np.float32)
# #         obs['feat_air'] = np.array([is_airborne], dtype=np.float32)
# #         obs['feat_item_angle'] = np.array([item_angle], dtype=np.float32)
# #         obs['feat_item_detected'] = np.array([has_item], dtype=np.float32)
# #         obs['feat_off_track'] = np.array([1.0 if dist_center > limit_dist else 0.0], dtype=np.float32)

# #         # --- B. Flatten avec Whitelist ---
# #         flat_list = []
# #         for key in self.keys_whitelist:
# #             if key in obs:
# #                 val = obs[key]
# #                 if isinstance(val, (int, float, np.number)): 
# #                     flat_list.append([val])
# #                 elif isinstance(val, np.ndarray): 
# #                     flat_list.append(val.flatten())
        
# #         flat_obs = np.concatenate(flat_list).astype(np.float32)
        
# #         # --- C. Padding Force 112 ---
# #         if len(flat_obs) < self.single_frame_size:
# #              padding = np.zeros(self.single_frame_size - len(flat_obs), dtype=np.float32)
# #              flat_obs = np.concatenate([flat_obs, padding])
# #         elif len(flat_obs) > self.single_frame_size:
# #              flat_obs = flat_obs[:self.single_frame_size]
             
# #         return flat_obs

# # class ActionConversionWrapper(gym.ActionWrapper):
# #     def __init__(self, env):
# #         super().__init__(env)
# #         self.actions_list = [
# #             (0.0, 1.0, 0, 0, 0, 0, 0), (-0.6, 1.0, 0, 0, 0, 0, 0), (0.6, 1.0, 0, 0, 0, 0, 0),
# #             (-1.0, 1.0, 0, 1, 0, 0, 0), (1.0, 1.0, 0, 1, 0, 0, 0), (-0.5, 1.0, 0, 1, 0, 0, 0),
# #             (0.5, 1.0, 0, 1, 0, 0, 0), (0.0, 0.3, 0, 0, 0, 0, 0), (-0.6, 0.3, 0, 0, 0, 0, 0),
# #             (0.6, 0.3, 0, 0, 0, 0, 0), (0.0, 0.0, 1, 0, 0, 0, 0), (-1.0, 0.0, 1, 0, 0, 0, 0),
# #             (1.0, 0.0, 1, 0, 0, 0, 0), (0.0, 1.0, 0, 0, 1, 0, 0), (0.0, 1.0, 0, 0, 0, 1, 0),
# #         ]
# #         self.action_space = spaces.Discrete(len(self.actions_list))

# #     def action(self, action_idx):
# #         if hasattr(action_idx, "item"): action_idx = int(action_idx.item())
# #         else: action_idx = int(action_idx)
# #         vals = self.actions_list[action_idx]
# #         return {
# #             'steer': np.array([vals[0]], dtype=np.float32),
# #             'acceleration': np.array([vals[1]], dtype=np.float32),
# #             'brake': vals[2], 'drift': vals[3], 'fire': vals[4], 'nitro': vals[5], 'rescue': vals[6]
# #         }
        

# # 5# import copy
# # # import logging
# # # import sys
# # # from typing import Any, Dict, Tuple, TypedDict, Optional

# # # import gymnasium as gym
# # # import numpy as np
# # # import pystk2
# # # from gymnasium import spaces

# # # from pystk2_gymnasium.envs import STKAction, STKRaceEnv
# # # from pystk2_gymnasium.definitions import ActionObservationWrapper
# # # from pystk2_gymnasium.utils import Discretizer, max_enum_value



# # class STKAction(TypedDict):
# #     # :> Acceleration, between 0 and 1
# #     acceleration: float
# #     # :> Steering, between -1 and 1 (but limited by max_steer)
# #     steering: float
# #     brake: bool
# #     drift: bool
# #     nitro: bool
# #     rescue: bool


# # def get_action(action: STKAction):
# #     return pystk2.Action(
# #         brake=int(action["brake"]) > 0,
# #         nitro=int(action["nitro"] > 0),
# #         drift=int(action["drift"] > 0),
# #         rescue=int(action["rescue"] > 0),
# #         fire=int(action["fire"] > 0),
# #         steer=float(action["steer"]),
# #         acceleration=float(action["acceleration"]),
# #     )

# # class STKRaceEnv(BaseSTKRaceEnv):
# #     """Single player race environment"""

# #     #: Use AI
# #     spec: AgentSpec

# #     def __init__(self, *, agent: Optional[AgentSpec] = None, **kwargs):
# #         """Creates a new race

# #         :param spec: Agent spec
# #         :param kwargs: General parameters, see BaseSTKRaceEnv
# #         """
# #         super().__init__(**kwargs)

# #         # Setup the variables
# #         self.agent = agent if agent is not None else AgentSpec()

# #         # Those will be set when the race is setup
# #         self.kart_ix = None

# #         # We have 4 actions, corresponding to "right", "up", "left", "down"
# #         self.action_space = kart_action_space()
# #         self.observation_space = kart_observation_space(self.agent.use_ai)

# #     def reset(
# #         self,
# #         *,
# #         seed: Optional[int] = None,
# #         options: Optional[Dict[str, Any]] = None,
# #     ) -> Tuple[pystk2.WorldState, Dict[str, Any]]:
# #         random = np.random.RandomState(seed)

# #         super().reset_race(random, options=options)

# #         # Set the controlled kart position (if any)
# #         self.kart_ix = self.agent.rank_start
# #         if self.kart_ix is None:
# #             self.kart_ix = np.random.randint(0, self.num_kart)
# #         logging.debug("Observed kart index %d", self.kart_ix)

# #         # Camera setup
# #         self.config.players[self.kart_ix].camera_mode = (
# #             pystk2.PlayerConfig.CameraMode.ON
# #         )
# #         self.config.players[self.kart_ix].name = self.agent.name

# #         if not self.agent.use_ai:
# #             self.config.players[self.kart_ix].controller = (
# #                 pystk2.PlayerConfig.Controller.PLAYER_CONTROL
# #             )

# #         self.warmup_race()
# #         self.world_update(False)

# #         return self.get_observation(self.kart_ix, self.agent.use_ai), {}

# #     def step(
# #         self, action: STKAction
# #     ) -> Tuple[pystk2.WorldState, float, bool, bool, Dict[str, Any]]:
# #         if self.agent.use_ai:
# #             self.race_step()
# #         else:
# #             self.race_step(get_action(action))

# #         self.world_update()

# #         obs, reward, terminated, info = self.get_state(self.kart_ix, self.agent.use_ai)

# #         return (obs, reward, terminated, False, info)


# # 5# import gymnasium as gym
# # # from gymnasium import spaces
# # # import numpy as np
# # # import copy
# # # import logging
# # # import sys

# # # import gymnasium as gym
# # # from gymnasium import spaces
# # # import numpy as np
# # # import copy
# # # import logging
# # # import sys

# # # import gymnasium as gym
# # # from gymnasium import spaces
# # # import numpy as np
# # # import copy
# # # import logging
# # # import sys

# # Standard STK imports (assumed based on your code)
# # from pystk2_gymnasium.stk_env import STKRaceEnv
# # import pystk2

# # 5# import gymnasium as gym
# # # from gymnasium import spaces
# # # import numpy as np
# # # import copy
# # # import logging
# # # import sys

# # # # The README indicates this wrapper is used for 'supertuxkart/simple-v0'.
# # # class ConstantSizedObservationsNew(gym.ObservationWrapper):
# # #     def __init__(
# # #         self,
# # #         env: gym.Env,
# # #         *,
# # #         state_items=5,
# # #         state_karts=5,
# # #         state_paths=5,
# # #         add_mask=False,
# # #         **kwargs,
# # #     ):
# # #         """A simpler race environment with fixed width data"""
# # #         super().__init__(env, **kwargs)
        
# # #         # 1. Engine Optimization
# # #         # Limit paths calculation to save performance.
# # #         unwrapped = env.unwrapped
# # #         if hasattr(unwrapped, 'max_paths') and unwrapped.max_paths is None:
# # #             logging.info("Setting unwrapped environment max_paths to %d", state_paths)
# # #             unwrapped.max_paths = min(
# # #                 state_paths, unwrapped.max_paths or sys.maxsize
# # #             )

# # #         self.state_items = state_items
# # #         self.state_karts = state_karts
# # #         self.state_paths = state_paths

# # #         # 2. Observation Space Transformation
# # #         # We transform 'Sequence' spaces into fixed-size 'Box' or 'MultiDiscrete' spaces.
# # #         self._observation_space = space = copy.deepcopy(self.env.observation_space)

# # #         # Path observations (Distance, Width, Start, End)
# # #         space["paths_distance"] = spaces.Box(0, float("inf"), shape=(self.state_paths, 2), dtype=np.float32)
# # #         space["paths_width"] = spaces.Box(0, float("inf"), shape=(self.state_paths, 1), dtype=np.float32)
# # #         space["paths_start"] = spaces.Box(-float("inf"), float("inf"), shape=(self.state_paths, 3), dtype=np.float32)
# # #         space["paths_end"] = spaces.Box(-float("inf"), float("inf"), shape=(self.state_paths, 3), dtype=np.float32)
        
# # #         # Item and Kart positions
# # #         space["items_position"] = spaces.Box(-float("inf"), float("inf"), shape=(self.state_items, 3), dtype=np.float32)
# # #         space["karts_position"] = spaces.Box(-float("inf"), float("inf"), shape=(self.state_karts, 3))

# # #         # Based on your obs.keys, items_type is Discrete(7).
# # #         n_item_types = 7 
# # #         space["items_type"] = spaces.MultiDiscrete([n_item_types for _ in range(self.state_items)])

# # #         # 3. Masking Logic
# # #         # Adds binary flags to indicate if a data slot is real or padded.
# # #         self.add_mask = add_mask
# # #         if add_mask:
# # #             space["paths_mask"] = spaces.Box(0, 1, shape=(self.state_paths,), dtype=np.int8)
# # #             space["items_mask"] = spaces.Box(0, 1, shape=(self.state_items,), dtype=np.int8)
# # #             space["karts_mask"] = spaces.Box(0, 1, shape=(self.state_karts,), dtype=np.int8)

# # #     def make_tensor(self, state, name: str, default_value=0):
# # #         """Standardizes variable length data from Sequence keys."""
# # #         value = state[name]
# # #         space = self.observation_space[name]

# # #         # Convert the list of arrays (Sequence) into a single NumPy array
# # #         value = np.stack(value) if len(value) > 0 else np.empty((0,) + space.shape[1:], dtype=np.float32)
        
# # #         # Verify that the inner dimensions (e.g., 3 for vectors) match the space.
# # #         assert (
# # #             space.shape[1:] == value.shape[1:]
# # #         ), f"Shape mismatch for {name}: {space.shape} vs {value.shape}"

# # #         delta = space.shape[0] - value.shape[0]
# # #         if delta > 0:
# # #             # PADDING: Fill the remaining slots with zeros or default_value
# # #             shape = [delta] + list(space.shape[1:])
            
# # #             # FIX: Use getattr to handle MultiDiscrete which does not have .dtype
# # #             target_dtype = getattr(space, 'dtype', value.dtype)
            
# # #             value = np.concatenate(
# # #                 [value, np.full(shape, default_value, dtype=target_dtype)], axis=0
# # #             )
# # #         elif delta < 0:
# # #             # TRUNCATION: Cut data that exceeds the fixed width.
# # #             value = value[:space.shape[0]]

# # #         # Final check to ensure output exactly matches the redefined Gymnasium space
# # #         assert (
# # #             space.shape == value.shape
# # #         ), f"Shape mismatch for {name}: {space.shape} vs {value.shape}"
# # #         state[name] = value

# # #     def observation(self, state):
# # #         """Processes the raw dictionary from the environment."""
# # #         state = {**state} # Shallow copy to preserve original data

# # #         # Mask helper function
# # #         def create_mask(length: int, size: int):
# # #             v = np.zeros((size,), dtype=np.int8)
# # #             v[:min(length, size)] = 1
# # #             return v

# # #         if self.add_mask:
# # #             state["paths_mask"] = create_mask(len(state.get("paths_width", [])), self.state_paths)
# # #             state["items_mask"] = create_mask(len(state.get("items_type", [])), self.state_items)
# # #             state["karts_mask"] = create_mask(len(state.get("karts_position", [])), self.state_karts)

# # #         # Standardize all sequence-based keys according to the README specs.
# # #         standard_keys = [
# # #             "paths_distance", "paths_width", "paths_start", "paths_end",
# # #             "items_position", "items_type", "karts_position"
# # #         ]
        
# # #         for key in standard_keys:
# # #             if key in state:
# # #                 self.make_tensor(state, key)

# # #         return state




# # import gymnasium as gym
# # from gymnasium import spaces
# # import numpy as np
# # from collections import deque

# # # =============================================================================
# # # 1. FEATURE ENGINEERING (ObservationWrapper)
# # # =============================================================================
# # class FeatureEngineeringWrapper(gym.ObservationWrapper):
# #     def __init__(self, env):
# #         super().__init__(env)
    
# #     def observation(self, obs):
# #         return self._compute_features(obs)

# #     def _compute_features(self, obs):
# #         # Utilisation de .get() pour être robuste si une clé manque
# #         velocity_vec = obs.get('velocity', np.zeros(3))
# #         speed = np.linalg.norm(velocity_vec)
# #         center_vec = obs.get('center_path', np.zeros(3))
# #         dist_center = np.linalg.norm(center_vec)
        
# #         angle_center = 0.0
# #         if center_vec.shape[0] >= 3 and center_vec[2] != 0:
# #             angle_center = np.arctan2(center_vec[0], center_vec[2])

# #         raw_widths = obs.get('paths_width', np.array([10.0]))
# #         # Gestion sécurité si tableau vide
# #         safe_width = np.min(raw_widths) if raw_widths.size > 0 else 10.0
# #         limit_dist = safe_width / 2.0
        
# #         future_risk = 0.0
# #         if limit_dist > 0:
# #             current_lateral_pos = -center_vec[0]
# #             future_pos = current_lateral_pos + (velocity_vec[0] * 0.5)
# #             future_risk = abs(future_pos) / limit_dist

# #         paths_end = obs.get('paths_end', [])
# #         lookahead_angle = 0.0
# #         if len(paths_end) > 3:
# #             target = paths_end[2]
# #             lookahead_angle = np.arctan2(target[0], target[2])
# #         curve_intensity = lookahead_angle - angle_center

# #         skeed = float(obs.get('skeed_factor', 0.0))
# #         is_airborne = float(obs.get('jumping', 0))
        
# #         items = obs.get('items_position', [])
# #         item_angle = 0.0
# #         has_item = 0.0
# #         if len(items) > 0:
# #             dists = np.linalg.norm(items, axis=1)
# #             min_idx = np.argmin(dists)
# #             if items[min_idx][2] > 0 and dists[min_idx] < 20.0:
# #                 item_angle = np.arctan2(items[min_idx][0], items[min_idx][2])
# #                 has_item = 1.0

# #         # Injection des features
# #         obs['feat_speed'] = np.array([speed], dtype=np.float32)
# #         obs['feat_dist_center'] = np.array([dist_center], dtype=np.float32)
# #         obs['feat_angle_center'] = np.array([angle_center], dtype=np.float32)
# #         obs['feat_future_risk'] = np.array([future_risk], dtype=np.float32)
# #         obs['feat_lookahead_angle'] = np.array([lookahead_angle], dtype=np.float32)
# #         obs['feat_curve_intensity'] = np.array([curve_intensity], dtype=np.float32)
# #         obs['feat_skeed'] = np.array([skeed], dtype=np.float32)
# #         obs['feat_air'] = np.array([is_airborne], dtype=np.float32)
# #         obs['feat_item_angle'] = np.array([item_angle], dtype=np.float32)
# #         obs['feat_item_detected'] = np.array([has_item], dtype=np.float32)
# #         obs['feat_off_track'] = np.array([1.0 if dist_center > limit_dist else 0.0], dtype=np.float32)

# #         return obs

# # # =============================================================================
# # # 2. ACTION CONVERSION (ActionWrapper)
# # # =============================================================================
# # class ActionConversionWrapper(gym.ActionWrapper):
# #     def __init__(self, env):
# #         super().__init__(env)
# #         self.actions_list = [
# #             (0.0, 1.0, 0, 0, 0, 0, 0), (-0.6, 1.0, 0, 0, 0, 0, 0), (0.6, 1.0, 0, 0, 0, 0, 0),
# #             (-1.0, 1.0, 0, 1, 0, 0, 0), (1.0, 1.0, 0, 1, 0, 0, 0), (-0.5, 1.0, 0, 1, 0, 0, 0),
# #             (0.5, 1.0, 0, 1, 0, 0, 0), (0.0, 0.3, 0, 0, 0, 0, 0), (-0.6, 0.3, 0, 0, 0, 0, 0),
# #             (0.6, 0.3, 0, 0, 0, 0, 0), (0.0, 0.0, 1, 0, 0, 0, 0), (-1.0, 0.0, 1, 0, 0, 0, 0),
# #             (1.0, 0.0, 1, 0, 0, 0, 0), (0.0, 1.0, 0, 0, 1, 0, 0), (0.0, 1.0, 0, 0, 0, 1, 0),
# #         ]
# #         self.action_space = spaces.Discrete(len(self.actions_list))

# #     def action(self, action_idx):
# #         if hasattr(action_idx, "item"): action_idx = int(action_idx.item())
# #         else: action_idx = int(action_idx)
# #         vals = self.actions_list[action_idx]
# #         return {
# #             'steer': np.array([vals[0]], dtype=np.float32),
# #             'acceleration': np.array([vals[1]], dtype=np.float32),
# #             'brake': vals[2], 'drift': vals[3], 'fire': vals[4], 'nitro': vals[5], 'rescue': vals[6]
# #         }

# # # =============================================================================
# # # 3. FLATTEN WRAPPER (LA CLÉ DU SUCCÈS : WHITELIST)
# # # =============================================================================
# # class FlattenWrapper(gym.ObservationWrapper):
# #     def __init__(self, env):
# #         super().__init__(env)
        
# #         # LISTE STRICTE des clés de simple-v0 + tes features.
# #         # Tout ce qui est envoyé en plus par le serveur (multi-full-v0) sera ignoré ici.
# #         self.keys_whitelist = sorted([
# #             'attachment', 'attachment_time_left', 'aux_ticks', 'center_path', 
# #             'center_path_distance', 'distance_down_track', 'energy', 'front', 
# #             'items_position', 'items_type', 'jumping', 'karts_position', 
# #             'max_steer_angle', 'paths_distance', 'paths_end', 'paths_start', 
# #             'paths_width', 'phase', 'powerup', 'shield_time', 'skeed_factor', 
# #             'velocity',
# #             # Tes Features
# #             'feat_speed', 'feat_dist_center', 'feat_angle_center', 'feat_future_risk',
# #             'feat_lookahead_angle', 'feat_curve_intensity', 'feat_skeed', 'feat_air',
# #             'feat_item_angle', 'feat_item_detected', 'feat_off_track'
# #         ])
        
# #         # 112 est la taille garantie par cette liste
# #         self.target_size = 112
        
# #         self.observation_space = spaces.Box(
# #             low=-np.inf, high=np.inf, shape=(self.target_size,), dtype=np.float32
# #         )

# #     def observation(self, obs):
# #         flat_list = []
# #         for key in self.keys_whitelist:
# #             if key in obs:
# #                 val = obs[key]
# #                 if isinstance(val, (int, float, np.number)): 
# #                     flat_list.append([val])
# #                 elif isinstance(val, np.ndarray): 
# #                     flat_list.append(val.flatten())
# #             else:
# #                 # Si une clé manque (ex: différence de version pystk), on ne crash pas.
# #                 # Mais simple-v0 est très stable.
# #                 pass
        
# #         flat_obs = np.concatenate(flat_list).astype(np.float32)
        
# #         # Sécurité dimensions (Padding)
# #         if len(flat_obs) < self.target_size:
# #              padding = np.zeros(self.target_size - len(flat_obs), dtype=np.float32)
# #              flat_obs = np.concatenate([flat_obs, padding])
        
# #         # Au cas où, on coupe l'excédent (mais la whitelist devrait empêcher ça)
# #         if len(flat_obs) > self.target_size:
# #              flat_obs = flat_obs[:self.target_size]
             
# #         return flat_obs

# # # =============================================================================
# # # 4. FRAME STACKING
# # # =============================================================================
# # class FrameStackingWrapper(gym.ObservationWrapper):
# #     def __init__(self, env, n_stack=4):
# #         super().__init__(env)
# #         self.n_stack = n_stack
# #         self.frames = deque(maxlen=n_stack)
        
# #         original_shape = 112
# #         self.observation_space = spaces.Box(
# #             low=-np.inf, high=np.inf, shape=(original_shape * n_stack,), dtype=np.float32
# #         )

# #     def reset(self, **kwargs):
# #         obs, info = self.env.reset(**kwargs)
# #         self.frames.clear()
# #         for _ in range(self.n_stack): 
# #             self.frames.append(obs)
# #         # Appel explicite à observation() pour formater la sortie
# #         return self.observation(obs), info

# #     def observation(self, obs):
# #         # On ajoute à la deque (le maxlen gère l'éjection des vieux frames)
# #         self.frames.append(obs)
# #         return np.concatenate(list(self.frames)).astype(np.float32)
# # # import gymnasium as gym
# # # from gymnasium import spaces
# # # import numpy as np
# # # from collections import deque

# # # # =============================================================================
# # # # 1. FEATURE ENGINEERING (ObservationWrapper)
# # # # =============================================================================
# # # class FeatureEngineeringWrapper(gym.ObservationWrapper):
# # #     def __init__(self, env):
# # #         super().__init__(env)
    
# # #     def observation(self, obs):
# # #         return self._compute_features(obs)

# # #     def _compute_features(self, obs):
# # #         # Extraction
# # #         velocity_vec = obs['velocity']
# # #         speed = np.linalg.norm(velocity_vec)
# # #         center_vec = obs['center_path'] 
# # #         dist_center = np.linalg.norm(center_vec)
# # #         angle_center = np.arctan2(center_vec[0], center_vec[2])

# # #         # Risk
# # #         raw_widths = obs.get('paths_width', np.array([10.0]))
# # #         safe_width = np.min(raw_widths) 
# # #         limit_dist = safe_width / 2.0
# # #         future_risk = 0.0
# # #         if limit_dist > 0:
# # #             current_lateral_pos = -center_vec[0]
# # #             future_pos = current_lateral_pos + (velocity_vec[0] * 0.5)
# # #             future_risk = abs(future_pos) / limit_dist

# # #         # Lookahead
# # #         paths_end = obs.get('paths_end', [])
# # #         lookahead_angle = 0.0
# # #         if len(paths_end) > 3:
# # #             target = paths_end[2]
# # #             lookahead_angle = np.arctan2(target[0], target[2])
# # #         curve_intensity = lookahead_angle - angle_center

# # #         # Physique & Items
# # #         skeed = float(obs.get('skeed_factor', 0.0))
# # #         is_airborne = float(obs.get('jumping', 0))
        
# # #         items = obs.get('items_position', [])
# # #         item_angle = 0.0
# # #         has_item = 0.0
# # #         if len(items) > 0:
# # #             dists = np.linalg.norm(items, axis=1)
# # #             min_idx = np.argmin(dists)
# # #             if items[min_idx][2] > 0 and dists[min_idx] < 20.0:
# # #                 item_angle = np.arctan2(items[min_idx][0], items[min_idx][2])
# # #                 has_item = 1.0

# # #         # Injection
# # #         obs['feat_speed'] = np.array([speed], dtype=np.float32)
# # #         obs['feat_dist_center'] = np.array([dist_center], dtype=np.float32)
# # #         obs['feat_angle_center'] = np.array([angle_center], dtype=np.float32)
# # #         obs['feat_future_risk'] = np.array([future_risk], dtype=np.float32)
# # #         obs['feat_lookahead_angle'] = np.array([lookahead_angle], dtype=np.float32)
# # #         obs['feat_curve_intensity'] = np.array([curve_intensity], dtype=np.float32)
# # #         obs['feat_skeed'] = np.array([skeed], dtype=np.float32)
# # #         obs['feat_air'] = np.array([is_airborne], dtype=np.float32)
# # #         obs['feat_item_angle'] = np.array([item_angle], dtype=np.float32)
# # #         obs['feat_item_detected'] = np.array([has_item], dtype=np.float32)
# # #         obs['feat_off_track'] = np.array([1.0 if dist_center > limit_dist else 0.0], dtype=np.float32)

# # #         return obs

# # # # =============================================================================
# # # # 2. ACTION CONVERSION (ActionWrapper)
# # # # =============================================================================
# # # class ActionConversionWrapper(gym.ActionWrapper):
# # #     def __init__(self, env):
# # #         super().__init__(env)
# # #         # 15 actions
# # #         self.actions_list = [
# # #             (0.0, 1.0, 0, 0, 0, 0, 0), (-0.6, 1.0, 0, 0, 0, 0, 0), (0.6, 1.0, 0, 0, 0, 0, 0),
# # #             (-1.0, 1.0, 0, 1, 0, 0, 0), (1.0, 1.0, 0, 1, 0, 0, 0), (-0.5, 1.0, 0, 1, 0, 0, 0),
# # #             (0.5, 1.0, 0, 1, 0, 0, 0), (0.0, 0.3, 0, 0, 0, 0, 0), (-0.6, 0.3, 0, 0, 0, 0, 0),
# # #             (0.6, 0.3, 0, 0, 0, 0, 0), (0.0, 0.0, 1, 0, 0, 0, 0), (-1.0, 0.0, 1, 0, 0, 0, 0),
# # #             (1.0, 0.0, 1, 0, 0, 0, 0), (0.0, 1.0, 0, 0, 1, 0, 0), (0.0, 1.0, 0, 0, 0, 1, 0),
# # #         ]
# # #         self.action_space = spaces.Discrete(len(self.actions_list))

# # #     def action(self, action_idx):
# # #         if hasattr(action_idx, "item"): action_idx = int(action_idx.item())
# # #         else: action_idx = int(action_idx)
# # #         vals = self.actions_list[action_idx]
# # #         return {
# # #             'steer': np.array([vals[0]], dtype=np.float32),
# # #             'acceleration': np.array([vals[1]], dtype=np.float32),
# # #             'brake': vals[2], 'drift': vals[3], 'fire': vals[4], 'nitro': vals[5], 'rescue': vals[6]
# # #         }

# # # # =============================================================================
# # # # 3. FLATTEN WRAPPER (WHITELIST ROBUSTE)
# # # # =============================================================================
# # # class FlattenWrapper(gym.ObservationWrapper):
# # #     def __init__(self, env):
# # #         super().__init__(env)
        
# # #         # LISTE STRICTE : On ne garde QUE ce que tu as utilisé en local (112 dims).
# # #         # Les clés du serveur non listées ici seront ignorées.
# # #         self.keys_whitelist = sorted([
# # #             'attachment', 'attachment_time_left', 'aux_ticks', 'center_path', 
# # #             'center_path_distance', 'distance_down_track', 'energy', 'front', 
# # #             'items_position', 'items_type', 'jumping', 'karts_position', 
# # #             'max_steer_angle', 'paths_distance', 'paths_end', 'paths_start', 
# # #             'paths_width', 'phase', 'powerup', 'shield_time', 'skeed_factor', 
# # #             'velocity',
# # #             # Tes Features
# # #             'feat_speed', 'feat_dist_center', 'feat_angle_center', 'feat_future_risk',
# # #             'feat_lookahead_angle', 'feat_curve_intensity', 'feat_skeed', 'feat_air',
# # #             'feat_item_angle', 'feat_item_detected', 'feat_off_track'
# # #         ])
        
# # #         self.target_size = 112
        
# # #         self.observation_space = spaces.Box(
# # #             low=-np.inf, high=np.inf, shape=(self.target_size,), dtype=np.float32
# # #         )

# # #     def observation(self, obs):
# # #         flat_list = []
# # #         # On itère sur la Whitelist : Si le serveur a des clés en plus, on ne les voit pas.
# # #         for key in self.keys_whitelist:
# # #             if key in obs:
# # #                 val = obs[key]
# # #                 if isinstance(val, (int, float, np.number)): 
# # #                     flat_list.append([val])
# # #                 elif isinstance(val, np.ndarray): 
# # #                     flat_list.append(val.flatten())
# # #             # Si une clé manque (peu probable), on ignore (pas de crash, juste vecteur plus court)
# # #             # Mais simple-v0 est stable sur les clés de base.
        
# # #         flat_obs = np.concatenate(flat_list).astype(np.float32)
        
# # #         # Sécurité ultime sur la taille (Padding si nécessaire)
# # #         if len(flat_obs) < self.target_size:
# # #              padding = np.zeros(self.target_size - len(flat_obs), dtype=np.float32)
# # #              flat_obs = np.concatenate([flat_obs, padding])
             
# # #         return flat_obs

# # # # =============================================================================
# # # # 4. FRAME STACKING
# # # # =============================================================================
# # # class FrameStackingWrapper(gym.ObservationWrapper):
# # #     def __init__(self, env, n_stack=4):
# # #         super().__init__(env)
# # #         self.n_stack = n_stack
# # #         self.frames = deque(maxlen=n_stack)
        
# # #         original_shape = 112
# # #         self.observation_space = spaces.Box(
# # #             low=-np.inf, high=np.inf, shape=(original_shape * n_stack,), dtype=np.float32
# # #         )

# # #     def reset(self, **kwargs):
# # #         obs, info = self.env.reset(**kwargs)
# # #         self.frames.clear()
# # #         for _ in range(self.n_stack): 
# # #             self.frames.append(obs)
# # #         return self._get_stacked_obs(), info

# # #     def observation(self, obs):
# # #         self.frames.append(obs)
# # #         return self._get_stacked_obs()

# # #     def _get_stacked_obs(self):
# # #         return np.concatenate(list(self.frames)).astype(np.float32)
    

# # # class DiscreteActionWrapper(gym.Wrapper):
# # #     def __init__(self, env):
# # #         super().__init__(env)
        
# # #         dummy_obs, _ = self.env.reset()
# # #         flat_size = self._flatten_obs(dummy_obs).shape[0]
# # #         self.observation_space = spaces.Box(
# # #             low=-np.inf, high=np.inf, shape=(flat_size,), dtype=np.float32
# # #         )

# # #         # Liste d'actions optimisée (15 actions)
# # #         # Format: (Steer, Accel, Brake, Drift, Fire, Nitro, Rescue)
# # #         self.actions_list = [
# # #             # --- VITESSE MAX (0-2) ---
# # #             (0.0, 1.0, 0, 0, 0, 0, 0),   # 0: Tout droit (Fond)
# # #             (-0.6, 1.0, 0, 0, 0, 0, 0),  # 1: Gauche (Fond)
# # #             (0.6, 1.0, 0, 0, 0, 0, 0),   # 2: Droite (Fond)
            
# # #             # --- DRIFT TECHNIQUE (3-6) ---
# # #             (-1.0, 1.0, 0, 1, 0, 0, 0),  # 3: Drift Gauche FORT
# # #             (1.0, 1.0, 0, 1, 0, 0, 0),   # 4: Drift Droite FORT
# # #             (-0.5, 1.0, 0, 1, 0, 0, 0),  # 5: Drift Gauche MOYEN
# # #             (0.5, 1.0, 0, 1, 0, 0, 0),   # 6: Drift Droite MOYEN
            
# # #             # --- VITESSE LENTE / PRÉCISION (7-9) ---
# # #             (0.0, 0.3, 0, 0, 0, 0, 0),   # 7: Tout droit (Lent)
# # #             (-0.6, 0.3, 0, 0, 0, 0, 0),  # 8: Gauche (Lent)
# # #             (0.6, 0.3, 0, 0, 0, 0, 0),   # 9: Droite (Lent)
            
# # #             # --- FREIN & RECUL (10-12) ---
# # #             (0.0, 0.0, 1, 0, 0, 0, 0),   # 10: Frein / Recul Droit
# # #             (-1.0, 0.0, 1, 0, 0, 0, 0),  # 11: Recul Gauche
# # #             (1.0, 0.0, 1, 0, 0, 0, 0),   # 12: Recul Droite
            
# # #             # --- BONUS (13-14) ---
# # #             (0.0, 1.0, 0, 0, 1, 0, 0),   # 13: Fire
# # #             (0.0, 1.0, 0, 0, 0, 1, 0),   # 14: Nitro
# # #         ]
# # #         self.action_space = spaces.Discrete(len(self.actions_list))

# # #     def reset(self, **kwargs):
# # #         obs, info = self.env.reset(**kwargs)
# # #         return self._flatten_obs(obs), info

# # #     def step(self, action_idx):
# # #         vals = self.actions_list[action_idx]
        
# # #         game_action = {
# # #             'steer': np.array([vals[0]], dtype=np.float32),
# # #             'acceleration': np.array([vals[1]], dtype=np.float32),
# # #             'brake': vals[2],
# # #             'drift': vals[3],
# # #             'fire': vals[4],
# # #             'nitro': vals[5],
# # #             'rescue': vals[6] # Sera toujours 0 ici
# # #         }
        
# # #         obs, reward, terminated, truncated, info = self.env.step(game_action)
# # #         return self._flatten_obs(obs), reward, terminated, truncated, info