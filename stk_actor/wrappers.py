import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

class UltraWrapper(gym.ObservationWrapper):
    """
    Ce wrapper fait TOUT :
    1. Calcule les features.
    2. Filtre et Aplatit (Whitelist -> 112 dims).
    3. Empile 4 frames (Stacking -> 448 dims).
    """
    def __init__(self, env):
        super().__init__(env)
        
        self.n_stack = 4
        self.frames = deque(maxlen=self.n_stack)
        
        # --- CONFIGURATION WHITELIST (112 dims) ---
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
        
        # --- OUTPUT SPACE (448 dims) ---
        self.target_size = self.single_frame_size * self.n_stack
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.target_size,), dtype=np.float32
        )

    def observation(self, obs):
        """
        Cette méthode est appelée par Gym. 
        Pour respecter le contrat ObservationWrapper, elle doit retourner l'observation FINALE (448).
        Donc elle doit gérer le stacking ici-même.
        """
        # 1. Calcul des Features (Feature Engineering)
        processed_frame = self._compute_features_and_flatten(obs)
        
        # 2. Gestion du Stacking
        # Astuce : Si c'est un reset (deque vide), on remplit tout.
        # Sinon (step), on ajoute juste une frame.
        if len(self.frames) == 0:
            for _ in range(self.n_stack):
                self.frames.append(processed_frame)
        else:
            self.frames.append(processed_frame)
            
        # 3. Retour du Stack (448)
        return np.concatenate(list(self.frames)).astype(np.float32)

    def reset(self, **kwargs):
        # On vide la mémoire avant le reset
        self.frames.clear()
        obs, info = self.env.reset(**kwargs)
        # Gym appellera self.observation(obs) automatiquement ou on le fait ici
        return self.observation(obs), info

    def _compute_features_and_flatten(self, obs):
        # --- A. Feature Engineering (Copie de ton code) ---
        velocity_vec = obs.get('velocity', np.zeros(3))
        speed = np.linalg.norm(velocity_vec)
        center_vec = obs.get('center_path', np.zeros(3))
        dist_center = np.linalg.norm(center_vec)
        angle_center = 0.0
        if center_vec.shape[0] >= 3 and center_vec[2] != 0:
            angle_center = np.arctan2(center_vec[0], center_vec[2])

        raw_widths = obs.get('paths_width', np.array([10.0]))
        safe_width = np.min(raw_widths) if raw_widths.size > 0 else 10.0
        limit_dist = safe_width / 2.0
        
        future_risk = 0.0
        if limit_dist > 0:
            current_lateral_pos = -center_vec[0]
            future_pos = current_lateral_pos + (velocity_vec[0] * 0.5)
            future_risk = abs(future_pos) / limit_dist

        paths_end = obs.get('paths_end', [])
        lookahead_angle = 0.0
        if len(paths_end) > 3:
            target = paths_end[2]
            lookahead_angle = np.arctan2(target[0], target[2])
        curve_intensity = lookahead_angle - angle_center

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

        # On modifie le dict (ou une copie)
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

        # --- B. Flatten avec Whitelist ---
        flat_list = []
        for key in self.keys_whitelist:
            if key in obs:
                val = obs[key]
                if isinstance(val, (int, float, np.number)): 
                    flat_list.append([val])
                elif isinstance(val, np.ndarray): 
                    flat_list.append(val.flatten())
        
        flat_obs = np.concatenate(flat_list).astype(np.float32)
        
        # --- C. Padding Force 112 ---
        if len(flat_obs) < self.single_frame_size:
             padding = np.zeros(self.single_frame_size - len(flat_obs), dtype=np.float32)
             flat_obs = np.concatenate([flat_obs, padding])
        elif len(flat_obs) > self.single_frame_size:
             flat_obs = flat_obs[:self.single_frame_size]
             
        return flat_obs

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
        

import copy
import logging
import sys
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import pystk2
from gymnasium import spaces

from .envs import STKAction, STKRaceEnv
from .definitions import ActionObservationWrapper
from pystk2_gymnasium.utils import Discretizer, max_enum_value
class ConstantSizedObservationsNew(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        state_items=5,
        state_karts=5,
        state_paths=5,
        add_mask=False,
        **kwargs,
    ):
        """A simpler race environment with fixed width data

        :param state_items: The number of items, defaults to 5
        :param state_karts: The number of karts, defaults to 5
        """
        super().__init__(env, **kwargs)
        if isinstance(env.unwrapped, STKRaceEnv) and env.unwrapped.max_paths is None:
            logging.info("Setting unwrapped environment max_paths to %d", state_paths)
            env.unwrapped.max_paths = min(
                state_paths, env.unwrapped.max_paths or sys.maxsize
            )

        self.state_items = state_items
        self.state_karts = state_karts
        self.state_paths = state_paths

        # Override some keys in the observation space
        self._observation_space = space = copy.deepcopy(self.env.observation_space)

        space["paths_distance"] = spaces.Box(
            0, float("inf"), shape=(self.state_paths, 2), dtype=np.float32
        )
        space["paths_width"] = spaces.Box(
            0, float("inf"), shape=(self.state_paths, 1), dtype=np.float32
        )
        space["paths_start"] = spaces.Box(
            -float("inf"), float("inf"), shape=(self.state_paths, 3), dtype=np.float32
        )
        space["paths_end"] = spaces.Box(
            -float("inf"), float("inf"), shape=(self.state_paths, 3), dtype=np.float32
        )
        space["items_position"] = spaces.Box(
            -float("inf"), float("inf"), shape=(self.state_items, 3), dtype=np.float32
        )
        n_item_types = max_enum_value(pystk2.Item)
        space["items_type"] = spaces.MultiDiscrete(
            [n_item_types for _ in range(self.state_items)]
        )
        space["karts_position"] = spaces.Box(
            -float("inf"), float("inf"), shape=(self.state_karts, 3)
        )

        self.add_mask = add_mask
        if add_mask:
            space["paths_mask"] = spaces.Box(
                0, 1, shape=(self.state_paths,), dtype=np.int8
            )
            space["items_mask"] = spaces.Box(
                0, 1, shape=(self.state_items,), dtype=np.int8
            )
            space["karts_mask"] = spaces.Box(
                0, 1, shape=(self.state_karts,), dtype=np.int8
            )

    def make_tensor(self, state, name: str, default_value=0):
        value = state[name]
        space = self.observation_space[name]

        value = np.stack(value)
        assert (
            space.shape[1:] == value.shape[1:]
        ), f"Shape mismatch for {name}: {space.shape} vs {value.shape}"

        delta = space.shape[0] - value.shape[0]
        if delta > 0:
            shape = [delta] + list(space.shape[1:])
            value = np.concatenate(
                [value, np.full(shape, default_value, dtype=space.dtype)], axis=0
            )
        elif delta < 0:
            value = value[:delta]

        assert (
            space.shape == value.shape
        ), f"Shape mismatch for {name}: {space.shape} vs {value.shape}"
        state[name] = value

    def observation(self, state):
        # Shallow copy
        state = {**state}

        # Add masks
        def mask(length: int, size: int):
            v = np.zeros((size,), dtype=np.int8)
            v[:length] = 1
            return v

        if self.add_mask:
            state["paths_mask"] = mask(len(state["paths_width"]), self.state_paths)
            state["items_mask"] = mask(len(state["items_type"]), self.state_items)
            state["karts_mask"] = mask(len(state["karts_position"]), self.state_karts)

        # Ensures that the size of observations is constant
        self.make_tensor(state, "paths_distance")
        self.make_tensor(state, "paths_width")
        self.make_tensor(state, "paths_start")
        self.make_tensor(state, "paths_end")
        self.make_tensor(state, "items_position")
        self.make_tensor(state, "items_type")
        self.make_tensor(state, "karts_position")

        return state


# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# from collections import deque

# # =============================================================================
# # 1. FEATURE ENGINEERING (ObservationWrapper)
# # =============================================================================
# class FeatureEngineeringWrapper(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
    
#     def observation(self, obs):
#         return self._compute_features(obs)

#     def _compute_features(self, obs):
#         # Utilisation de .get() pour être robuste si une clé manque
#         velocity_vec = obs.get('velocity', np.zeros(3))
#         speed = np.linalg.norm(velocity_vec)
#         center_vec = obs.get('center_path', np.zeros(3))
#         dist_center = np.linalg.norm(center_vec)
        
#         angle_center = 0.0
#         if center_vec.shape[0] >= 3 and center_vec[2] != 0:
#             angle_center = np.arctan2(center_vec[0], center_vec[2])

#         raw_widths = obs.get('paths_width', np.array([10.0]))
#         # Gestion sécurité si tableau vide
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
#             target = paths_end[2]
#             lookahead_angle = np.arctan2(target[0], target[2])
#         curve_intensity = lookahead_angle - angle_center

#         skeed = float(obs.get('skeed_factor', 0.0))
#         is_airborne = float(obs.get('jumping', 0))
        
#         items = obs.get('items_position', [])
#         item_angle = 0.0
#         has_item = 0.0
#         if len(items) > 0:
#             dists = np.linalg.norm(items, axis=1)
#             min_idx = np.argmin(dists)
#             if items[min_idx][2] > 0 and dists[min_idx] < 20.0:
#                 item_angle = np.arctan2(items[min_idx][0], items[min_idx][2])
#                 has_item = 1.0

#         # Injection des features
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

#         return obs

# # =============================================================================
# # 2. ACTION CONVERSION (ActionWrapper)
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
# # 3. FLATTEN WRAPPER (LA CLÉ DU SUCCÈS : WHITELIST)
# # =============================================================================
# class FlattenWrapper(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
        
#         # LISTE STRICTE des clés de simple-v0 + tes features.
#         # Tout ce qui est envoyé en plus par le serveur (multi-full-v0) sera ignoré ici.
#         self.keys_whitelist = sorted([
#             'attachment', 'attachment_time_left', 'aux_ticks', 'center_path', 
#             'center_path_distance', 'distance_down_track', 'energy', 'front', 
#             'items_position', 'items_type', 'jumping', 'karts_position', 
#             'max_steer_angle', 'paths_distance', 'paths_end', 'paths_start', 
#             'paths_width', 'phase', 'powerup', 'shield_time', 'skeed_factor', 
#             'velocity',
#             # Tes Features
#             'feat_speed', 'feat_dist_center', 'feat_angle_center', 'feat_future_risk',
#             'feat_lookahead_angle', 'feat_curve_intensity', 'feat_skeed', 'feat_air',
#             'feat_item_angle', 'feat_item_detected', 'feat_off_track'
#         ])
        
#         # 112 est la taille garantie par cette liste
#         self.target_size = 112
        
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(self.target_size,), dtype=np.float32
#         )

#     def observation(self, obs):
#         flat_list = []
#         for key in self.keys_whitelist:
#             if key in obs:
#                 val = obs[key]
#                 if isinstance(val, (int, float, np.number)): 
#                     flat_list.append([val])
#                 elif isinstance(val, np.ndarray): 
#                     flat_list.append(val.flatten())
#             else:
#                 # Si une clé manque (ex: différence de version pystk), on ne crash pas.
#                 # Mais simple-v0 est très stable.
#                 pass
        
#         flat_obs = np.concatenate(flat_list).astype(np.float32)
        
#         # Sécurité dimensions (Padding)
#         if len(flat_obs) < self.target_size:
#              padding = np.zeros(self.target_size - len(flat_obs), dtype=np.float32)
#              flat_obs = np.concatenate([flat_obs, padding])
        
#         # Au cas où, on coupe l'excédent (mais la whitelist devrait empêcher ça)
#         if len(flat_obs) > self.target_size:
#              flat_obs = flat_obs[:self.target_size]
             
#         return flat_obs

# # =============================================================================
# # 4. FRAME STACKING
# # =============================================================================
# class FrameStackingWrapper(gym.ObservationWrapper):
#     def __init__(self, env, n_stack=4):
#         super().__init__(env)
#         self.n_stack = n_stack
#         self.frames = deque(maxlen=n_stack)
        
#         original_shape = 112
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(original_shape * n_stack,), dtype=np.float32
#         )

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         self.frames.clear()
#         for _ in range(self.n_stack): 
#             self.frames.append(obs)
#         # Appel explicite à observation() pour formater la sortie
#         return self.observation(obs), info

#     def observation(self, obs):
#         # On ajoute à la deque (le maxlen gère l'éjection des vieux frames)
#         self.frames.append(obs)
#         return np.concatenate(list(self.frames)).astype(np.float32)
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
# #         # Extraction
# #         velocity_vec = obs['velocity']
# #         speed = np.linalg.norm(velocity_vec)
# #         center_vec = obs['center_path'] 
# #         dist_center = np.linalg.norm(center_vec)
# #         angle_center = np.arctan2(center_vec[0], center_vec[2])

# #         # Risk
# #         raw_widths = obs.get('paths_width', np.array([10.0]))
# #         safe_width = np.min(raw_widths) 
# #         limit_dist = safe_width / 2.0
# #         future_risk = 0.0
# #         if limit_dist > 0:
# #             current_lateral_pos = -center_vec[0]
# #             future_pos = current_lateral_pos + (velocity_vec[0] * 0.5)
# #             future_risk = abs(future_pos) / limit_dist

# #         # Lookahead
# #         paths_end = obs.get('paths_end', [])
# #         lookahead_angle = 0.0
# #         if len(paths_end) > 3:
# #             target = paths_end[2]
# #             lookahead_angle = np.arctan2(target[0], target[2])
# #         curve_intensity = lookahead_angle - angle_center

# #         # Physique & Items
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

# #         # Injection
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
# #         # 15 actions
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
# # # 3. FLATTEN WRAPPER (WHITELIST ROBUSTE)
# # # =============================================================================
# # class FlattenWrapper(gym.ObservationWrapper):
# #     def __init__(self, env):
# #         super().__init__(env)
        
# #         # LISTE STRICTE : On ne garde QUE ce que tu as utilisé en local (112 dims).
# #         # Les clés du serveur non listées ici seront ignorées.
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
        
# #         self.target_size = 112
        
# #         self.observation_space = spaces.Box(
# #             low=-np.inf, high=np.inf, shape=(self.target_size,), dtype=np.float32
# #         )

# #     def observation(self, obs):
# #         flat_list = []
# #         # On itère sur la Whitelist : Si le serveur a des clés en plus, on ne les voit pas.
# #         for key in self.keys_whitelist:
# #             if key in obs:
# #                 val = obs[key]
# #                 if isinstance(val, (int, float, np.number)): 
# #                     flat_list.append([val])
# #                 elif isinstance(val, np.ndarray): 
# #                     flat_list.append(val.flatten())
# #             # Si une clé manque (peu probable), on ignore (pas de crash, juste vecteur plus court)
# #             # Mais simple-v0 est stable sur les clés de base.
        
# #         flat_obs = np.concatenate(flat_list).astype(np.float32)
        
# #         # Sécurité ultime sur la taille (Padding si nécessaire)
# #         if len(flat_obs) < self.target_size:
# #              padding = np.zeros(self.target_size - len(flat_obs), dtype=np.float32)
# #              flat_obs = np.concatenate([flat_obs, padding])
             
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
# #         return self._get_stacked_obs(), info

# #     def observation(self, obs):
# #         self.frames.append(obs)
# #         return self._get_stacked_obs()

# #     def _get_stacked_obs(self):
# #         return np.concatenate(list(self.frames)).astype(np.float32)
    

# # class DiscreteActionWrapper(gym.Wrapper):
# #     def __init__(self, env):
# #         super().__init__(env)
        
# #         dummy_obs, _ = self.env.reset()
# #         flat_size = self._flatten_obs(dummy_obs).shape[0]
# #         self.observation_space = spaces.Box(
# #             low=-np.inf, high=np.inf, shape=(flat_size,), dtype=np.float32
# #         )

# #         # Liste d'actions optimisée (15 actions)
# #         # Format: (Steer, Accel, Brake, Drift, Fire, Nitro, Rescue)
# #         self.actions_list = [
# #             # --- VITESSE MAX (0-2) ---
# #             (0.0, 1.0, 0, 0, 0, 0, 0),   # 0: Tout droit (Fond)
# #             (-0.6, 1.0, 0, 0, 0, 0, 0),  # 1: Gauche (Fond)
# #             (0.6, 1.0, 0, 0, 0, 0, 0),   # 2: Droite (Fond)
            
# #             # --- DRIFT TECHNIQUE (3-6) ---
# #             (-1.0, 1.0, 0, 1, 0, 0, 0),  # 3: Drift Gauche FORT
# #             (1.0, 1.0, 0, 1, 0, 0, 0),   # 4: Drift Droite FORT
# #             (-0.5, 1.0, 0, 1, 0, 0, 0),  # 5: Drift Gauche MOYEN
# #             (0.5, 1.0, 0, 1, 0, 0, 0),   # 6: Drift Droite MOYEN
            
# #             # --- VITESSE LENTE / PRÉCISION (7-9) ---
# #             (0.0, 0.3, 0, 0, 0, 0, 0),   # 7: Tout droit (Lent)
# #             (-0.6, 0.3, 0, 0, 0, 0, 0),  # 8: Gauche (Lent)
# #             (0.6, 0.3, 0, 0, 0, 0, 0),   # 9: Droite (Lent)
            
# #             # --- FREIN & RECUL (10-12) ---
# #             (0.0, 0.0, 1, 0, 0, 0, 0),   # 10: Frein / Recul Droit
# #             (-1.0, 0.0, 1, 0, 0, 0, 0),  # 11: Recul Gauche
# #             (1.0, 0.0, 1, 0, 0, 0, 0),   # 12: Recul Droite
            
# #             # --- BONUS (13-14) ---
# #             (0.0, 1.0, 0, 0, 1, 0, 0),   # 13: Fire
# #             (0.0, 1.0, 0, 0, 0, 1, 0),   # 14: Nitro
# #         ]
# #         self.action_space = spaces.Discrete(len(self.actions_list))

# #     def reset(self, **kwargs):
# #         obs, info = self.env.reset(**kwargs)
# #         return self._flatten_obs(obs), info

# #     def step(self, action_idx):
# #         vals = self.actions_list[action_idx]
        
# #         game_action = {
# #             'steer': np.array([vals[0]], dtype=np.float32),
# #             'acceleration': np.array([vals[1]], dtype=np.float32),
# #             'brake': vals[2],
# #             'drift': vals[3],
# #             'fire': vals[4],
# #             'nitro': vals[5],
# #             'rescue': vals[6] # Sera toujours 0 ici
# #         }
        
# #         obs, reward, terminated, truncated, info = self.env.step(game_action)
# #         return self._flatten_obs(obs), reward, terminated, truncated, info