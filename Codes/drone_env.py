import math
from typing import Optional, Tuple, Dict, Any

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from drone_model import QuadrotorDynamics, QuadParams


class QuadHoverEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = "drone.xml",
        target_height: float = 1.0,
        frame_skip: float = 0.6,
        max_steps: int = 500,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.params = QuadParams()
        self.quad = QuadrotorDynamics(self.model, self.data, self.params)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.target_height = target_height
        self.p_target = np.array([0.0, 0.0, target_height])
        
        obs_high = np.ones(12, dtype=float) * np.inf
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._viewer = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)
        
        # HOVER: Start near target in Air
        self.data.qpos[0:3] = self.p_target + self.np_random.uniform(-0.1, 0.1, size=3)
        self.data.qpos[3:7] = np.array([1., 0., 0., 0.])
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self.step_count += 1
        a = np.clip(action, -1.0, 1.0)
        for _ in range(self.frame_skip):
            self.quad.step(a)
            
        obs = self._get_obs()
        p, v, R, omega = self.quad.get_state()
        dist = np.linalg.norm(p - self.p_target)
        
        r_pos = -2.0 * (dist**2)
        r_vel = -0.2 * np.dot(v, v)
        r_act = -0.05 * np.dot(a, a)
        r_tilt = -0.5 * (1.0 - R[2,2])
        
        reward = r_pos + r_vel + r_act + r_tilt + 0.5 
        
        terminated = (dist > 3.0) or (p[2] < 0.1) or (R[2, 2] < 0.5)
        truncated = self.step_count >= self.max_steps
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        p, v, R, omega = self.quad.get_state()
        e_p = p - self.p_target
        z_b = R[:, 2]
        omega_body = R.T @ omega
        return np.concatenate([e_p, v, z_b, omega_body]).astype(np.float32)

    def render(self):
        try:
            from mujoco import viewer
            if self._viewer is None:
                self._viewer = viewer.launch_passive(self.model, self.data)
            else:
                self._viewer.sync()
        except ImportError:
            pass


class QuadWaypointEnv(QuadHoverEnv):
    def __init__(
        self,
        xml_path: str = "drone.xml",
        frame_skip: int = 2,
        max_steps: int = 500,
        seed: Optional[int] = None,
        min_radius: float = 1.0,
        max_radius: float = 3.0,
        target_height: float = 1.0,
        goal_radius: float = 0.50,
        goal_bonus: float = 10.0,
    ):
        super().__init__(
            xml_path=xml_path,
            target_height=target_height,
            frame_skip=frame_skip,
            max_steps=max_steps,
            seed=seed,
        )
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.goal_radius = goal_radius
        self.goal_bonus = goal_bonus
        self.max_pos_error = 10.0 
        self.max_tilt_deg = 80.0
        self.prev_dist = 0.0
        # Initialize prev_wp with current position just for safety
        self.prev_wp = np.array([0.0, 0.0, target_height]) 

    def _sample_waypoint(self) -> np.ndarray:
        r = self.np_random.uniform(self.min_radius, self.max_radius)
        theta = self.np_random.uniform(-np.pi, np.pi)
        z = self.target_height + self.np_random.uniform(-0.5, 0.5)
        z = max(0.5, z)
        return np.array([r * np.cos(theta), r * np.sin(theta), z])

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.step_count = 0
        self.p_target = self._sample_waypoint()
        
        mujoco.mj_resetData(self.model, self.data)
        
        self.data.qpos[0:3] = np.array([0.0, 0.0, 1.0])
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[:] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        
        p = self.data.qpos[0:3]
        self.prev_dist = np.linalg.norm(p - self.p_target)
        self.prev_wp = p.copy() # Set start point as previous waypoint
        
        return self._get_obs(), {"p_target": self.p_target}

    def _get_obs(self) -> np.ndarray:
        p, v, R, omega = self.quad.get_state()
        e_p = p - self.p_target
        z_b = R[:, 2]
        omega_body = R.T @ omega
        return np.concatenate([e_p, v, z_b, omega_body]).astype(np.float32)

    def step(self, action: np.ndarray):
        self.step_count += 1
        a = np.clip(action, -1.0, 1.0)
        
        for _ in range(self.frame_skip):
            self.quad.step(a)
            
        obs = self._get_obs()
        p, v, R, omega_world = self.quad.get_state()
        dist = float(np.linalg.norm(p - self.p_target))
        
        # --- REWARDS ---
        
        # 1. Progress Reward
        progress = self.prev_dist - dist
        r_prog = 5.0 * progress
        
        # 2. Centering Reward (Cross-Track Error)
        # Vector from prev_wp to target
        line_vec = self.p_target - self.prev_wp
        line_len_sq = np.dot(line_vec, line_vec)
        
        if line_len_sq > 1e-6:
            # Vector from prev_wp to drone
            drone_vec = p - self.prev_wp
            # Project drone_vec onto line_vec
            t = np.dot(drone_vec, line_vec) / line_len_sq
            t = np.clip(t, 0.0, 1.0) # Clamp to segment
            closest_point = self.prev_wp + t * line_vec
            
            # Distance from line
            cross_track_error = float(np.linalg.norm(p - closest_point))
            r_center = -0.5 * cross_track_error # Penalize deviation
        else:
            r_center = 0.0

        r_alive = 0.05
        r_act = -0.05 * float(np.dot(a, a))
        
        r_ground = 0.0
        if p[2] < 0.2: r_ground = -0.5 

        z_b = R[:, 2]
        r_tilt = 0.0
        if z_b[2] < 0.5: r_tilt = -0.1
            
        reward = r_prog + r_alive + r_act + r_tilt + r_ground + r_center + 0.5 #alive 
        
        reached = dist < self.goal_radius
        if reached:
            reward += self.goal_bonus
            
        terminated = False
        if dist > self.max_pos_error: terminated = True
        if p[2] < 0.05: terminated = True 
        if z_b[2] < 0.2: terminated = True 
        
        if reached:
            terminated = True 
            
        truncated = self.step_count >= self.max_steps
        self.prev_dist = dist
        
        return obs, reward, terminated, truncated, {"dist": dist, "reached": reached}


class QuadRacingEnv(QuadWaypointEnv):
    def __init__(
        self,
        xml_path: str = "drone.xml",
        frame_skip: int = 2, 
        max_steps: int = 4000, 
        seed: Optional[int] = None,
        gate_radius: float = 0.70, 
    ):
        super().__init__(
            xml_path=xml_path,
            frame_skip=frame_skip,
            max_steps=max_steps,
            seed=seed,
            goal_radius=gate_radius, # Map gate_radius to goal_radius
            goal_bonus=10.0 
        )
        
        self.track_waypoints = [
            np.array([ 2.0,  0.0, 1.5]),
            np.array([ 2.0,  2.0, 1.5]),
            np.array([ 0.0,  2.0, 1.5]),
            np.array([-2.0,  2.0, 1.5]),
            np.array([-2.0,  0.0, 1.5]),
            np.array([-2.0, -2.0, 1.5]),
            np.array([ 0.0, -2.0, 1.5]),
            np.array([ 2.0, -2.0, 1.5]),
        ]
        self.current_gate_idx = 0
        self.laps_completed = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.step_count = 0
        
        mujoco.mj_resetData(self.model, self.data)
        
        # AIR START
        self.data.qpos[0:3] = np.array([0.0, 0.0, 1.0])
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(self.model, self.data)
        
        self.current_gate_idx = 0
        self.laps_completed = 0
        self.p_target = self.track_waypoints[self.current_gate_idx]
        
        p = self.data.qpos[0:3]
        self.prev_dist = np.linalg.norm(p - self.p_target)
        self.prev_wp = p.copy() # Set initial prev_wp
        
        return self._get_obs(), {"p_target": self.p_target}

    def step(self, action: np.ndarray):
        # We re-implement step to handle the specific racing logic 
        # (switching gates, updating prev_wp correctly)
        
        self.step_count += 1
        a = np.clip(action, -1.0, 1.0)
        
        for _ in range(self.frame_skip):
            self.quad.step(a)
            
        obs = self._get_obs()
        p, v, R, omega_world = self.quad.get_state()
        dist = float(np.linalg.norm(p - self.p_target))
        
        # 1. Progress Reward
        progress = self.prev_dist - dist
        r_prog = 5.0 * progress
        
        # 2. Centering Reward
        line_vec = self.p_target - self.prev_wp
        line_len_sq = np.dot(line_vec, line_vec)
        r_center = 0.0
        if line_len_sq > 1e-6:
            drone_vec = p - self.prev_wp
            t = np.dot(drone_vec, line_vec) / line_len_sq
            t = np.clip(t, 0.0, 1.0)
            closest_point = self.prev_wp + t * line_vec
            cross_track_error = float(np.linalg.norm(p - closest_point))
            r_center = -0.5 * cross_track_error

        r_alive = 0.05
        r_act = -0.10 * float(np.dot(a, a))
        
        omega_body = R.T @ omega_world
        r_spin = -0.10 * float(np.dot(omega_body, omega_body))
        
        r_ground = 0.0
        if p[2] < 0.2: r_ground = -0.5
        
        z_b = R[:, 2]
        r_tilt = 0.0
        if z_b[2] < 0.5: r_tilt = -0.1

        reward = r_prog + r_alive + r_act + r_tilt + r_ground + r_center + r_spin + 0.5 #alive bonus

        reached = dist < self.goal_radius
        terminated = False
        
        if reached:
            # We reached the gate!
            # 1. Update previous waypoint to the CURRENT gate center (before switching)
            self.prev_wp = self.track_waypoints[self.current_gate_idx].copy()
            
            self.current_gate_idx += 1
            
            if self.current_gate_idx >= len(self.track_waypoints):
                self.current_gate_idx = 0
                self.laps_completed += 1
                reward += 50.0 
                #print(f"Lap {self.laps_completed} Completed!")
                
                if self.laps_completed >= 3:
                    terminated = True
                
            self.p_target = self.track_waypoints[self.current_gate_idx]
            reward += 10.0
            
            # 2. Reset prev_dist for the new leg
            self.prev_dist = float(np.linalg.norm(p - self.p_target))
        else:
            self.prev_dist = dist
            
        if p[2] < 0.05: terminated = True 
        if z_b[2] < 0.2: terminated = True
        
        truncated = self.step_count >= self.max_steps
        
        info = {
            "gate_idx": self.current_gate_idx,
            "laps": self.laps_completed,
            "dist": dist
        }
        
        return obs, reward, terminated, truncated, info