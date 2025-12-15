from dataclasses import dataclass
from typing import Tuple
import numpy as np
import mujoco

@dataclass
class QuadParams:
    # UPDATED MASS AND THRUST
    mass: float = 0.6          # Sync with XML density
    arm_length: float = 0.14   
    k_tau: float = 0.01        
    max_thrust: float = 4.0    # TWR ~2.7
    min_thrust: float = 0.0    
    g: float = 9.81            

class QuadrotorDynamics:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 params: QuadParams | None = None):
        self.model = model
        self.data = data
        self.params = params or QuadParams()
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quad")
        self._quat = np.zeros(4)
        self._R = np.zeros((3, 3))

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = self.data.qpos[0:3].copy()
        quat = self.data.qpos[3:7]
        mujoco.mju_quat2Mat(self._R.ravel(), quat)
        R = self._R.copy()
        v = self.data.qvel[0:3].copy()
        omega = self.data.qvel[3:6].copy()
        return p, v, R, omega

    def action_to_thrusts(self, action: np.ndarray) -> np.ndarray:
        # Full range mapping [-1, 1] -> [0, max]
        a = np.asarray(action, dtype=float)
        scale = (self.params.max_thrust - self.params.min_thrust) / 2.0
        offset = (self.params.max_thrust + self.params.min_thrust) / 2.0
        f = scale * a + offset
        return np.clip(f, self.params.min_thrust, self.params.max_thrust)

    def step(self, action: np.ndarray) -> None:
        thrusts = self.action_to_thrusts(action)
        f1, f2, f3, f4 = thrusts
        l = self.params.arm_length
        k_tau = self.params.k_tau

        u1 = f1 + f2 + f3 + f4
        tau_x = l * (f2 - f4)
        tau_y = l * (f3 - f1)
        tau_z = k_tau * (-f1 + f2 - f3 + f4)

        # Force in body frame
        F_body = np.array([0, 0, u1])
        tau_body = np.array([tau_x, tau_y, tau_z])

        # Rotate to world frame
        _, _, R, _ = self.get_state()
        F_world = R @ F_body
        tau_world = R @ tau_body

        self.data.xfrc_applied[:] = 0.0
        self.data.xfrc_applied[self.body_id, 0:3] = F_world
        self.data.xfrc_applied[self.body_id, 3:6] = tau_world

        mujoco.mj_step(self.model, self.data)