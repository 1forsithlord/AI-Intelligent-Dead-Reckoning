"""Extended Kalman Filter for GNSS-denied dead reckoning.

Clean reimplementation of dead-sih-main.ipynb cell 35 with:
- correct nonlinear mean propagation (previous version used F @ x -> drift)
- separate measurement updates for AI (v, h) and GNSS (x, y)
- heading innovation + state wrapping to [-pi, pi]
- adaptive Q / R hooks for the future AI-adaptive fusion

State: [x, y, v, heading]  (position m, speed m/s, heading rad)
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class EKFState:
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    heading: float = 0.0


class ExtendedKalmanFilter:
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.x = np.zeros((4, 1))                 # [x, y, v, heading]
        self.P = np.eye(4) * 1.0
        # Reasonable physical priors for a car-like target
        self.Q      = np.diag([1e-3, 1e-3, 1e-2, 1e-3])   # pos, pos, vel, heading
        self.R_ai   = np.diag([0.30, 0.008])              # AI:  2 km/h, 5 deg
        self.R_gnss = np.diag([25.0, 25.0])               # GNSS: 5 m per axis
        self.H_ai   = np.array([[0, 0, 1, 0],
                                [0, 0, 0, 1]])            # measure v, heading
        self.H_gnss = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0]])            # measure x, y
        self.F = np.eye(4)

    # ---------- state initialization ----------
    def set_state(self, x: float, y: float, v: float, heading: float) -> None:
        self.x = np.array([[float(x)], [float(y)], [float(v)], [float(heading)]])

    # ---------- adaptive hooks ----------
    def set_process_noise(self, Q: np.ndarray) -> None:
        assert Q.shape == (4, 4)
        self.Q = Q

    def set_ai_noise(self, R: np.ndarray) -> None:
        assert R.shape == (2, 2)
        self.R_ai = R

    def set_gnss_noise(self, R: np.ndarray) -> None:
        assert R.shape == (2, 2)
        self.R_gnss = R

    # ---------- predict ----------
    def predict(self, gyro_yaw_rate: float = 0.0) -> np.ndarray:
        x, y, v, h = (self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0])

        # Nonlinear mean propagation
        x_new = x + self.dt * v * np.cos(h)
        y_new = y + self.dt * v * np.sin(h)
        h_new = h + self.dt * gyro_yaw_rate
        self.x = np.array([[x_new], [y_new], [v], [h_new]])

        # Jacobian at old state (only for covariance)
        self.F = np.array([
            [1.0, 0.0, self.dt * np.cos(h), -self.dt * v * np.sin(h)],
            [0.0, 1.0, self.dt * np.sin(h),  self.dt * v * np.cos(h)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    # ---------- generic update ----------
    def _apply_update(self, z, H, R, wrap_idx=None):
        y = z - H @ self.x
        if wrap_idx is not None:
            for i in wrap_idx:
                y[i, 0] = ((y[i, 0] + np.pi) % (2 * np.pi)) - np.pi
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.x[3, 0] = ((self.x[3, 0] + np.pi) % (2 * np.pi)) - np.pi
        return self.x

    # ---------- convenience updates ----------
    def update_ai(self, v_meas: float, h_meas: float) -> np.ndarray:
        z = np.array([[v_meas], [h_meas]])
        return self._apply_update(z, self.H_ai, self.R_ai, wrap_idx=[1])

    def update_gnss(self, x_meas: float, y_meas: float) -> np.ndarray:
        z = np.array([[x_meas], [y_meas]])
        return self._apply_update(z, self.H_gnss, self.R_gnss, wrap_idx=None)

    def get_state(self) -> EKFState:
        return EKFState(
            x=float(self.x[0, 0]), y=float(self.x[1, 0]),
            v=float(self.x[2, 0]), heading=float(self.x[3, 0]),
        )
