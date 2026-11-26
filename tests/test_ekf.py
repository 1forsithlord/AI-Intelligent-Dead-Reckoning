"""GNSS-outage benchmark for the EKF."""
import numpy as np
import pytest
from core.fusion.ekf import ExtendedKalmanFilter


def simulate(n_steps, dt=0.1, v0=15.0, h0=0.5, yaw_rate=0.05):
    v = v0 * np.ones(n_steps)
    h = h0 + yaw_rate * np.arange(n_steps) * dt
    x = np.cumsum(v * np.cos(h) * dt)
    y = np.cumsum(v * np.sin(h) * dt)
    return x, y, v, h


def test_ekf_tracks_ground_truth_with_gnss():
    """With AI + GNSS every step, final position error stays small (< 20 m)."""
    np.random.seed(0)
    n = 600
    x_t, y_t, v_t, h_t = simulate(n)
    ekf = ExtendedKalmanFilter(dt=0.1)
    ekf.set_state(x_t[0], y_t[0], v_t[0], h_t[0])

    for k in range(n):
        gyro = (h_t[k] - h_t[k - 1]) / 0.1 if k > 0 else 0.0
        ekf.predict(gyro_yaw_rate=gyro)
        ekf.update_ai(v_t[k] + np.random.randn() * 0.5,
                      h_t[k] + np.random.randn() * 0.05)
        ekf.update_gnss(x_t[k] + np.random.randn() * 2.0,
                        y_t[k] + np.random.randn() * 2.0)

    err = float(np.hypot(ekf.x[0, 0] - x_t[-1], ekf.x[1, 0] - y_t[-1]))
    print(f"GNSS-available: 60s, final position error = {err:.2f} m")
    assert err < 20.0, f"with GNSS, final position error = {err:.2f} m"


def test_ekf_survives_60s_gnss_outage():
    """60 s outage with AI-only updates: drift < 5% of distance traveled."""
    np.random.seed(1)
    n = 1200
    outage_start = 600
    x_t, y_t, v_t, h_t = simulate(n)
    ekf = ExtendedKalmanFilter(dt=0.1)
    ekf.set_state(x_t[0], y_t[0], v_t[0], h_t[0])

    for k in range(n):
        gyro = (h_t[k] - h_t[k - 1]) / 0.1 if k > 0 else 0.0
        ekf.predict(gyro_yaw_rate=gyro)
        ekf.update_ai(v_t[k] + np.random.randn() * 0.3,
                      h_t[k] + np.random.randn() * 0.02)
        if k < outage_start:
            ekf.update_gnss(x_t[k] + np.random.randn() * 2.0,
                            y_t[k] + np.random.randn() * 2.0)

    dist = float(sum(v_t[outage_start:] * 0.1))
    err = float(np.hypot(ekf.x[0, 0] - x_t[-1], ekf.x[1, 0] - y_t[-1]))
    pct = 100.0 * err / dist
    print(f"60s outage: traveled {dist:.0f} m, final error {err:.1f} m ({pct:.2f}%)")
    assert pct < 5.0, f"60s outage drift = {pct:.2f}% (should be < 5%)"


def test_adaptive_hooks_change_noise():
    ekf = ExtendedKalmanFilter()
    ekf.set_process_noise(np.eye(4) * 0.5)
    ekf.set_ai_noise(np.eye(2) * 2.0)
    assert ekf.Q[0, 0] == 0.5
    assert ekf.R_ai[0, 0] == 2.0
