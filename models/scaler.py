"""FixedGroupScaler — the exact preprocessing used during V4 training.

CRITICAL: inference parity depends on these values being identical to training.
Source: dead-sih-main.ipynb cell 24.

x_mean  = [mean(acc[:3])]*3 + [mean(gyro[:3])]*3 + [mean(mag[:3])]*3
x_scale = [rms(acc[:3])] *3 + [rms(gyro[:3])] *3 + [rms(mag[:3])] *3
where rms(a) = std(a - mean(a))  (equivalently, std(a))

encode_y(v, h_deg) -> [(v - v_mean)/v_std, sin(h_rad), cos(h_rad)]
decode_y(y)        -> (v, heading_deg)
"""
import json
import numpy as np


class FixedGroupScaler:
    """Scales the 3D vectors but ALSO subtracts the mean (crucial for gravity)."""
    def fit(self, df, imu_cols, vel_col):
        X = df[imu_cols].to_numpy(np.float32)
        self.x_mean = np.repeat(
            [X[:, 0:3].mean(), X[:, 3:6].mean(), X[:, 6:9].mean()], 3
        ).astype(np.float32)
        rms = lambda a: float(np.sqrt(np.mean((a - a.mean()) ** 2))) or 1.0
        self.x_scale = np.repeat(
            [rms(X[:, 0:3]), rms(X[:, 3:6]), rms(X[:, 6:9])], 3
        ).astype(np.float32)
        self.v_mean = float(df[vel_col].mean())
        self.v_std = float(df[vel_col].std() or 1.0)
        return self

    def transform_x(self, arr):
        return ((arr - self.x_mean) / (self.x_scale + 1e-8)).astype(np.float32)

    def encode_y(self, vel, heading_deg):
        v = (vel - self.v_mean) / (self.v_std + 1e-8)
        h = np.deg2rad(heading_deg)
        return np.stack([v, np.sin(h), np.cos(h)], axis=-1).astype(np.float32)

    def decode_y(self, y):
        y = np.asarray(y)
        return y[..., 0] * self.v_std + self.v_mean, np.rad2deg(np.arctan2(y[..., 1], y[..., 2])) % 360.0

    # -------- persistence (needed for TFLite/Dart parity) --------
    def to_dict(self):
        return {
            "x_mean": self.x_mean.tolist(),
            "x_scale": self.x_scale.tolist(),
            "v_mean": self.v_mean,
            "v_std": self.v_std,
        }

    @classmethod
    def from_dict(cls, d):
        if d.get("placeholder"):
            raise RuntimeError(
                "scaler_v4.json is a PLACEHOLDER. Recover real values from Kaggle "
                "(see docs/scaler_recovery.md) before deploying."
            )
        s = cls()
        s.x_mean = np.asarray(d["x_mean"], dtype=np.float32)
        s.x_scale = np.asarray(d["x_scale"], dtype=np.float32)
        s.v_mean = float(d["v_mean"])
        s.v_std = float(d["v_std"])
        return s

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))
