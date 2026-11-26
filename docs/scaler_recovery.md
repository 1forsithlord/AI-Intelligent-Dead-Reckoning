# Recovering the V4 scaler values

The .pt file contains only model weights. The FixedGroupScaler parameters
(x_mean, x_scale, v_mean, v_std) were never saved. Without them, on-device
inference produces garbage.

## Fix - one Kaggle cell (30 seconds)

Run this in a fresh Kaggle cell after loading s_m (Driver B sensor df):

    import json, numpy as np
    IMU_COLS = ["ACCELEROMETER X","ACCELEROMETER Y","ACCELEROMETER Z",
                "GYROSCOPE Yaw","GYROSCOPE Pitch","GYROSCOPE Roll",
                "MAGNETIC FIELD X","MAGNETIC FIELD Y","MAGNETIC FIELD Z"]
    VEL_COL = "Velocity (km/hr)"
    X = s_m[IMU_COLS].to_numpy(np.float32)
    rms = lambda a: float(np.sqrt(np.mean((a - a.mean())**2))) or 1.0
    out = {
        "x_mean":  np.repeat([X[:,0:3].mean(), X[:,3:6].mean(), X[:,6:9].mean()], 3).tolist(),
        "x_scale": np.repeat([rms(X[:,0:3]), rms(X[:,3:6]), rms(X[:,6:9])], 3).tolist(),
        "v_mean":  float(s_m[VEL_COL].mean()),
        "v_std":   float(s_m[VEL_COL].std()),
    }
    print(json.dumps(out, indent=2))

Copy the output over models/scaler_v4.json (remove the "placeholder": true key).
Verify:

    python -c "from models import FixedGroupScaler; s = FixedGroupScaler.load('models/scaler_v4.json'); print(s.v_mean, s.v_std)"
