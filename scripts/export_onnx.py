"""PyTorch -> ONNX with parity gate.

Uses the legacy TorchScript exporter (dynamo=False) because onnx2tf (the
ONNX->TFLite step used downstream) handles that graph more reliably than
the dynamo-exported graph.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import onnxruntime as ort
from models import TCN_Transformer

CKPT = "best_tcn_transformer_v4.pt"
OUT = "assets/models/saarthi_speed_v4.onnx"
OPSET = 17
N_PARITY = 256
TOL_VEL = 0.05    # max abs diff on normalized velocity
TOL_ANG = 0.2     # max abs diff on heading, degrees


def export():
    model = TCN_Transformer(in_features=9).eval()
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    dummy = torch.zeros(1, 50, 9, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        OUT,
        input_names=["imu_window"],
        output_names=["velocity_heading"],
        dynamic_axes={"imu_window": {0: "batch"}, "velocity_heading": {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported {OUT}")
    return model


def parity(pt_model):
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((N_PARITY, 50, 9), dtype=np.float32)
    with torch.no_grad():
        y_pt = pt_model(torch.from_numpy(x_np)).numpy()
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    y_onnx = sess.run(None, {"imu_window": x_np})[0]
    d_vel = np.abs(y_pt[:, 0] - y_onnx[:, 0])
    d_ang = np.abs(
        (np.rad2deg(np.arctan2(y_pt[:, 1], y_pt[:, 2]))
         - np.rad2deg(np.arctan2(y_onnx[:, 1], y_onnx[:, 2])) + 180.0) % 360.0 - 180.0
    )
    print(f"parity N={N_PARITY}")
    print(f"  max |dvel_norm| = {d_vel.max():.6f}  (mean {d_vel.mean():.6f})")
    print(f"  max |dheading|  = {d_ang.max():.6f} deg  (mean {d_ang.mean():.6f} deg)")
    if d_vel.max() > TOL_VEL or d_ang.max() > TOL_ANG:
        print("PARITY FAIL")
        return 1
    print("PARITY PASS")
    return 0


if __name__ == "__main__":
    m = export()
    sys.exit(parity(m))
