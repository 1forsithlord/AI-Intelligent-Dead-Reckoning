"""Parity: PyTorch vs ONNX runtime."""
import os
import numpy as np
import pytest
import torch
from models import TCN_Transformer

ONNX = "assets/models/saarthi_speed_v4.onnx"


@pytest.mark.skipif(not os.path.exists(ONNX), reason="run scripts/export_onnx.py first")
def test_onnx_matches_pytorch():
    import onnxruntime as ort
    model = TCN_Transformer(in_features=9).eval()
    model.load_state_dict(torch.load("best_tcn_transformer_v4.pt",
                                     map_location="cpu", weights_only=True))
    x = np.random.default_rng(0).standard_normal((32, 50, 9), dtype=np.float32)
    with torch.no_grad():
        y_pt = model(torch.from_numpy(x)).numpy()
    y_onnx = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"]).run(
        None, {"imu_window": x})[0]
    np.testing.assert_allclose(y_pt[:, 0], y_onnx[:, 0], atol=1e-4)
