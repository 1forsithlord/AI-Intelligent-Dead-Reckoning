"""Smoke test: the checkpoint loads into the module and forward runs."""
import pytest
import torch
from models import TCN_Transformer

CKPT = "best_tcn_transformer_v4.pt"


def test_load_and_forward():
    model = TCN_Transformer(in_features=9).eval()
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt, strict=True)
    assert not missing, f"missing: {missing}"
    assert not unexpected, f"unexpected: {unexpected}"

    with torch.no_grad():
        out = model(torch.zeros(1, 50, 9))
    assert out.shape == (1, 3)
    # heading components should be near a unit circle before normalization
    norm = (out[0, 1:] ** 2).sum().sqrt().item()
    assert 0.0 < norm < 2.0  # untrained-from-ckpt sanity only
