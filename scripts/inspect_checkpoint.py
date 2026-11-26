import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from models import TCN_Transformer

path = sys.argv[1] if len(sys.argv) > 1 else "best_tcn_transformer_v4.pt"
ckpt = torch.load(path, map_location="cpu", weights_only=True)
model = TCN_Transformer(in_features=9)

missing, unexpected = model.load_state_dict(ckpt, strict=False)
print(f"loaded: {len(ckpt)} keys")
print(f"missing: {missing or 'none'}")
print(f"unexpected: {unexpected or 'none'}")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
buffers = sum(b.numel() for n, b in model.named_buffers())
print(f"trainable params: {trainable:,}")
print(f"buffer elements: {buffers:,}")
print(f"total tensors: {trainable + buffers:,}")

# shape smoke test
with torch.no_grad():
    out = model(torch.zeros(1, 50, 9))
print(f"forward OK — output shape {tuple(out.shape)}")
