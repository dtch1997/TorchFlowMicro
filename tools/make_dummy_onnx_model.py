import torch
import torch.nn as nn


model = nn.Sequential(
    nn.Linear(300, 100, bias=True),
    nn.ReLU(),
    nn.Linear(100, 10, bias=True)
)

dummy_input = torch.randn(1, 300)
torch.onnx.export(model, dummy_input, "saved_models/onnx/dummy.onnx", verbose=True, opset_version=11)
