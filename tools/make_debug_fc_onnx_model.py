import torch
import torch.nn as nn

INPUT_SHAPE = (300,)

model = nn.Sequential(
    nn.Linear(300, 100, bias=True),
    nn.ReLU(),
    nn.Linear(100, 10, bias=True)
)

# Model is assumed to have one input and one output
input_names = ['model_input']
output_names = ['model_output']

dummy_input = torch.randn(1, *INPUT_SHAPE)
torch.onnx.export(model, dummy_input, "debug_models/onnx/debug_fc.onnx", verbose=True, opset_version=11, input_names = input_names, output_names = output_names)
