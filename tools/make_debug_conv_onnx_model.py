import torch
import torch.nn as nn

INPUT_SHAPE = (3,32,32)

model = nn.Sequential(
    torch.nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1, bias=True),
    torch.nn.BatchNorm2d(num_features=32),
    torch.nn.ReLU(), 
    torch.nn.AvgPool2d(kernel_size = (32,32), stride=1),
    torch.nn.Flatten(),
    torch.nn.Linear(32,2)
)

# Model is assumed to have one input and one output
input_names = ['model_input']
output_names = ['model_output']

dummy_input = torch.randn(1, INPUT_SHAPE)
torch.onnx.export(model, dummy_input, "debug_models/onnx/debug_conv.onnx", verbose=True, opset_version=11, input_names = input_names, output_names = output_names)
