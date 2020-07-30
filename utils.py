import onnx
from onnx_tf.backend import prepare

import os
import tensorflow as tf
import pathlib

class Model:
    def __init__(self, name: str)
        self.name = name

    @property
    def onnx_name(self) -> str:
        return f"{self.name}.onnx"
    
    @property
    def onnx_path(self) -> str:
        return f"saved_models/onnx/{self.onnx_path}"
    
    @property
    def tf_dir(self) -> str:
        return f"saved_models/tf/{self.name}"

    @property
    def tf_path(self) -> str:
        return f"{self.tf_dir}/saved_model.pb"

    @property
    def tflite_path(self) -> str:
        return f"saved_models/tflite/{self.name}.tflite"

    @property
    def micro_dir(self) -> str:
        return f"saved_models/tflite/{self.name}"

    @property
    def micro_path(self) -> str:
        return f"{self.micro_dir}/model_data.cc"

def make_dir_if_not_exist(dirpath: str)
    dirpath = pathlib.Path(dirpath)
    if dirpath.exists() and dirpath.is_dir():
        return
    else:
        dirpath.mkdir(parents=True, exist_ok=True)

def convert_onnx_to_tf(model_name: str):
    model = Model(model_name)
    onnx_model = onnx.load(model.onnx_path)
    # Check the model is well-formed
    onnx.checker.check_model(onnx_model)
    tf_rep = prepare(onnx_model)
    make_dir_if_not_exist(model.tf_dir)
    tf_rep.export_graph(model.tf_path)

def convert_tf_to_tflite(model_name: str):
    model = Model(model_name)
    converter = tf.lite.TFLiteConverter.from_saved_model(model.tf_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_model = converter.convert()
    make_dir_if_not_exist(model.tflite_dir)
    open(model.tflite_path, "wb").write(quantized_model)
    
def convert_tflite_to_micro(model_name: str):
    model = Model(model_name)
    cmd = f"xxd -i {model.
    
