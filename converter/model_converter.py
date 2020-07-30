import onnx
from onnx_tf.backend import prepare

import os
import logging
import tensorflow as tf

from utils import make_parentdir_if_not_exist
from converter.convert_graph_to_saved_model import convert_graph_to_saved_model

class ModelConverter:
    """
    Converter will look for:
        ONNX model at self.onnx_path
        TF Graph at self.tf_graph_path
        TF SavedModel at self.tf_model_dir
        TFLite model at self.tf_lite_path
        TFMicro model at self.tf_micro_path
    """
    def __init__(self, name: str, model_dir = "saved_models"):
        self.name = name
        self.model_dir = model_dir
        self.logger = logging.getLogger(self.name)
        
    def onnx_to_tf_graph(self):
        self.logger.info(f"Converting ONNX model at {self.onnx_path} to TF Graph at {self.tf_graph_path}")
        onnx_model = onnx.load(self.onnx_path)
        # Check the model is well-formed
        onnx.checker.check_model(onnx_model)
        tf_rep = prepare(onnx_model)
        make_parentdir_if_not_exist(self.tf_graph_path)
        tf_rep.export_graph(self.tf_graph_path)
        return self
    
    def tf_graph_to_tf_model(self):
        self.logger.info(f"Converting TF Graph at {self.tf_graph_path} to TF SavedModel at {self.tf_model_dir}")
        make_parentdir_if_not_exist(self.tf_model_dir)
        convert_graph_to_saved_model(self.tf_graph_path, self.tf_model_dir)
        return self
    
    def tf_model_to_tf_lite(self):
        self.logger.info(f"Converting TF SavedModel at {self.tf_model_dir} to TFLite model at {self.tf_lite_path}")
        converter = tf.lite.TFLiteConverter.from_saved_model(self.tf_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_model = converter.convert()
        make_parentdir_if_not_exist(self.tf_lite_path)
        open(self.tf_lite_path, "wb").write(quantized_model)
        return self
        
    def tf_lite_to_tf_micro(self):
        self.logger.info(f"Converting TFLite model at {self.tf_lite_path} to TFMicro model at {self.tf_micro_path}")
        make_parentdir_if_not_exist(self.tf_micro_path)
        cmd = f"xxd -i {self.tf_lite_path} > {self.tf_micro_path}"
        os.system(cmd)
        return self
    
    @property
    def onnx_path(self) -> str:
        return f"{self.model_dir}/onnx/{self.name}.onnx"
    
    @property
    def tf_graph_path(self) -> str:
        return f"{self.model_dir}/tf_graph/{self.name}/graph.pb"
    
    @property
    def tf_model_dir(self) -> str:
        return f"{self.model_dir}/tf_model/{self.name}"

    @property
    def tf_lite_path(self) -> str:
        return f"{self.model_dir}/tf_lite/{self.name}.tflite"


    @property
    def tf_micro_path(self) -> str:
        return f"{self.model_dir}/tf_micro/{self.name}/model_data.cc"


