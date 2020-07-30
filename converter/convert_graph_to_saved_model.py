# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:52:55 2020

@author: Daniel Tan
"""


import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

def convert_graph_to_saved_model(graph_path: str, model_dir: str):
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(model_dir)
    
    with tf.io.gfile.GFile(graph_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        
    sigs = {}
    
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        # name="" is important to ensure we don't get spurious prefixing
        tf.compat.v1.import_graph_def(graph_def, name="")
        g = tf.compat.v1.get_default_graph()

        """
        Names here are determined by "input_names" and "output_names"
        arguments in the initial call to torch.onnx.export(...)

        The pipeline assumes that these have been set to:
            input_names = ['model_input']
            output_names = ['model_output']
        """ 
        inp = g.get_tensor_by_name("model_input:0")
        out = g.get_tensor_by_name("model_output:0")
    
        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
                {"in": inp}, {"out": out})
    
        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)
    
    builder.save()
