import argparse
import logging

from converter import ModelConverter

parser = argparse.ArgumentParser(description="Convert an ONNX model to TF Lite Micro")
parser.add_argument("model_name", type=str, help="Unique name for model. See converter.model_converter for usage")
parser.add_argument("--model-dir", default="saved_models", help="Directory in which saved models are stored. See converter.model_converter for usage")
parser.add_argument("--data-dir", default="crowdhuman_100", help="Directory containing images which form representative dataset")
parser.add_argument("--verbose", action='store_true', help="Run the converter with verbose logging")
parser.add_argument("--onnx-to-tf-graph", action='store_true')
parser.add_argument("--tf-graph-to-tf-model", action='store_true')
parser.add_argument("--tf-model-to-tf-lite", action='store_true')
parser.add_argument("--tf-lite-to-tf-micro", action='store_true')

def main():
    args = parser.parse_args()
    # If no step-specific flags provided, default behaviour is to run all the steps
    run_all_steps = not (args.onnx_to_tf_graph or \
                         args.tf_graph_to_tf_model or \
                         args.tf_model_to_tf_lite or \
                         args.tf_lite_to_tf_micro)

    logging_level = logging.INFO if args.verbose else logging.WARNING        
    logging.basicConfig(filename='example.log',level=logging_level)
        
    converter = ModelConverter(args.model_name, args.model_dir, args.data_dir)
    
    if run_all_steps or args.onnx_to_tf_graph:
        converter.onnx_to_tf_graph()
    if run_all_steps or args.tf_graph_to_tf_model:
        converter.tf_graph_to_tf_model()
    if run_all_steps or args.tf_model_to_tf_lite:
        converter.tf_model_to_tf_lite()
    if run_all_steps or args.tf_lite_to_tf_micro:
        converter.tf_lite_to_tf_micro()

if __name__ == "__main__":
    main()
