# Sources: https://colab.research.google.com/github/lexfridman/mit-deep-learning/blob/master/tutorial_driving_scene_segmentation/tutorial_driving_scene_segmentation.ipynb#scrollTo=c4oXKmnjw6i_
#          https://colab.research.google.com/github/sayakpaul/Adventures-in-TensorFlow-Lite/blob/master/DeepLabV3/DeepLab_TFLite_CityScapes.ipynb

import argparse
import pathlib

import tensorflow as tf
import numpy as np


from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

# parse the passed arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fg", required=True, help="frozen graph dir")
    parser.add_argument("-m", "--ds", required=True, help="dataset")
    parser.add_argument("-o", "--output", required=True, help="output dir")
    return vars(parser.parse_args())

def create_cityscapes_dataset_tflite_models_and_save_to_disk(frozen_graph_dir,output_dir):
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=frozen_graph_dir,
        input_arrays=['sub_7'],  # For the Xception model it needs to be `sub_7`, for MobileNet it would be `sub_2`
        output_arrays=['ResizeBilinear_2']
    )


    # convert to unquantized model
    tflite_unquantized_model = converter.convert()
    path_tflite_model = "{}/quantized_models/{}".format(output_dir, 'float32.tflite')
    open(path_tflite_model, "wb").write(tflite_unquantized_model)

    # convert to dynamic-range quantized model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dynamic_range_model = converter.convert()
    path_dynamic_range = "{}/quantized_models/{}".format(output_dir, 'dynamic_range.tflite')
    open(path_dynamic_range, "wb").write(tflite_dynamic_range_model)

    # convert to float16 quantized model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_float16_model = converter.convert()
    path_float16 = "{}/quantized_models/{}".format(output_dir, 'float16.tflite')
    open(path_float16, "wb").write(tflite_float16_model)
    
    # convert to int8 quantized model
    def representative_dataset_gen_for_cityscapse():
        for _ in range(100): 
            dummy_image = tf.random.uniform([1, 1025, 2049, 3], 0., 255., dtype=tf.float32)
            dummy_image = dummy_image / 127.5 - 1
            yield [dummy_image]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen_for_cityscapse
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #converter.inference_input_type = tf.uint8
    #converter.inference_output_type = tf.uint8
    tflite_int8_model = converter.convert()
    path_int8 = "{}/quantized_models/{}".format(output_dir, 'int8.tflite')
    open(path_int8, "wb").write(tflite_int8_model)


def main():
    parsed_args = parse_arguments()
    print(parsed_args)
    frozen_graph_dir = parsed_args["fg"]
    output_dir = parsed_args["output"]
    dataset = parsed_args["ds"]s
    create_cityscapes_dataset_tflite_models_and_save_to_disk(frozen_graph_dir, output_dir)


if __name__ == '__main__':
    main()
