# Source: https://www.tensorflow.org/lite/performance/post_training_integer_quant
import logging
import os
from pathlib import Path

import keras

#logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
import numpy as np

# define the output text file
OUTPUT_FILE = ("size_changes.txt")
output_file_dir = os.path.abspath('../output/conversion_output/'+OUTPUT_FILE)
file = open(output_file_dir, "w+")

# get the keras model
keras_model = keras.models.load_model('../models/model.h5')
keras_model_size = Path('../models/model.h5').stat().st_size

# write keras model size into the text file
print(keras_model_size)
file.write("Size of the original keras model is: {} bytes.\n".format(keras_model_size))

# convert the keras model to TFLite without quantization and write the size in the file
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
float32_model = converter.convert()
open("../models/converted_model_f32.tflite", "wb").write(float32_model)
float32_model_size = Path('../models/converted_model_f32.tflite').stat().st_size
file.write("Size of the converted model without quantization is: {} bytes.\n".format(float32_model_size))
print(float32_model_size)

# convert the keras model to TFLite with dynamic range quantization and write the size in the file
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
dynamic_range_quantized = converter.convert()
open("../models/dynamic_range_quantized.tflite", "wb").write(dynamic_range_quantized)
dynamic_range_quantized_size = Path('../models/dynamic_range_quantized.tflite').stat().st_size
file.write("Size of the dynamic range quantized model is: {} bytes.\n".format(dynamic_range_quantized_size))
print(dynamic_range_quantized_size)

# convert the keras model to TFLite with float16 quantization and write the size in the file
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
float16_quantized = converter.convert()
open("../models/float16_quantized.tflite", "wb").write(float16_quantized)
float16_quantized_size = Path('::/models/float16_quantized.tflite').stat().st_size
file.write("Size of the float16 quantized model is: {} bytes.\n".format(float16_quantized_size))
print(float16_quantized_size)

# get the test images and preprocess the images before quantizing into int8
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0


def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    yield [input_value]

# convert the keras model to TFLite with int8 quantization and write the size in the file
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
int8_quantized = converter.convert()
open("../models/int8_quantized.tflite", "wb").write(int8_quantized)
int8_quantized_size = Path('../models/int8_quantized.tflite').stat().st_size
file.write("Size of the int8 quantized model is: {} bytes.".format(int8_quantized_size))
file.close()
print(int8_quantized_size)


