# Source: https://www.tensorflow.org/lite/performance/post_training_integer_quant
import time

import tensorflow as tf
import numpy as np
import os
from pathlib import Path

# get the keras dataset to use for the inference
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# define the output file
OUTPUT_FILE = 'inference_output.txt'
output_file_dir = os.path.abspath('../output/laptop/inference_output/'+OUTPUT_FILE)
file = open(output_file_dir, "w+")


# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices):
    global test_images

    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.zeros((len(test_image_indices),), dtype=int)
    for i, test_image_index in enumerate(test_image_indices):
        test_image = test_images[test_image_index]
        test_label = test_labels[test_image_index]

        # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point

        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        predictions[i] = output.argmax()

    return predictions


import matplotlib.pylab as plt

# Change this to test a different image
test_image_index = 9


## Helper function to test the models on one image
def test_model(tflite_file, test_image_index, model_type):
    global test_labels

    predictions = run_tflite_model(tflite_file, [test_image_index])
    plt.savefig('{}-sample_output.png'.format(model_type))
    plt.imshow(test_images[test_image_index])
    template = model_type + " Model \n True:{true}, Predicted:{predict}"
    _ = plt.title(template.format(true=str(test_labels[test_image_index]), predict=str(predictions[0])))
    plt.grid(False)
    plt.show()



# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file, model_type):
    global test_images
    global test_labels

    test_image_indices = range(test_images.shape[0])
    predictions = run_tflite_model(tflite_file, test_image_indices)

    accuracy = (np.sum(test_labels == predictions) * 100) / len(test_images)
    file.write('%s model accuracy is %.4f%% (Number of test samples=%d)\n' % (
        model_type, accuracy, len(test_images)))
    print('%s model accuracy is %.4f%% (Number of test samples=%d)\n' % (
        model_type, accuracy, len(test_images)))

# test on one image
test_model('../models/converted_model_f32.tflite', test_image_index, model_type="Converted TF Lite Model Without quantization")
test_model('../models/dynamic_range_quantized.tflite', test_image_index, model_type="Dynamic Range Quantization")
test_model('../models/float16_quantized.tflite', test_image_index, model_type="Float16 Quantization")
test_model('../models/int8_quantized.tflite', test_image_index, model_type="INT8 Quantized")

# evaluate the model on 10,000 samples and calculate the inference time for the float32 unquantized model
time_before_inference_no_quant = time.time()
evaluate_model('../models/converted_model_f32.tflite', model_type="Converted TF Lite Model Without quantization")
time_after_inference_no_quant = time.time()

# evaluate the model on 10,000 samples and calculate the inference time for the dynamic range quantized model
time_before_inference_dr = time.time()
evaluate_model('../models/dynamic_range_quantized.tflite', model_type="Dynamic Range Quantization")
time_after_inference_dr = time.time()

# evaluate the model on 10,000 samples and calculate the inference time for the float16 quantized model
time_before_inference_float16 = time.time()
evaluate_model('../models/float16_quantized.tflite', model_type="Float16 Quantization")
time_after_inference_float16 = time.time()

# evaluate the model on 10,000 samples and calculate the inference time for the int8 quantized model
time_before_inference_full_int = time.time()
evaluate_model('../models/int8_quantized.tflite', model_type="INT8 Quantized")
time_after_inference_full_int = time.time()

# write the actual inference time and the evaluation results into the file.
inference_time_no_quantization = np.round(time_after_inference_no_quant - time_before_inference_no_quant, 3)
file.write("Inference time with no quantization: {} seconds.\n".format(inference_time_no_quantization))
print("Inference time with no quantization: ", inference_time_no_quantization, "seconds.")

inference_time_dynamic_range = np.round(time_after_inference_dr - time_before_inference_dr, 3)
file.write("Inference time with dynamic range quantization: {} seconds.\n".format(inference_time_dynamic_range))
print("Inference time with dynamic range: ", inference_time_dynamic_range, "seconds.")

inference_time_float16 = np.round(time_after_inference_float16 - time_before_inference_float16, 3)
file.write("Inference time with float16 quantization: {} seconds.\n".format(inference_time_float16))
print("Inference time float16: ", inference_time_float16, "seconds.")


inference_time_int8 = np.round(time_after_inference_full_int - time_before_inference_full_int, 3)
file.write("Inference time with int8 quantization: {} seconds.\n".format(inference_time_int8))
print("Inference time int8: ", inference_time_int8, "seconds.")

