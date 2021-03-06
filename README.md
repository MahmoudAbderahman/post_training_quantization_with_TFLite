# Post-Training Quantization of Machine Learning models with TensorFlow Lite
This project is the practical implementation of the Bachelor's Thesis's topic: Benchmarking Post-Training Quantization for Optimizing Machine Learning Inference on compute-limited edge devices. 
## Task
The task of the topic is using TensorFlow Lite to optimize two machine learning models focused on image classification and semantic segmentation, respectively using the post-training quantization technique provided by TensorFlow Lite. After the model was quantized/optimized, inference was ran using the unoptimized and optimized models on a laptop (Intel-processor based) and on a Raspberry Pi 4 (ARM-processor based). Inference results were compared after running inference with both models in their quantized/unquantized formats.

The image classification model was trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and the semantic segmentation model was trained on the [Cityscapes](https://www.cityscapes-dataset.com/) dataset. 

## Evaluation metrics
The evaluation metrics for the MNIST-classification model are inference time, model size reduction after quantization and accuracy. While for the semantic segmentation task, the evaluation metrics are inference time, number of frames processed per second, model size reduction after quantization, pixel accuracy and Mean Intersection over Union (mIoU).

## Structure
In the ```docker_config``` directory, there are two directories that contain the Docker Configuration to create Docker image for each of the models to be deployed and ran on Raspberry Pi. 
The ```models``` directory contains two directories:
1. [image_classification_mnist](https://github.com/MahmoudAbderahman/post_training_quantization_with_TFLite/tree/main/models/image_classification_mnist), which contains the following directories:
      1. ```src``` directory which contains the python code for:
            1. Training the MNIST-classification model and saving the model as ```model.h5```. This is done using the [train_mnist_and_save_into_disk.py](https://github.com/MahmoudAbderahman/post_training_quantization_with_TFLite/blob/main/models/image_classification_mnist/src/train_mnist_and_save_into_disk.py) python script.
            2. Converting the original MNIST-classification TensorFlow model into TensorFlow Lite models (Float32 unquantized model, Float16 quantized model, dynamic range quantized model and INT8 quantized model). This is done using the [convert_mnist_model_to_tflite.py](https://github.com/MahmoudAbderahman/post_training_quantization_with_TFLite/blob/main/models/image_classification_mnist/src/convert_mnist_model_to_tflite.py) python script.
            3. Running inference using the all types of quantization on the 10,000 photos from the MNIST test dataset and evaluating the accuracy and inference run time. This is done using the [run_inference_tflite.py](https://github.com/MahmoudAbderahman/post_training_quantization_with_TFLite/blob/main/models/image_classification_mnist/src/run_inference_tflite.py) python script.
      2.  ```models``` directory which contains the original TensorFlow model in ```h5``` format and the all other formats to be used for inference.
      3.  ```output``` contains the results of running inference, it contains information about the size reduction, accuracy and inference run time.
2. [semantic_segmentation_cityscapes](https://github.com/MahmoudAbderahman/post_training_quantization_with_TFLite/tree/main/models/semantic_segmentation_cityscapes), which contains the following directories:
      1. ```input_model/cityscapes/``` directory which contains the following directories:
          1. ```evaluation_output_single_image``` contains the results after running inference with all types of models on a single image.
          2. ```evaluation_output_video``` contains the results after running inference with all types of models on a defined number of frames of the video in the  [video directory](https://github.com/MahmoudAbderahman/post_training_quantization_with_TFLite/tree/main/models/semantic_segmentation_cityscapes/input_model/cityscapes/video). 
          3. ```frozen_graph``` contains the frozen_graph of the semantic segmentation model, which was downloaded from [available deeplab models](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md). 
          4. ```img``` contains the ```ground_truth```, ```input_image``` and the output images (segmentation map and segmentation overlay) for the single image and video evaluation.
          5. ```quantized_models``` contains the models used for inference run.
          6. ```video``` contains the video, from which the frames are being taken to run inference on.
      2. ```convert_to_tflite``` directory contains a shell file and a python script to convert the frozen graph to quantized and unquantized models.
      3. ```run_inference``` directory contains a shell file and python script to define which type to run inference with and saves results on the disk.
