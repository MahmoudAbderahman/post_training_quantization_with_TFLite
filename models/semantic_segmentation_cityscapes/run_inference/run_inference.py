# Sources: https://colab.research.google.com/github/lexfridman/mit-deep-learning/blob/master/tutorial_driving_scene_segmentation/tutorial_driving_scene_segmentation.ipynb#scrollTo=c4oXKmnjw6i_
#          https://colab.research.google.com/github/sayakpaul/Adventures-in-TensorFlow-Lite/blob/master/DeepLabV3/DeepLab_TFLite_CityScapes.ipynb

import argparse
import os
import tarfile
import time

import IPython
from PIL import ImageOps

import PIL
import tensorflow as tf
import numpy as np
import cv2 as cv
from PIL.Image import Image
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from matplotlib import pyplot as plt, gridspec
from tqdm import tqdm
from datetime import datetime

OUTPUT_FILE = None

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--ds", required=True, help="data set")
    parser.add_argument("-q", "--qt", required=True, help="quantization type")
    parser.add_argument("-i", "--im", required=True, help="input image")
    return vars(parser.parse_args())


def vis_segmentation_stream(image, seg_map, index):
    """Visualizes segmentation overlay view and stream it with IPython display."""
    plt.figure(figsize=(12, 7))

    seg_image = label_to_color_image_for_cityscapes(seg_map).astype(np.uint8)
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay | frame #%d' % index)
    plt.grid('off')
    plt.tight_layout()
    output_dir = os.path.abspath('../input_model/cityscapes/img/')
    images_output_dir = os.path.join(output_dir, 'output_figures_video/raspberrypi/int8/')
    #images_output_dir = os.path.join(output_dir, 'output_figures_video/laptop/int8/')
    index_for_filename = index + 1
    plt.savefig(images_output_dir + 'figure:' + str(index_for_filename) + '.png')
    plt.close()


def run_visualization_video(frame, index):
    """Inferences DeepLab model on a video file and stream the visualization."""
    original_im = PIL.Image.fromarray(frame[..., ::-1])
    cropped_image, seg_map = get_segmentation_map(original_im)
    vis_segmentation_stream(cropped_image, seg_map, index)


def run_inference_video_cityscapes():
    """Run inference on the number of frames passed to the script and get the results of the evaluation metrics."""
    parsed_arguments = parse_arguments()
    quantization_type = parsed_arguments['qt']
    
    OUTPUT_FILE = ('{}-{}.txt').format(quantization_type,datetime.now().strftime("%d_%m_%Y %H:%M:%S"))
    
    output_file_dir = os.path.abspath('../input_model/cityscapes/evaluation_output_video/raspberrypi/int8/' + OUTPUT_FILE)
    #output_file_dir = os.path.abspath('../input_model/cityscapes/evaluation_output_video/laptop/int8/' + OUTPUT_FILE)
    file = open(output_file_dir, "w+")
    print("Running inference on cityscapes dataset with quantization type: {} and sample video.\n".format(quantization_type))
    file.write("Running inference on cityscapes dataset with quantization type: {} and sample video.\n".format(quantization_type))
    #print("Evaluation results: \n\n")
    file.write("Evaluation results: \n\n")
    dataset_name = get_dataset_name()
    SAMPLE_VIDEO = '../input_model/{}/video/mit_driveseg_sample.mp4'.format(dataset_name)

    print('running deeplab on the sample video...')

    video = cv.VideoCapture(SAMPLE_VIDEO)
    # num_frames = 598  # uncomment to use the full sample video
    num_frames = 5
    print("Number of frames to run on: {}.\n".format(num_frames))
    file.write("Number of frames to run on: {}.\n".format(num_frames))
    time_before_inference = time.time()
    try:
        for i in range(num_frames):
            _, frame = video.read()
            if not _: break
            run_visualization_video(frame, i)
            IPython.display.clear_output(wait=True)
    except KeyboardInterrupt:
        plt.close()
        print("Stream stopped.")

    print('evaluating on the sample video...', flush=True)

    acc = []
    intersection = []
    union = []

    for i in tqdm(range(num_frames)):
        _, frame = video.read()
        original_im = PIL.Image.fromarray(frame[..., ::-1])
        cropped_image, seg_map = get_segmentation_map(original_im)
        SAMPLE_GT = '../input_model/cityscapes/img/ground_truth/mit_driveseg_sample_gt .tar.gz'
        dataset = DriveSeg(SAMPLE_GT)
        gt = dataset.fetch(i)
        seg_map = cv.resize(seg_map, (1920, 1080), interpolation=cv.INTER_NEAREST)

        _acc, _intersection, _union = evaluate_single_image_for_cityscapes(seg_map, gt)
        intersection.append(_intersection)
        union.append(_union)
        acc.append(_acc)
    time_after_inference = time.time()
    inference_time = np.round((time_after_inference - time_before_inference), 3)
    frames_per_second = num_frames / inference_time

    class_iou = np.round(np.sum(intersection, 0) / np.sum(union, 0), 4)
    file.write('pixel accuracy: %.4f\n' % np.mean(acc))
    print('pixel accuracy: %.4f' % np.mean(acc))
    file.write('mean class IoU: %.4f' % np.mean(class_iou))

    print('mean class IoU: %.4f' % np.mean(class_iou))
    print('class IoU:')
    file.write('\nclass IoU:\n')

    print(tabulate([class_iou], headers=LABEL_NAMES_CITYSCAPES[[0, 1, 2, 5, 6, 7, 8, 9, 11, 13]]))
    file.write(tabulate([class_iou], headers=LABEL_NAMES_CITYSCAPES[[0, 1, 2, 5, 6, 7, 8, 9, 11, 13]]))
    print('\nInference time: {} seconds.'.format(inference_time))
    file.write('\nInference time: {} seconds.'.format(inference_time))
    print('\nFrames Per Second: {} frames.'.format(np.round(frames_per_second, 3)))
    file.write('\nFrames Per Second: {} frames.'.format(np.round(frames_per_second, 3)))

    file.close()


def label_to_color_image_for_cityscapes(label):
    """Adds color defined by the dataset colormap to the label.

        Args:
            label: A 2D array with integer type, storing the segmentation label.

        Returns:
            result: A 2D array with floating type. The element of the array
                is the color indexed by the corresponding element in the input label
                to the PASCAL color map.

        Raises:
            ValueError: If label is not of rank 2 or its value is larger than color
                map maximum entry.
        """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_label_colormap_for_cityscapes()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def create_label_colormap_for_cityscapes():
    """Creates a label colormap used in Cityscapes segmentation benchmark.

        Returns:
            A Colormap for visualizing segmentation results.
        """
    colormap = np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 0]], dtype=np.uint8)
    return colormap


def get_quantized_model_path():
    """Get the path of the quantized model passed to the script."""
    parsed_arguments = parse_arguments()
    dataset = parsed_arguments['ds']
    quantization_type = parsed_arguments['qt']
    model_path = "../input_model/{}/quantized_models/{}.tflite".format(dataset, quantization_type)
    return model_path


def get_cropped_image(image, input_size):
    """Crop the image to the required input shape from the model"""
    old_size = image.size  # old_size is in (width, height) format
    desired_ratio = input_size[0] / input_size[1]
    old_ratio = old_size[0] / old_size[1]

    if old_ratio < desired_ratio:  # '<': cropping, '>': padding
        new_size = (old_size[0], int(old_size[0] / desired_ratio))
    else:
        new_size = (int(old_size[1] * desired_ratio), old_size[1])

    #print(new_size, old_size)

    # Cropping the original image to the desired aspect ratio
    delta_w = new_size[0] - old_size[0]
    delta_h = new_size[1] - old_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    cropped_image = ImageOps.expand(image, padding)
    return cropped_image


def get_segmentation_map(SAMPLE_IMAGE):
    """Get the segmentation map after running inference on the passed image."""
    model_path = get_quantized_model_path()
    interpreter = tf.lite.Interpreter(model_path=model_path)

    # Set model input.
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()

    # Get image size - converting from BHWC to WH # ([1,1025,2049,19]: Shape of ResizeBilinear_2 op)
    input_size = input_details[0]['shape'][2], input_details[0]['shape'][1]
    #print(input_size)

    image = SAMPLE_IMAGE

    cropped_image = get_cropped_image(image, input_size)

    # Resize the cropped image to the desired model size
    resized_image = cropped_image.convert('RGB').resize(input_size, PIL.Image.BILINEAR)

    # Convert to a NumPy array, add a batch dimension, and normalize the image.
    image_for_prediction = np.asarray(resized_image).astype(np.float32)
    image_for_prediction = np.expand_dims(image_for_prediction, 0)
    image_for_prediction = image_for_prediction / 127.5 - 1

    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=model_path)

    # Invoke the interpreter to run inference.
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], image_for_prediction)
    interpreter.invoke()

    # Retrieve the raw output map.
    raw_prediction = interpreter.tensor(
        interpreter.get_output_details()[0]['index'])()

    width, height = cropped_image.size
    seg_map = tf.argmax(tf.image.resize(raw_prediction, (height, width)), axis=3)
    seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)

    return cropped_image, seg_map

def vis_segmentation_for_cityscapes_image(image, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(20, 4))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image_for_cityscapes(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP_CITYSCAPES[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES_CITYSCAPES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    #plt.show()

    output_dir = os.path.abspath('../input_model/cityscapes/img/')
    
    images_output_dir = os.path.join(output_dir, 'output_figure_single_image/laptop/float16/')
    # index_for_filename = index + 1
    plt.savefig(images_output_dir + get_image_name() + '.png')


# cityscapes classes used during the inference.
LABEL_NAMES_CITYSCAPES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle', 'void'])
# reshape the classes names
FULL_LABEL_MAP_FOR_CITYSCAPES = np.arange(len(LABEL_NAMES_CITYSCAPES)).reshape(len(LABEL_NAMES_CITYSCAPES), 1)

FULL_COLOR_MAP_CITYSCAPES = label_to_color_image_for_cityscapes(FULL_LABEL_MAP_FOR_CITYSCAPES)


class DriveSeg(object):
    """Class to load ground truth Dataset."""

    def __init__(self, tarball_path):
        self.tar_file = tarfile.open(tarball_path)
        self.tar_info = self.tar_file.getmembers()

    def fetch(self, index):
        """Get ground truth by index.

        Args:
            index: The frame number.

        Returns:
            gt: Ground truth segmentation map.
        """
        tar_info = self.tar_info[index + 1]  # exclude index 0 which is the parent directory
        file_handle = self.tar_file.extractfile(tar_info)
        gt = np.frombuffer(file_handle.read(), np.uint8)
        gt = cv.imdecode(gt, cv.IMREAD_COLOR)
        gt = gt[:, :, 0]  # select a single channel from the 3-channel image
        gt[gt == 255] = 19  # void class, does not count for accuracy
        return gt


def evaluate_single_image_for_cityscapes(seg_map, ground_truth):
    """Evaluate a single frame with the MODEL loaded."""
    # merge label due to different annotation scheme
    seg_map[np.logical_or(seg_map == 14, seg_map == 15)] = 13
    seg_map[np.logical_or(seg_map == 3, seg_map == 4)] = 2
    seg_map[seg_map == 12] = 11

    # calculate accuracy on valid area
    acc = np.sum(seg_map[ground_truth != 19] == ground_truth[ground_truth != 19]) / np.sum(ground_truth != 19)

    # select valid labels for evaluation
    cm = confusion_matrix(ground_truth[ground_truth != 19], seg_map[ground_truth != 19],
                          labels=np.array([0, 1, 2, 5, 6, 7, 8, 9, 11, 13]))
    intersection = np.diag(cm)
    union = np.sum(cm, 0) + np.sum(cm, 1) - np.diag(cm)
    return acc, intersection, union


def get_input_image_path():
    parsed_arguments = parse_arguments()
    dataset = parsed_arguments['ds']
    input_image = parsed_arguments['im']
    if dataset == 'cityscapes':
        return ('../input_model/{}/img/input_image/{}.png'.format(dataset, input_image))
    else:
        return ('../input_model/{}/img/input_image/{}.jpg'.format(dataset, input_image))


def get_image_name():
    parsed_arguments = parse_arguments()
    return parsed_arguments['im']


def get_dataset_name():
    parsed_args = parse_arguments()
    dataset_name = parsed_args['ds']
    return dataset_name


def run_inference_single_image_cityscapes():
    """Method to run the inference on a single image and get results of the evaluation metrics."""
    parsed_arguments = parse_arguments()
    quantization_type = parsed_arguments['qt']
    image_name = get_image_name()
    OUTPUT_FILE = ('{}-{}.txt').format(quantization_type, datetime.now().strftime("%d_%m_%Y %H:%M:%S"))
    output_file_dir = os.path.abspath('../input_model/cityscapes/evaluation_output_single_image/laptop/float16/'+OUTPUT_FILE)

    file = open(output_file_dir, "w+")
    file.write("Running inference on cityscapes dataset with quantization type: {} and image: {}.\n".format(quantization_type,image_name))
    file.write("Evaluation results: \n\n")

    input_image = PIL.Image.open(get_input_image_path())

    cropped_image, seg_map = get_segmentation_map(input_image)

    vis_segmentation_for_cityscapes_image(cropped_image, seg_map)

    SAMPLE_GT = '../input_model/cityscapes/img/ground_truth/mit_driveseg_sample_gt .tar.gz'
    dataset = DriveSeg(SAMPLE_GT)

    gt = dataset.fetch(0)  # sample image is frame 0
    print('visualizing ground truth annotation on the sample image...')
    time_before_inference = time.time()
    #vis_segmentation_for_cityscapes_image(input_image, gt)

    print('evaluating on the sample image...')

    gt = dataset.fetch(0)  # sample image is frame 0
    seg_map = cv.resize(seg_map, (1920, 1080), interpolation=cv.INTER_NEAREST)
    acc, intersection, union = evaluate_single_image_for_cityscapes(seg_map, gt)
    time_after_inference = time.time()
    class_iou = np.round(intersection / union, 5)
    print('pixel accuracy: %.5f' % acc)
    print('mean class IoU:', np.mean(class_iou))
    print('class IoU:')
    print(tabulate([class_iou], headers=LABEL_NAMES_CITYSCAPES[[0, 1, 2, 5, 6, 7, 8, 9, 11, 13]]))
    file.write('pixel accuracy: %.5f' % acc)
    file.write('\nmean class IoU:{}'.format(np.mean(class_iou)))
    file.write('\nclass IoU:\n')
    file.write(tabulate([class_iou], headers=LABEL_NAMES_CITYSCAPES[[0, 1, 2, 5, 6, 7, 8, 9, 11, 13]]))
    inference_time = np.round((time_after_inference - time_before_inference), 3)
    file.write('\nInference time: {} seconds.'.format(inference_time))
    file.close()



def main():
    parsed_args = parse_arguments()
    print(parsed_args)
    dataset = parsed_args["ds"]
    run_inference_video_cityscapes()
    #run_inference_single_image_cityscapes()
    

if __name__ == '__main__':
    main()
