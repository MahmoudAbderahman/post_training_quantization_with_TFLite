#!/bin/bash

frozen_graph_dir='../input_model/cityscapes/frozen_graph/frozen_inference_graph.pb' 
dataset='cityscapes'
output_dir="../input_model/cityscapes"
BUILD_SCRIPT="convert_to_tflite.py"

python3 ${BUILD_SCRIPT} -f ${frozen_graph_dir} -m ${dataset} -o ${output_dir}
