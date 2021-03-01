#!/bin/bash

dataset='cityscapes'
quantization_type='float16' # 'float32', 'dynamic_range', 'float16', 'int8'
BUILD_SCRIPT='run_inference.py'
input_image='mit_driveseg_sample' # mit_driveseg_sample
python3 ${BUILD_SCRIPT} -d ${dataset} -q ${quantization_type} -i ${input_image} 
