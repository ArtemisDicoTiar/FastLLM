#!/bin/bash

# This script is a bash script using ../pseudo_dataset.py to generate a pseudo dataset for the FastLLM algorithm.
# It assumes that the dataset is already splitted.
# It runs the pseudo_dataset.py on a single split using a single gpu.
# It takes 2 arguments:
# 1. GPU number on which the pseudo dataset will be generated
# 2. Split number of the dataset on which the pseudo dataset will be generated

GPU_NUMBER=$1
SPLIT_NUMBER=$2

SPLITS_DIR="../splits"
PSEUDO_DATASET_OUTPUT_DIR="../pseudo_dataset"

CUDA_VISIBLE_DEVICES=$GPU_NUMBER nohup python ../pseudo_dataset.py --generate_pseudo_dataset --split_number $SPLIT_NUMBER --splits_dir $SPLITS_DIR --pseudo_dataset_output_dir $PSEUDO_DATASET_OUTPUT_DIR > "../logs/${SPLIT_NUMBER}-${GPU_NUMBER}.log" 2>&1 &