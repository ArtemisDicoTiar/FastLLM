#!/bin/bash

# This script is a bash script using ../pseudo_dataset.py to generate a pseudo dataset for the FastLLM algorithm.
# First it splits the dataset into n. (n is the number of gpus)
# Then it runs the pseudo_dataset.py on each split using different gpus in parralel to generate the pseudo dataset.
# Finally it merges the pseudo datasets into one dataset.

SPLITS_DIR="../splits"
PSEUDO_DATASET_OUTPUT_DIR="../pseudo_dataset"
PSEUDO_DATASET_NAME="pdataset"

GPUs=(0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7) # gpus to use (Used RTX3090 24.576MB, could fit 2 splits on one gpu)
N=${#GPUs[@]}
PIDS=()

# Making sure that all processes are killed when the script is killed
trap "kill 0" SIGINT
trap "kill 0" INT TERM HUP

# Cleaning up
echo "Cleaning up..."
rm -rf $SPLITS_DIR
rm -rf $PSEUDO_DATASET_OUTPUT_DIR
rm -rf "./logs"
mkdir "./logs"

# Splitting the dataset
echo "Splitting the dataset into $N"
python ../pseudo_dataset.py --split_dataset --number_of_splits $N --output_dir $SPLITS_DIR

# Generating the pseudo dataset parallelly on different gpus
echo "Starting to generate the pseudo dataset on $N gpus..."
for ((i=0;i<N;i++)); do
    echo "Starting process on gpu ${GPUs[$i]}"
    CUDA_VISIBLE_DEVICES=${GPUs[$i]} nohup python ../pseudo_dataset.py --generate_pseudo_dataset --split_number $i --splits_dir $SPLITS_DIR --pseudo_dataset_output_dir $PSEUDO_DATASET_OUTPUT_DIR > "./logs/gpu${GPUs[$i]}_log.txt" 2>&1 &
    PIDS+=($!)
    disown
done

# Checking the status of the processes
sleep 30
echo "Status:"

while :; do
    echo "--------------- $(date) ---------------"
    all_done=true
    for ((i=0;i<N;i++)); do
        if kill -0 ${PIDS[$i]} 2>/dev/null; then
            all_done=false
            echo "GPU ${GPUs[$i]}: $(tail -n 1 "./logs/gpu${GPUs[$i]}_log.txt")"
            echo "" > "./logs/gpu${GPUs[$i]}_log.txt"
        else
            echo "GPU ${GPUs[$i]}: done"
        fi
    done
    $all_done && break
    sleep 30
done

# Merging the pseudo datasets splits
echo "All processes finished"

echo "Merging the pseudo datasets into one dataset"
python ../pseudo_dataset.py --merge_generated_data --generated_pseudo_dataset_dir $PSEUDO_DATASET_OUTPUT_DIR --merged_output_dir $PSEUDO_DATASET_OUTPUT_DIR --merged_output_name $PSEUDO_DATASET_NAME

echo "Done"