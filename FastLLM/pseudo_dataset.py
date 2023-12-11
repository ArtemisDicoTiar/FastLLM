import argparse
import os
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from constants import TARGET_MODEL_NAME, DATASET_NAME, DATASET_VERSION


def split_dataset(data, number_of_splits=4, output_dir="./splits"):
    if not data:
        return
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # split the dataset into n splits
    splits = []
    for i in range(number_of_splits):
        splits.append(data.shard(number_of_splits, i))
    
    # save the splits
    for i, split in enumerate(splits):
        split.save_to_disk(os.path.join(output_dir, f"split_{i}"))


def generate_pseudo_dataset(model, tokenizer, splitted_data, output_dir="./pseudo_dataset", split_number=0):
    if not splitted_data:
        return
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # generate the pseudo dataset
    with open(os.path.join(output_dir, f"split_{split_number}.txt"), 'w') as f:
        for text in tqdm(enumerate(splitted_data), total=len(splitted_data), desc=f"Generating pseudo dataset from split {split_number}", mininterval=20):
            inputs = tokenizer.encode(text[1]["highlights"], padding=True, truncation=True, return_tensors="pt")
            inputs = inputs.to(model.device)
            outputs = model.generate(inputs, max_length=2500)
            outputs = [o.cpu() for o in outputs]
            outputs = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            for item in outputs:
                f.write(f"{item}\n")
                f.flush() # flush the buffer to disk (needed for script to work properly)
    

def merge_generated_data(generated_data_dir="./pseudo_dataset", output_dir="./pseudo_dataset", output_name="pdataset"):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # merge the generated data
    generated = []
    for i in tqdm(range(len(os.listdir(generated_data_dir))), total=len(os.listdir(generated_data_dir)), desc="Reading data"):
        with open(os.path.join(generated_data_dir, f"split_{i}.txt"), 'r') as f:
            generated.append(f.readlines())
    generated = [item for sublist in generated for item in sublist]

    # save the generated data
    with open(os.path.join(output_dir, f"{output_name}.txt"), 'w') as f:
        for item in generated:
            f.write(f"{item}\n")
            f.flush()


if __name__ == '__main__':
    # ============= Gather Arguments ============= #
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dataset", action="store_true")
    parser.add_argument("--number_of_splits", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./splits")
    parser.add_argument("--generate_pseudo_dataset", action="store_true")
    parser.add_argument("--splits_dir", type=str, default="./splits")
    parser.add_argument("--split_number", type=int, default=0)
    parser.add_argument("--pseudo_dataset_output_dir", type=str, default="./pseudo_dataset")
    parser.add_argument("--merge_generated_data", action="store_true")
    parser.add_argument("--generated_pseudo_dataset_dir", type=str, default="./pseudo_dataset")
    parser.add_argument("--merged_output_dir", type=str, default="./pseudo_dataset")
    parser.add_argument("--merged_output_name", type=str, default="pdataset")
    
    args = parser.parse_args()

    # ============= Split Dataset ============= #
    # split the dataset into n splits and save them to disk
    # this is used for parallel inference
    # parameters:
    #  --split_dataset: needed to split the dataset
    #  --number_of_splits: number of splits to create (default: 4)
    #  --output_dir: directory to save the splits (default: ./splits)
    if args.split_dataset:
        split_dataset(load_dataset(DATASET_NAME, DATASET_VERSION, split="train"), args.number_of_splits, args.output_dir)
        exit()

    # ============= Generate Pseudo Dataset ============= #
    # generate the pseudo dataset from the splits
    # parameters:
    #  --generate_pseudo_dataset: needed to generate the pseudo dataset
    #  --pseudo_dataset_dir: directory containing the splits (default: ./splits)
    #  --split_number: which split to generate the pseudo dataset from (default: 0)
    #  --pseudo_dataset_output_dir: directory to save the pseudo dataset (default: ./pseudo_dataset)
    if args.generate_pseudo_dataset:
        device = 0
        model = AutoModelForSeq2SeqLM.from_pretrained(TARGET_MODEL_NAME)
        model.to(f"cuda:{device}")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)
        generate_pseudo_dataset(model, tokenizer, load_from_disk(os.path.join(args.splits_dir, f"split_{args.split_number}")), args.pseudo_dataset_output_dir, args.split_number)
        exit()

    # ============= Merge Generated Data ============= #
    # merge the generated data into a single dataset
    # parameters:
    #  --merge_generated_data: needed to merge the generated data
    #  --generated_pseudo_dataset_dir: directory containing the generated data (default: ./pseudo_dataset)
    #  --merged_output_dir: directory to save the merged data (default: ./pseudo_dataset)
    #  --merged_output_name: name of the merged data (default: pdataset)
    if args.merge_generated_data:
        merge_generated_data(args.generated_pseudo_dataset_dir, args.merged_output_dir, args.merged_output_name)
        exit()