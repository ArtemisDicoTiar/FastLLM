"""
This code evaluates the model performance on the validation set.
"""

import torch
from datasets import load_dataset
from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_constant_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, Adafactor

from FastLLM.constants import TARGET_MODEL_NAME, DATASET_NAME, DATASET_VERSION
from FastLLM.models.base import Model
from FastLLM.models.ngrams import NgramModel, prepare_pseudo_dataset
from FastLLM.utils import distillation_loss

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--drafter', choices=['t5small', 'lstm', 'cnn', 'ngram'], required=True)
parser.add_argument('--exp_name', required=True, type=str)
parser.add_argument('--ckpt_path', type=str)
# ngram specific parameters
parser.add_argument('--ngram_n', type=int, default=3)
parser.add_argument('--pseudo_dataset', type=str, default="./FastLLM/pseudo_dataset/pdataset.txt")


if __name__ == '__main__':
    args = parser.parse_args()

    # ============= PARAMETERs ============= #
    device = 0

    # ============= DATASET ============= #
    dataset = load_dataset(DATASET_NAME, DATASET_VERSION, split="test")

    # ============= TOKENIZER ============= #
    # this tokenizer is used for both the draft and target models
    tokenizer = AutoTokenizer.from_pretrained(
        TARGET_MODEL_NAME,
    )

    # ============= MODELs ============= #
    target_model = AutoModelForSeq2SeqLM.from_pretrained(
        TARGET_MODEL_NAME
    )
    target_model.to(f"cuda:{device}")
    target_model.eval()

    if args.drafter == 'ngram':
        if args.ckpt_path:
            print("Using a pretrained ngram model...")
            draft_model = NgramModel(args.ngram_n, tokenizer.vocab_size, device=f"cuda:{device}", resume=args.ckpt_path)
        else:
            draft_model = NgramModel(args.ngram_n, tokenizer.vocab_size, device=f"cuda:{device}")
            print("Fitting the ngram model on the pseudo dataset...")
            pseudo_dataset = prepare_pseudo_dataset(args.pseudo_dataset, tokenizer)
            draft_model.fit(pseudo_dataset)
    else:
        draft_model: Model = Model()
        if args.ckpt_path:
            draft_model.load_state_dict(torch.load(args.ckpt_path))

    draft_model.to(f"cuda:{device}")
    draft_model.eval()

    # ============= Evaluation ============= #
    for record in tqdm(dataset):
        record_id = record["id"]
        input_string = record["article"]
        label_string = record["highlights"]

        # shape: (batch_size, max_token_length)
        input_tokens = tokenizer(input_string, return_tensors="pt")
        max_token_length = input_tokens["input_ids"].shape[1]

        # TODO: speculative decoding apply

        for infer_idx in range(1, max_token_length):
            current_step_input_tokens = {
                k: v[:, :infer_idx] for k, v in input_tokens.items()
            }

            next_step_draft_tokens = draft_model.generate(
                input_tokens=current_step_input_tokens,
            )

            with torch.no_grad():
                target_tokens = target_model.generate(
                    input_ids=current_step_input_tokens["input_ids"],
                    attention_mask=current_step_input_tokens["attention_mask"],
                    # TODO: check the hyperparameters
                    # max_length=512,
                    # num_beams=4,
                    # early_stopping=True,
                    # no_repeat_ngram_size=3,
                    # num_return_sequences=1,
                )

        # TODO: evaluation metrics implementation
