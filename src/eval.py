"""
This code evaluates the model performance on the validation set.
"""

import torch
from datasets import load_dataset
from tqdm.rich import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    Adafactor,
)

from FastLLM.constants import TARGET_MODEL_NAME, DATASET_NAME, DATASET_VERSION
from FastLLM.models.base import Model
from FastLLM.utils import distillation_loss


def eval():
    # ============= PARAMETERs ============= #
    ckpt_path = ""
    device = 0

    # ============= DATASET ============= #
    dataset = load_dataset(DATASET_NAME, DATASET_VERSION, split="train")

    # ============= TOKENIZER ============= #
    # this tokenizer is used for both the draft and target models
    tokenizer = AutoTokenizer.from_pretrained(
        TARGET_MODEL_NAME,
    )

    # ============= MODELs ============= #
    target_model = AutoModelForSeq2SeqLM.from_pretrained(TARGET_MODEL_NAME)
    target_model.to(f"cuda:{device}")
    target_model.eval()

    draft_model: Model = Model()
    draft_model.load_state_dict(torch.load(ckpt_path))
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
