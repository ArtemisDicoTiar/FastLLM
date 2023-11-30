"""
This code runs distillation on the draft model.
Ref: https://arxiv.org/pdf/2310.08461.pdf

"""

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR
from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_constant_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, Adafactor

from FastLLM.constants import TARGET_MODEL_NAME, DATASET_NAME, DATASET_VERSION
from FastLLM.models.base import Model
from FastLLM.utils import distillation_loss

if __name__ == '__main__':
    # ============= PARAMETERs ============= #
    device = 0

    distil_method = NotImplemented

    # both should be in [0, 1]
    fixed_data_fraction = 0.1
    drafter_data_fraction = 0.1

    # https://arxiv.org/pdf/2310.08461.pdf
    training_steps = 300_000
    batch_size = 32

    learning_rate = 3e-4
    learning_rate_warmup_steps = 5_000
    learning_rate_cooldown_step_start = 150_000
    learning_rate_cooldown_step_end = 300_000

    dropout = 0.0
    n_epochs = 1

    # ============= DATASET ============= #
    dataset = load_dataset(DATASET_NAME, DATASET_VERSION, split="train")

    # ============= TOKENIZER ============= #
    # this tokenizer is used for both the draft and target models
    tokenizer = AutoTokenizer.from_pretrained(
        TARGET_MODEL_NAME,
    )

    # ============= MODELs ============= #
    target_model = AutoModelForSeq2SeqLM.from_pretrained(
        TARGET_MODEL_NAME,
    )
    target_model.to(f"cuda:{device}")
    target_model.eval()

    draft_model: Model = ...  # device=device

    # ============= LOSS FUNCTION (Distillation) ============= #
    loss_fn = distillation_loss.fns[distil_method]

    # ============= OPTIMIZER ============= #
    # https://arxiv.org/pdf/2310.08461.pdf
    optimizer = Adafactor(
        draft_model.parameters(),
        lr=learning_rate,
        relative_step=False,
    )
    warmup_scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=learning_rate_warmup_steps
    )
    cooldown_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=learning_rate_cooldown_step_end - learning_rate_cooldown_step_start,
        eta_min=0.1 * learning_rate,
        last_epoch=-1
    )
    scheduler = SequentialLR(
        optimizer,
        [warmup_scheduler, cooldown_scheduler],
        [learning_rate_cooldown_step_start]
    )

    # ============= Train ============= #
    for epoch in range(n_epochs):
        for record in tqdm(dataset):
            record_id = record["id"]
            input_string = record["article"]
            label_string = record["highlights"]

            # shape: (batch_size, max_token_length)
            input_tokens = tokenizer(input_string, return_tensors="pt")
            max_token_length = input_tokens["input_ids"].shape[1]

            # TODO: distillation implementation

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

                loss = loss_fn(
                    next_step_draft_tokens,
                    target_tokens,
                )

                loss.backward()

            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
