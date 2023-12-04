"""
This code runs distillation on the draft model.
Ref: https://arxiv.org/pdf/2310.08461.pdf

"""
from datetime import datetime

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR
from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_constant_schedule_with_warmup, Adafactor

from FastLLM.constants import TARGET_MODEL_NAME, DATASET_NAME, DATASET_VERSION
from FastLLM.models.base import Model
from FastLLM.utils import distillation_loss


if __name__ == '__main__':
    # ============= Experiment NAME ============= #
    drafter_model_name = "example"

    # ============= PATH ============= #
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = f"./{drafter_model_name}-drafter-{current_time}.pt"

    # ============= SEED ============= #
    torch.manual_seed(42)

    # ============= PARAMETERs ============= #
    device = 0

    distil_method = "Seq_KD"

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
    dataset = dataset.map(function=lambda batch: batch, batched=True, batch_size=batch_size)

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

    draft_model: Model = Model(
        vocab_size=target_model.config.vocab_size,
        pad_token_id=tokenizer.pad_token_id
    )
    draft_model.to(f"cuda:{device}")
    draft_model.train()

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
        for batch_index in tqdm(range(0, len(dataset), batch_size)):
            batch_start = batch_index
            batch_end = batch_index + batch_size
            record_id = dataset[batch_start: batch_end]["id"]
            input_string = dataset[batch_start: batch_end]["article"]
            label_string = dataset[batch_start: batch_end]["highlights"]

            # shape: (batch_size, max_token_length)
            input_tokens = tokenizer(
                input_string,
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).to(f"cuda:{device}")
            label_tokens = tokenizer(
                label_string,
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).to(f"cuda:{device}")
            max_token_length = input_tokens["input_ids"].shape[1]

            # TODO: distillation implementation

            for infer_idx in range(1, max_token_length):
                current_step_label_tokens = {
                    k: v[:, :infer_idx-1] for k, v in label_tokens.items()
                }
                total_input_ids = {
                    k: torch.cat([input_tokens[k], current_step_label_tokens[k]], dim=-1)
                    for k, _ in input_tokens.items()
                }

                next_token_draft_logit, next_token_draft_probs = draft_model(
                    input_tokens=total_input_ids,
                )

                with torch.no_grad():
                    target_tokens = target_model.generate(
                        input_ids=total_input_ids["input_ids"],
                        attention_mask=total_input_ids["attention_mask"],
                        max_new_tokens=1,
                        output_scores=True,
                        return_dict_in_generate=True,
                        # TODO: check the hyperparameters
                        # num_beams=4,
                        # early_stopping=True,
                        # no_repeat_ngram_size=3,
                        # num_return_sequences=1,
                    )
                    next_token_target_tokens = target_tokens["sequences"][:, -1]
                    next_token_target_logits = target_tokens["scores"][0]
                    next_token_target_probs = torch.softmax(next_token_target_logits, dim=-1)

                # this is example for Seq_KD
                loss = loss_fn(
                    next_token_draft_logit.log_softmax(dim=-1),
                    next_token_target_probs,
                )

                loss.backward()

            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

    # save draft model
    torch.save(draft_model.state_dict(), model_save_path)
