"""
This code runs distillation on the draft model.
Ref: https://arxiv.org/pdf/2310.08461.pdf

"""
import argparse
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR
from tqdm.rich import tqdm
from transformers import (
    Adafactor,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
)

from FastLLM.constants import (
    DATASET_NAME,
    DATASET_VERSION,
    T5_DRAFTER_MODEL_NAME,
    TARGET_MODEL_NAME,
)
from FastLLM.models.base import Model
from FastLLM.models.cnn import CNNTextSummarizationModel
from FastLLM.models.lstm import LSTMTextSummarizationModel
from FastLLM.utils import distillation_loss

parser = argparse.ArgumentParser()
parser.add_argument("--drafter", choices=["t5small", "lstm", "cnn"], required=True)
parser.add_argument("--exp_name", required=True, type=str)
parser.add_argument("--loss", default=1, type=float)
parser.add_argument("--kd_loss", default=1, type=float)
parser.add_argument("--kd_temp", default=8, type=float)
parser.add_argument(
    "--distil_method",
    default="Seq_KD",
    choices=["Seq_KD", "Supervised_KD", "Imit_KD", "f_Distill", "on_policy_GKD"],
)


def train():
    args = parser.parse_args()

    # ============= Experiment NAME ============= #
    drafter_model_name = f"{args.drafter}_{args.exp_name}"

    # ============= PATH ============= #
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = f"./{drafter_model_name}-drafter-{current_time}.pt"

    # ============= Logger ============= #
    max_file_size_bytes = 10 * 1024 * 1024  # 10 MB
    handler = RotatingFileHandler(
        f"{model_save_path}.log", maxBytes=max_file_size_bytes, backupCount=10
    )
    handler.setLevel(logging.INFO)

    # Set the log format
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)

    # Get the root logger and add the rotating file handler
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the logger level to INFO
    logger.addHandler(handler)

    # ============= SEED ============= #
    torch.manual_seed(42)

    # ============= PARAMETERs ============= #
    device = 0

    distil_method = args.distil_method

    # both should be in [0, 1]
    fixed_data_fraction = 0.1
    drafter_data_fraction = 0.1

    # https://arxiv.org/pdf/2310.08461.pdf
    training_steps = 35000
    batch_size = 8

    learning_rate = 3e-4
    learning_rate_warmup_steps = 5_000
    learning_rate_cooldown_step_start = 10000
    learning_rate_cooldown_step_end = 30000

    dropout = 0.0
    n_epochs = 1

    # ============= DATASET ============= #
    dataset = load_dataset(DATASET_NAME, DATASET_VERSION, split="train")
    dataset = dataset.map(
        function=lambda batch: batch, batched=True, batch_size=batch_size
    )

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

    if args.drafter == "t5small":
        draft_model = AutoModelForSeq2SeqLM.from_pretrained(T5_DRAFTER_MODEL_NAME)
    if args.drafter == "lstm":
        draft_model = LSTMTextSummarizationModel(
            vocab_size=target_model.config.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
        )
    if args.drafter == "cnn":
        draft_model = CNNTextSummarizationModel(
            vocab_size=target_model.config.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
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
        last_epoch=-1,
    )
    scheduler = SequentialLR(
        optimizer,
        [warmup_scheduler, cooldown_scheduler],
        [learning_rate_cooldown_step_start],
    )

    # ============= Train ============= #
    num_step = 0

    logger.info("Train Start!")

    while num_step < training_steps:
        for batch_index in tqdm(range(0, len(dataset), batch_size)):
            batch_start = batch_index
            batch_end = batch_index + batch_size
            record_id = dataset[batch_start:batch_end]["id"]
            input_string = dataset[batch_start:batch_end]["article"]
            label_string = dataset[batch_start:batch_end]["highlights"]

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

            with torch.no_grad():
                target_tokens = target_model.forward(
                    input_ids=input_tokens["input_ids"],
                    attention_mask=input_tokens["attention_mask"],
                    decoder_input_ids=label_tokens["input_ids"],
                    decoder_attention_mask=label_tokens["attention_mask"],
                )
                target_model_logits = target_tokens.logits

            draft_tokens = draft_model.forward(
                input_ids=input_tokens["input_ids"],
                attention_mask=input_tokens["attention_mask"],
                decoder_input_ids=label_tokens["input_ids"],
                decoder_attention_mask=label_tokens["attention_mask"],
            )
            draft_logits = (
                draft_tokens.logits
                if not isinstance(draft_tokens, dict)
                else draft_tokens["logits"]
            )
            target_logits = target_tokens.logits
            label_ids = label_tokens["input_ids"]
            num_classes = draft_logits.shape[-1]

            loss = torch.nn.CrossEntropyLoss(reduction="none")(
                draft_logits.view(-1, num_classes), label_ids.view(-1)
            )
            loss = (
                args.loss
                * loss.view_as(label_ids)[label_tokens["attention_mask"]].mean()
            )

            kd_loss = loss_fn(draft_logits, target_logits, temperature=args.kd_temp)
            kd_loss = kd_loss.sum(dim=-1)
            kd_loss = args.kd_loss * kd_loss[label_tokens["attention_mask"]].mean()

            final_loss = loss + kd_loss
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            scheduler.step()

            logger.info(f"Step {num_step}: loss {loss}, kd_loss: {kd_loss}")

            num_step += 1
            # save draft model for every 100 steps
            if num_step % 100 == 0:
                torch.save(draft_model.state_dict(), model_save_path)