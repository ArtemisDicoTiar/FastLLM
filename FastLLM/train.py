"""
This code runs distillation on the draft model.
Ref: https://arxiv.org/pdf/2310.08461.pdf

"""

import torch
from datasets import load_dataset
from torch.optim import AdamW
from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from FastLLM.constants import TARGET_MODEL_NAME, DATASET_NAME, DATASET_VERSION
from FastLLM.models.base import Model

# ============= PARAMETERs ============= #
device = 0

n_epochs = 1

learning_rate = 1e-3

distil_method = NotImplemented

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
# https://arxiv.org/pdf/2310.08461.pdf
loss_fns = {
    # Forward KL Divergence
    "Seq_KD": NotImplemented,
    # Forward KL Divergence
    "Supervised_KD": NotImplemented,
    # Forward KL Divergence
    "Imit_KD": NotImplemented,
    # Forward KL Divergence
    "f_Distill": NotImplemented,
    # Forward KL Divergence / Jensen-Shannon Divergence
    "on_policy_GKD": NotImplemented,
}
loss_fn = loss_fns[distil_method]

# ============= OPTIMIZER ============= #
optimizer: AdamW = torch.optim.AdamW(
    params=draft_model.parameters(),
    lr=learning_rate,
)

training_steps = 0
global_step = 0
for epoch in range(n_epochs):
    for record in tqdm(dataset):
        record_id = record["id"]
        input_string = record["article"]
        label_string = record["highlights"]

        # shape: (batch_size, max_token_length)
        input_tokens = tokenizer(input_string, return_tensors="pt")
        max_token_length = input_tokens["input_ids"].shape[1]

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
        training_steps += 1
    global_step += 1
