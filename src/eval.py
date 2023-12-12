"""
This code evaluates the model performance on the validation set.
"""

from typing import List

import torch
from datasets import load_dataset
from pydantic import BaseModel, Extra
from tqdm.rich import tqdm
from transformers import (Adafactor, AutoModelForSeq2SeqLM, AutoTokenizer,
                          get_constant_schedule_with_warmup,
                          get_cosine_schedule_with_warmup)

from FastLLM.constants import DATASET_NAME, DATASET_VERSION, TARGET_MODEL_NAME, T5_DRAFTER_MODEL_NAME
from FastLLM.models.base import Model
from FastLLM.models.ngrams import NgramModel
from FastLLM.models.cnn import CNNTextSummarizationModel
from FastLLM.models.lstm import LSTMTextSummarizationModel

from FastLLM.utils import distillation_loss


class Evaluator(BaseModel, extra=Extra.allow):
    ckpt_path: str
    drafter: str
    device: int = 0
    use_ngram_drafter: bool = False
    ngram_n: int = 3

    def __init__(self, **data):
        super().__init__(**data)

    def _load_dataset(self):
        self.dataset = load_dataset(DATASET_NAME, DATASET_VERSION, split="test")
        print("Dataset loaded.")

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME)
        print("Tokenizer loaded.")

    def _load_target_model(self):
        self.target_model = AutoModelForSeq2SeqLM.from_pretrained(TARGET_MODEL_NAME)
        self.target_model.to(f"cuda:{self.device}")
        self.target_model.eval()
        print("Target model loaded.")

    def _load_draft_model(self):
        if self.use_ngram_drafter:
            self.draft_model = NgramModel(n=self.ngram_n, vocab_size=self.tokenizer.vocab_size, resume=self.ckpt_path, device=f"cuda:{self.device}")
            self.draft_model.to(f"cuda:{self.device}")
            self.draft_model.eval()
        elif self.drafter == "t5small":
            self.draft_model = AutoModelForSeq2SeqLM.from_pretrained(T5_DRAFTER_MODEL_NAME)
        elif self.drafter == "lstm":
            self.draft_model = LSTMTextSummarizationModel(
                vocab_size=self.target_model.config.vocab_size,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        elif self.drafter == "cnn":
            self.draft_model = CNNTextSummarizationModel(
                vocab_size=self.target_model.config.vocab_size,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        else:
            raise NotImplementedError(f"Drafter {self.drafter} is not implemented.")
        self.draft_model.load_state_dict(torch.load(self.ckpt_path))
        self.draft_model.to(f"cuda:{self.device}")
        self.draft_model.eval()
        print("Draft model loaded.")

    def _eval(self):
        for record in tqdm(self.dataset):
            record_id = record["id"]
            input_string = record["article"]
            label_string = record["highlights"]

            # shape: (batch_size, max_token_length)
            input_tokens = self.tokenizer(
                input_string, return_tensors="pt"
            )
            max_token_length = input_tokens["input_ids"].shape[1]

            # TODO: speculative decoding apply

            for infer_idx in range(1, max_token_length):
                current_step_input_tokens = {
                    k: v[:, :infer_idx] for k, v in input_tokens.items()
                }

                next_step_draft_tokens = self.draft_model.generate(
                    input_tokens=current_step_input_tokens,
                )

                with torch.no_grad():
                    target_tokens = self.target_model.generate(
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


    def run(self):
        self._load_dataset()
        self._load_tokenizer()
        self._load_target_model()
        self._load_draft_model()
        self._eval()
