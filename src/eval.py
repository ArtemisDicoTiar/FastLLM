"""
This code evaluates the model performance on the validation set.
"""
import logging

from typing import List
from datetime import datetime

import torch
import torchmetrics.text
from datasets import load_dataset
from pydantic import BaseModel, Extra
from tqdm.rich import tqdm
from transformers import (Adafactor, AutoModelForSeq2SeqLM, AutoTokenizer,
                          get_constant_schedule_with_warmup,
                          get_cosine_schedule_with_warmup)

from FastLLM.constants import DATASET_NAME, DATASET_VERSION, TARGET_MODEL_NAME, T5_DRAFTER_MODEL_NAME
from FastLLM.models.base import Model
from FastLLM.models.cnn import CNNTextSummarizationModel
from FastLLM.models.lstm import LSTMTextSummarizationModel
from FastLLM.models.ngrams import NgramModel
from FastLLM.sampling.base_decoding import base_decoding
from FastLLM.sampling.speculative_sampling import speculative_decoding
from FastLLM.utils import distillation_loss
from FastLLM.utils.benchmark import benchmark


class Evaluator(BaseModel, extra=Extra.allow):
    ckpt_path: str
    drafter: str
    device: int = 0
    exp_name: str = ""
    gen_length: int = 32
    batch_size: int = 4

    use_ngram_drafter: bool = False
    ngram_n: int = 3

    def __init__(self, **data):
        super().__init__(**data)

    def __post_init__(self):
        self.experiment_name = f"{self.drafter}_{self.exp_name}"

    def _logger_init(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        handler = logging.FileHandler(
            f"{self.experiment_name}.{current_time}.log",
        )
        handler.setLevel(logging.INFO)

        # Set the log format
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
        handler.setFormatter(formatter)

        # Get the root logger and add the file handler
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # Set the logger level to INFO
        self.logger.addHandler(handler)

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
            self.draft_model = NgramModel(n=self.ngram_n, vocab_size=self.tokenizer.vocab_size, resume=self.ckpt_path,
                                          device=f"cuda:{self.device}")
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
            raise NotImplementedError(f"drafter {self.drafter} not implemented.")

        if self.drafter != "ngram":
            log = self.draft_model.load_state_dict(torch.load(self.ckpt_path))
            print(log)
            # self.draft_model = torch.load(self.ckpt_path, map_location=f"cuda:{self.device}")
        self.draft_model.to(f"cuda:{self.device}")
        self.draft_model.eval()
        print("Draft model loaded.")

    def _metrics_init(self):
        self.target_metric = torchmetrics.text.ROUGEScore(accumulate="avg")
        # self.draft_metric = torchmetrics.text.ROUGEScore(accumulate="avg")
        self.spec_metric = torchmetrics.text.ROUGEScore(accumulate="avg")

        self.target_time = []
        # self.draft_time = []
        self.spec_time = []

        self.spec_num_accepted = []

    def _eval(self):
        for batch_index in tqdm(range(0, 8, self.batch_size)):
            batch_start = batch_index
            batch_end = batch_index + self.batch_size
            record_id = self.dataset[batch_start:batch_end]["id"]
            input_string = self.dataset[batch_start:batch_end]["article"]
            label_string = self.dataset[batch_start:batch_end]["highlights"]

            # shape: (batch_size, max_token_length)
            input_tokens = self.tokenizer(
                input_string, return_tensors="pt",
                padding="max_length", truncation=True,
                # max_length=512,
            )
            input_tokens = input_tokens.to(f"cuda:{self.device}")
            max_token_length = input_tokens["input_ids"].shape[1]

            GENERATE_LENGTH = max_token_length + self.gen_length
            target_sampled, target_base_decode_elapsed = benchmark(base_decoding)(
                self.target_model, input_tokens["input_ids"], GENERATE_LENGTH, target=True
            )
            target_base_decode_output = self.tokenizer.batch_decode(target_sampled)
            #
            # draft_sampled, draft_base_decode_elapsed = benchmark(base_decoding)(
            #     self.draft_model, input_tokens["input_ids"], GENERATE_LENGTH, target=False
            # )
            # draft_base_decode_output = self.tokenizer.batch_decode(draft_sampled)

            # set gamma value as default: 5
            GAMMA = 5
            (spec_decode_sampled, num_accepted), spec_decode_elapsed = benchmark(speculative_decoding)(
                self.target_model, self.draft_model, input_tokens["input_ids"], GENERATE_LENGTH, GAMMA,
                force_to_target=False
            )
            spec_decode_output = self.tokenizer.batch_decode(spec_decode_sampled)

            self.target_metric.update(target_base_decode_output, label_string)
            # self.draft_metric.update(draft_base_decode_output, label_string)
            self.spec_metric.update(spec_decode_output, label_string)

            self.target_time.append(target_base_decode_elapsed)
            # self.draft_time.append(draft_base_decode_elapsed)
            self.spec_time.append(spec_decode_elapsed)

            self.spec_num_accepted.append(num_accepted)

            self.logger.info(f"\ntarget decoding:\n\n{target_base_decode_output}\n")
            self.logger.info(f'target decoding in: {target_base_decode_elapsed:.3f}ms\n')

            # self.logger.info(f"\ndraft decoding:\n\n{draft_base_decode_output}\n")
            # self.logger.info(f'draft decoding in: {draft_base_decode_elapsed:.3f}ms\n')

            self.logger.info(f"\nspeculative decoding{spec_decode_output}\n")
            self.logger.info(f'speculative decoding in: {spec_decode_elapsed:.3f}ms\n')
            self.logger.info(f'num accepted: {num_accepted:.1f} / {GAMMA}\n')

        target_metric = self.target_metric.compute()
        # draft_metric = self.draft_metric.compute()
        spec_metric = self.spec_metric.compute()
        for rouge2_metric in ["rouge2_fmeasure", "rouge2_precision", "rouge2_recall"]:
            self.logger.info(f"target metric{rouge2_metric}: {target_metric[rouge2_metric]:.3f}")
            # self.logger.info(f"draft metric{rouge2_metric}: {draft_metric[rouge2_metric]:.3f}")
            self.logger.info(f"speculative metric{rouge2_metric}: {spec_metric[rouge2_metric]:.3f}")

        self.logger.info(f"average target time: {sum(self.target_time) / len(self.target_time):.3f}ms")
        # self.logger.info(f"average draft time: {sum(self.draft_time) / len(self.draft_time):.3f}ms")
        self.logger.info(f"average speculative time: {sum(self.spec_time) / len(self.spec_time):.3f}ms")

        self.logger.info(f"average speculative num accepted: {sum(self.spec_num_accepted) / len(self.spec_num_accepted):.1f}")


    def run(self):
        self.__post_init__()
        self._logger_init()
        self._load_dataset()
        self._load_tokenizer()
        self._load_target_model()
        self._load_draft_model()
        self._metrics_init()
        self._eval()
