import torch
import torch.nn as nn
from torch import Tensor
from collections import defaultdict
from typing import Tuple, Dict, List
from tqdm.rich import tqdm
import pickle
import os


class NgramModel(nn.Module):
    def __init__(
        self,
        n: int,
        vocab_size: int,
        laplace_smoothing: float = 1.0,
        device: str = "cuda",
        resume=None,
        *args,
        **kwargs,
    ):
        """
        This class implements a ngram model. It is used to compute the probability of the next token given the previous prefix.
        :param n: The n in ngram. It is the number of previous tokens to consider.
        :param vocab_size: The size of the vocabulary
        :param laplace_smoothing: The laplace smoothing parameter
        :param device: The device to use for the computations
        :param resume: The path to a saved model folder to resume training, if None, the model is built from scratch
        """
        super(NgramModel, self).__init__(*args, **kwargs)

        assert n >= 1, "n must be greater than 0"

        self.n = n
        self.vocab_size = vocab_size
        self.laplace_smoothing = laplace_smoothing
        self.fitted = False
        if resume is not None:
            with open(os.path.join(resume, "{}.ckpt".format(n)), "rb") as f:
                self.ngram_counts, self.total_counts = pickle.load(f)
        else:
            self.ngram_counts = defaultdict(lambda: defaultdict(float))
            self.total_counts = defaultdict(
                lambda: self.vocab_size * self.laplace_smoothing
            )
        self.sm = nn.Softmax(dim=0)
        self.is_unigram = n == 1
        self.unigram_logits = None
        self.unigram_probabilities = None
        self.backoff = (
            None
            if self.is_unigram
            else NgramModel(n - 1, vocab_size, laplace_smoothing, device, resume)
        )
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def forward(self, input_ids: Tensor, input_tokens: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        This function is used for model inference.
        :param input_ids: input_ids of the tokenized prefix
        :return: logits and probabilities of the next tokens
        """

        assert self.fitted, "The model must be fitted before being used."

        if input_tokens is not None:
            input_ids = input_tokens["input_ids"][0]

        # Case 1: Uni-gram model
        if self.is_unigram:
            return self.unigram_logits, self.unigram_probabilities

        logits = torch.zeros(self.vocab_size, device=self.device, dtype=torch.float32)

        # Case where prefix is lower than n
        # Use a backoff model
        if len(input_ids) < self.n - 1 and not self.is_unigram:
            return self.backoff(input_ids)

        # Case 2: n-gram model with n > 1

        # Fetch the ngram from the prefix (last n-1 tokens)
        ngram = tuple([tensor.item() for tensor in input_ids[-(self.n - 1) :]])

        # Compute the logits and probabilities
        counts = torch.tensor(
            [self.ngram_counts[ngram][i] for i in range(self.vocab_size)],
            dtype=torch.float32,
            device=self.device,
        )
        probabilities = (counts + self.laplace_smoothing) / self.total_counts[ngram]
        logits = torch.log(probabilities)
        probabilities = self.sm(logits)

        return logits, probabilities

    def generate(self, input_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """
        This function is used for model inference. (Same as a forward pass)
        """
        return self.forward(input_ids)

    def fit(self, data: List[Dict[str, Tensor]]) -> None:
        """
        This function is used to fit the model on the data. It is not a training since the model is not trainable.
        It only builds the ngram counts based on the given data.
        :param data: The data to fit the model on. List of tokenized sentences on the give vocabulary
        """
        if not self.is_unigram:
            for sentence in tqdm(
                data,
                desc="Fitting {}-gram model".format(self.n),
                total=len(data),
                miniters=20,
            ):
                sequence = sentence["input_ids"][0]
                for i in range(len(sequence) - self.n + 1):
                    ngram = tuple(
                        [tensor.item() for tensor in sequence[i : i + self.n - 1]]
                    )
                    next_token = sequence[i + self.n - 1]
                    self.ngram_counts[ngram][next_token.item()] += 1.0
                    self.total_counts[ngram] += 1.0

            print("Training backoff model of size {}...".format(self.n - 1))
            self.backoff.fit(data)

        else:
            for sentence in tqdm(
                data, desc="Fitting uni-gram model", total=len(data), miniters=20
            ):
                sequence = sentence["input_ids"][0]
                for i in range(len(sequence)):
                    next_token = sequence[i]
                    self.total_counts[next_token.item()] += 1.0

            counts = torch.tensor(
                [self.total_counts[i] for i in range(self.vocab_size)],
                dtype=torch.float32,
                device=self.device,
            )
            probabilities = (counts + self.laplace_smoothing) / (
                self.vocab_size * self.laplace_smoothing
            )
            self.unigram_logits = torch.log(probabilities)
            self.unigram_probabilities = self.sm(self.unigram_logits)

        self.fitted = True

    def save(self, path: str) -> None:
        """
        This function is used to save the model to disk.
        :param path: The path to the folder where the model will be saved
        """
        print("Saving {}-gram model...".format(self.n))
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path, "{}.ckpt".format(self.n)), "wb") as f:
            pickle.dump((self.ngram_counts, self.total_counts), f)
        if not self.is_unigram:
            self.backoff.save(path)


def prepare_pseudo_dataset(pseudo_dataset_dir: str, tokenizer) -> List[Dict[str, Tensor]]:
    """
    This function is used to prepare the data of the pseudo dataset for the ngram model fitting.
    :param pseudo_dataset_dir: The path to the pseudo dataset
    :param tokenizer: The tokenizer to use
    :return: The data to fit the model on. List of tokenized sentences on the give vocabulary
    """
    print("Preparing data...")
    data = []
    with open(pseudo_dataset_dir, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, total=len(lines), miniters=20, desc="Tokenizing pseudo-dataset"):
            data.append(tokenizer(line, return_tensors="pt"))
    return data