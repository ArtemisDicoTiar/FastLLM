import torch
from torch import nn, Tensor
from typing import Dict, Tuple, Any


class Model(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            pad_token_id: int = 0,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        hidden_size = kwargs.get("hidden_size", 10)
        self.embeddings = nn.Embedding(32128, hidden_size)
        self.model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.logit2vocab = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_tokens: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        This function is used for model training.
        :param input_tokens: [batch_size, max_token_length]
        :param label_tokens: [batch_size, max_token_length]
        :return: [batch_size, vocab_size]
        """
        # define forward pass here

        input_ids = input_tokens["input_ids"]
        # shape: [batch_size, max_token_length, hidden_size]
        embeddings = self.embeddings(input_ids)
        # shape: [batch_size, max_token_length, hidden_size]
        hidden_states = self.model(embeddings)
        # shape: [batch_size, max_token_length, vocab_size]
        logits = self.logit2vocab(hidden_states)
        probs = torch.softmax(logits, dim=-1)

        return logits[:, -1], probs[:, -1]

    def generate(self, input_tokens: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        This function is used for evaluation.
        :param input_tokens:
        :return:
        """
        input_ids = input_tokens["input_ids"]
        # shape: [batch_size, max_token_length, hidden_size]
        embeddings = self.embeddings(input_ids)
        # shape: [batch_size, max_token_length, hidden_size]
        hidden_states = self.model(embeddings)
        # shape: [batch_size, max_token_length, vocab_size]
        logits = self.logit2vocab(hidden_states)
        probs = torch.softmax(logits, dim=-1)

        return logits[:, -1], probs[:, -1]

