import torch
from torch import nn, Tensor


class Model(nn.Module):
    def __init__(self, ckpt_path: str = None, device: str = None):
        super().__init__()
        if ckpt_path is not None:
            self.load_state_dict(
                torch.load(ckpt_path, map_location=device)
            )
        else:
            # define model architecture here
            # this is example
            self.model = nn.Sequential(
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 10),
                nn.ReLU(),
                nn.Linear(10, 1),
            )

    def forward(self, input_tokens: dict[str, Tensor], label_tokens: dict[str, Tensor]):
        """
        This function is used for model training
        :param input_ids: input token ids
        :param label_ids: label token ids
        :return: next token logits
        """
        # define forward pass here
        next_token_logits = self.forward_pass(input_ids, label_ids)

        return next_token_logits

    def generate(self, input_tokens: dict[str, Tensor]):
        """
        This function forwards the seq2seq model.
        :param input_tokens:
        :param label_tokens:
        :return:
        """
        input_ids = input_tokens["input_ids"]
        # define generation logic
        next_token_logits = self.forward_pass(input_ids, label_ids)

        return y_t1
