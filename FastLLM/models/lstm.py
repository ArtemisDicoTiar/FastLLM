import torch
from torch import nn, Tensor, softmax, cat
from typing import Optional, Tuple, Union, Dict, Any


class LSTMTextSummarizationModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        embedding_dim: int = 300,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=pad_token_id)
        self.empty_embedding = nn.Embedding(num_embeddings=1, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, vocab_size)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs
    ) -> Union[Dict[str, Union[Tensor, Any]], Tuple[Tensor, Any]]:
        # Apply the embedding layer to input_ids and decoder_input_ids
        input_embeddings = self.embedding(input_ids)
        decoder_embeddings = self.embedding(decoder_input_ids)
        sep_embedding = self.empty_embedding.weight.expand(input_embeddings.size(0), -1, -1)

        # Concatenate the embeddings along the sequence dimension
        combined_embeddings = cat((input_embeddings, sep_embedding, decoder_embeddings), dim=1)

        # Assuming input_ids and decoder_input_ids are sequences, pass them through the LSTM
        lstm_output, _ = self.lstm(combined_embeddings)
        lstm_output = lstm_output[:, input_ids.shape[1]:-1]

        # Pass the LSTM output through the fully connected layer
        logits = self.fc(lstm_output)

        # Apply softmax to get probabilities
        probabilities = softmax(logits, dim=-1)

        if return_dict:
            return {
                'logits': logits,
                'probs': probabilities
            }
        else:
            return probabilities, logits

    def generate(self, input_ids) -> Tensor:
        """
        This function is used for evaluation.
        :param input_tokens:
        :return: next step tokens
        """

        with torch.no_grad():
            input_ids = input_ids.unsqueeze(0)
            device = next(self.parameters()).device
            input_ids = input_ids.to(device)
            # Pass the input_embeddings through the model to get the probabilities of next tokens
            outputs = self(input_ids=input_ids[:, :0], decoder_input_ids=input_ids)
            return outputs['logits']
