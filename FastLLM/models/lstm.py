from torch import nn, Tensor, softmax
from typing import Optional, Tuple, Union

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
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # Assuming input_ids and decoder_input_ids are used for input embeddings
        input_embeddings = self.embedding(input_ids)
        decoder_embeddings = self.embedding(decoder_input_ids)

        # Assuming input_ids and decoder_input_ids are sequences, pass them through the LSTM
        lstm_output, _ = self.lstm(decoder_embeddings)

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
