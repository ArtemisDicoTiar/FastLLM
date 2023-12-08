from torch import nn, Tensor, softmax, cat
from typing import Optional, Tuple, Union

class CNNTextSummarizationModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        embedding_dim: int = 300,
        cnn_out_channels: int = 128,
        cnn_kernel_size: int = 3,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=pad_token_id)

        # Add CNN layer
        self.cnn = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size,
            padding=(cnn_kernel_size - 1) // 2  # Maintain the same sequence length
        )

        # Fully connected layer for prediction
        self.fc = nn.Linear(cnn_out_channels, vocab_size)

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
        # Apply the embedding layer to input_ids and decoder_input_ids
        input_embeddings = self.embedding(input_ids)
        decoder_embeddings = self.embedding(decoder_input_ids)

        # Apply attention_mask to input_embeddings
        if attention_mask is not None:
            input_embeddings = input_embeddings * attention_mask.unsqueeze(-1)

        # Apply decoder_attention_mask to decoder_embeddings
        if decoder_attention_mask is not None:
            decoder_embeddings = decoder_embeddings * decoder_attention_mask.unsqueeze(-1)

        # Concatenate input_embeddings and decoder_embeddings along the sequence dimension
        combined_embeddings = cat((input_embeddings, decoder_embeddings), dim=1)

        # Transpose embeddings to fit the Conv1d input shape
        combined_embeddings = combined_embeddings.permute(0, 2, 1)

        # Apply CNN layer
        cnn_output = self.cnn(combined_embeddings)
        print(cnn_output.shape)

        # Transpose back to (batch_size, sequence_length, embedding_dim)
        cnn_output = cnn_output.permute(0, 2, 1)
        print(cnn_output.shape)
        cnn_output = cnn_output[:, :decoder_input_ids.shape[1]]

        # Pass the CNN output through the fully connected layer
        logits = self.fc(cnn_output)
        print(logits.shape)

        # Apply softmax to get probabilities
        probabilities = softmax(logits, dim=-1)

        if return_dict:
            return {
                'logits': logits,
                'probs': probabilities
            }
        else:
            return probabilities, logits