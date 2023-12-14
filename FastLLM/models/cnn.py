import torch
from torch import nn, Tensor, softmax, cat
from typing import Optional, Tuple, Union, Dict, Any


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
        self.prev_embedding = nn.Embedding(num_embeddings=1, embedding_dim=embedding_dim)
        self.sep_embedding = nn.Embedding(num_embeddings=1, embedding_dim=embedding_dim)

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
    ) -> Union[Dict[str, Union[Tensor, Any]], Tuple[Tensor, Any]]:
        # Apply the embedding layer to input_ids and decoder_input_ids
        input_embeddings = self.embedding(input_ids)
        decoder_embeddings = self.embedding(decoder_input_ids)
        prev_embedding = self.prev_embedding.weight.expand(input_embeddings.size(0), -1, -1)
        sep_embedding = self.sep_embedding.weight.expand(input_embeddings.size(0), -1, -1)

        # Concatenate input_embeddings and decoder_embeddings along the sequence dimension
        combined_embeddings = cat((prev_embedding, input_embeddings, sep_embedding, decoder_embeddings), dim=1)

        # Transpose embeddings to fit the Conv1d input shape
        combined_embeddings = combined_embeddings.permute(0, 2, 1)

        # Apply CNN layer
        cnn_output = self.cnn(combined_embeddings)

        # Transpose back to (batch_size, sequence_length, embedding_dim)
        cnn_output = cnn_output.permute(0, 2, 1)
        cnn_output = cnn_output[:, input_ids.shape[1]+1:-1]

        # Pass the CNN output through the fully connected layer
        logits = self.fc(cnn_output)

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
            next_token_probs = outputs['probs']  # tensor representing the probabilities of next tokens

            # Get the predicted token indices (assuming you want the most likely token)
            _, predicted_indices = torch.max(next_token_probs, dim=-1)

            return predicted_indices[0]