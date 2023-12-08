from torch import nn, Tensor, softmax, cat
from typing import Optional, Tuple, Union

class CNNTextSummarizationModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int = 0,
        embedding_dim: int = 300,
        num_filters: int = 256,
        filter_sizes: Tuple[int, int, int] = (3, 4, 5),
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=pad_token_id)
        
        # Define 1D CNN layers with different filter sizes
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.embedding_dim,
                out_channels=num_filters,
                kernel_size=filter_size,
                padding=filter_size // 2  # Maintain the same length after convolution
            )
            for filter_size in filter_sizes
        ])
        
        self.fc = nn.Linear(num_filters * len(filter_sizes), vocab_size)

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

        # Pass the embeddings through 1D CNN layers
        conv_outputs = [conv1d(decoder_embeddings.transpose(1, 2)).transpose(1, 2) for conv1d in self.conv1d_list]
        cnn_output = cat(conv_outputs, dim=-1)

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
