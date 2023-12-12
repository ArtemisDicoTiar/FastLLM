import torch
from torch import Tensor
from torch.nn import Module
from transformers import T5ForConditionalGeneration

from FastLLM.sampling.speculative_sampling import top_k, gumbel_sample


@torch.no_grad()
def base_decoding(
    net: Module,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
    filter_thres = 0.9,
):
    batch_size = prompt.shape[0]
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    # cache = None
    decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=prompt.device)
    for _ in range(sample_num_times - 1):
        output = net(
            out,
            decoder_input_ids=decoder_input_ids,
        )
        logits = output.logits if not isinstance(output, dict) else output['logits']
        logits = logits[:, -1, :]

        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample[..., None]), dim = -1)
        decoder_input_ids = torch.cat((decoder_input_ids, sample[..., None]), dim = -1)

    return decoder_input_ids