"""
Reference: https://github.com/lucidrains/speculative-decoding/blob/main/speculative_decoding/speculative_decoding.py
"""
import math

import torch
from functorch.einops import rearrange
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from einops import rearrange
from transformers import T5ForConditionalGeneration


# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# sampling helpers


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs


# speculative decoding functions


def safe_div(num, den, eps=1e-10):
    return num / max(den, eps)


def find_first_true_index(bool_tensor, dim=-1):
    return (bool_tensor.cumsum(dim=dim) == 0).sum(dim=dim)


@torch.no_grad()
def speculative_decoding(
    tokenizer,
    net: Module,
    small_net: Module,
    prompt: Tensor,
    seq_len: int,
    gamma: int = 5,
    temperature=1.0,
    filter_thres=0.9,
    lenience=1.0,
    pad_id=0,
):
    """
    eq. algorithm 1 in paper https://arxiv.org/abs/2211.17192
    """

    batch, prompt_seq_len, out, device = *prompt.shape, prompt.clone(), prompt.device
    sample_num_times = max(0, seq_len - prompt_seq_len)

    num_steps = 0
    total_accepted = 0

    batch_range = torch.arange(batch, device=device, dtype=torch.long)[..., None]
    seq_lens = torch.full((batch,), prompt_seq_len, device=device, dtype=torch.long)

    print("### out ###")
    print(tokenizer.decode(out[0]))
    while (seq_lens < seq_len).any():
        prefix = out.clone()

        # predict with smaller network

        all_small_logits = []
        q_sampled_out = []

        small_output = small_net.generate(
            out,
            min_new_tokens=gamma,
            max_new_tokens=gamma,
            output_scores=True,
            return_dict_in_generate=True,
            top_p=filter_thres,
            temperature=temperature,
            do_sample=True,
        )
        small_logits = torch.cat(small_output.scores, dim=0)
        small_logits = rearrange(small_logits, "b n -> 1 b n")

        q_sampled_out = small_output.sequences[:, 1:]
        print("### q_sampled_out ###")
        print(tokenizer.decode(q_sampled_out[0]))
        # out = torch.cat((out, q_sampled_out), dim=-1)
        q_sampled_out = rearrange(q_sampled_out, "b n -> b n 1")

        # verify with larger network

        logits = net(
            prefix,
            decoder_input_ids=small_output.sequences,
        ).logits
        logits = top_k(logits, thres=filter_thres)

        # prob and prob of small model (p(x) and q(x) in algorithm 1)

        prob = safe_div(logits, temperature).softmax(dim=-1)
        small_prob = safe_div(small_logits, temperature).softmax(dim=-1)

        p, prob_next = prob[:, :-1], prob[:, -1]

        p = p.gather(-1, q_sampled_out)
        q = small_prob.gather(-1, q_sampled_out) * lenience

        p, q = [rearrange(t, "b n 1 -> b n") for t in (p, q)]

        r = random_uniform = torch.zeros_like(q).float().uniform_(0, 1)

        print("### p/q ###")
        print(p / q)

        accepted = find_first_true_index(r > (p / q))

        print("### accepted ###")
        print(accepted)

        total_accepted += accepted.float().mean()
        num_steps += 1

        num_rejected = gamma - accepted
        has_rejected = num_rejected > 0

        accepted = rearrange(accepted, "b -> b 1")
        accepted.clamp_(max=gamma - 1)
        adjusted_prob = F.relu(
            prob[batch_range, accepted] - small_prob[batch_range, accepted]
        )
        adjusted_prob = adjusted_prob / adjusted_prob.sum(dim=-1, keepdim=True)
        adjusted_prob = rearrange(adjusted_prob, "b 1 d -> b d")

        prob_next = torch.where(
            rearrange(has_rejected, "... -> ... 1"), adjusted_prob, prob_next
        )

        # do a bunch of slicing and align everything to the right, including kv caches

        max_num_rejected = num_rejected.amax()
        seq_arange = torch.arange(out.shape[-1], device=device, dtype=torch.long)
        seq_offset_indices = seq_arange + (max_num_rejected - num_rejected)[..., None]

        seq_lens -= num_rejected
        max_seq_len = seq_lens.amax()

        if batch > 1:
            out = F.pad(out, (0, max_num_rejected), value=pad_id)
            out = out[batch_range, seq_offset_indices]

            cache = tuple(
                F.pad(t, (0, 0, 0, max_num_rejected), value=pad_id) for t in cache
            )
            small_cache = tuple(
                F.pad(t, (0, 0, 0, max_num_rejected), value=pad_id) for t in small_cache
            )

            cache = tuple(rearrange(t, "b ... n d -> b n ... d") for t in cache)
            small_cache = tuple(
                rearrange(t, "b ... n d -> b n ... d") for t in small_cache
            )

            cache = tuple(t[batch_range, seq_offset_indices] for t in cache)
            small_cache = tuple(t[batch_range, seq_offset_indices] for t in small_cache)

            cache = tuple(rearrange(t, "b n ... d -> b ... n d") for t in cache)
            small_cache = tuple(
                rearrange(t, "b n ... d -> b ... n d") for t in small_cache
            )

            if out.shape[-1] > max_seq_len:
                left_index = out.shape[-1] - max_seq_len
                out = out[:, left_index:]
                cache = tuple(t[..., left_index:, :] for t in cache)
                small_cache = tuple(t[..., left_index:, :] for t in small_cache)

        # sample the additional token, one of the tricks in the paper to better bound the worst case

        next_token = torch.multinomial(prob_next, 1)

        print("### next_token ###")
        print(tokenizer.decode(next_token[0]))

        q_sampled_out = rearrange(q_sampled_out, "b n 1-> b n")
        accepted_q_sampled_out = q_sampled_out[:, : accepted]
        print("### accepted_q_sampled_out ###")
        print(tokenizer.decode(accepted_q_sampled_out[0]))

        out = torch.cat((out, accepted_q_sampled_out), dim=1)
        if not has_rejected.any():
            out = torch.cat((out, next_token), dim=-1)
        print("### out ###")
        print(tokenizer.decode(out[0]))
        seq_lens += 1

        print("###########")

    # now left align

    num_pad_left = out.shape[-1] - seq_lens
    max_pad_left = num_pad_left.amax()
    out = F.pad(out, (0, max_pad_left), value=pad_id)

    seq_len_range = torch.arange(seq_len, device=device, dtype=torch.long)
    out = out[batch_range, seq_len_range + num_pad_left[..., None]]

    return out[..., prompt_seq_len:], total_accepted / num_steps
