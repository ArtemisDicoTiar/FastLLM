import torch.nn.functional as F
from torch import nn


def seq_kd_fn(input_logits, target_logits, temperature=1):
    loss_fn = nn.KLDivLoss(reduction='none')

    return loss_fn(
        F.log_softmax(input_logits / temperature, dim=-1), 
        F.softmax(target_logits / temperature, dim=-1)
    )


# https://arxiv.org/pdf/2310.08461.pdf
fns = {
    # Forward KL Divergence
    "Seq_KD": seq_kd_fn,
    # Forward KL Divergence
    "Supervised_KD": NotImplemented,
    # Forward KL Divergence
    "Imit_KD": NotImplemented,
    # Total Variation Distance
    "f_Distill": NotImplemented,
    # Forward KL Divergence / Jensen-Shannon Divergence
    "on_policy_GKD": NotImplemented,
}

