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
    "Seq_KD": seq_kd_fn,
    "Supervised_KD": NotImplemented,
    "Imit_KD": NotImplemented,
    "f_Distill": NotImplemented,
    # Forward KL Divergence / Jensen-Shannon Divergence
    "on_policy_GKD": NotImplemented,
}

