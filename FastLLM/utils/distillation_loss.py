from torch import nn

# https://arxiv.org/pdf/2310.08461.pdf
fns = {
    # Forward KL Divergence
    "Seq_KD": nn.KLDivLoss(reduction="batchmean"),
    # Forward KL Divergence
    "Supervised_KD": NotImplemented,
    # Forward KL Divergence
    "Imit_KD": NotImplemented,
    # Total Variation Distance
    "f_Distill": NotImplemented,
    # Forward KL Divergence / Jensen-Shannon Divergence
    "on_policy_GKD": NotImplemented,
}