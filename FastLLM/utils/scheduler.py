from typing import Union

import transformers
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler, ChainedScheduler, SequentialLR, CosineAnnealingLR

from matplotlib import pyplot as plt

from transformers import AdamW, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, Adafactor


if __name__ == '__main__':
    # ============= PARAMETERs ============= #
    device = 0

    distil_method = NotImplemented

    # both should be in [0, 1]
    fixed_data_fraction = 0.1
    drafter_data_fraction = 0.1

    # https://arxiv.org/pdf/2310.08461.pdf
    training_steps = 300_000
    batch_size = 32

    learning_rate = 3e-4
    learning_rate_warmup_steps = 5_000
    learning_rate_cooldown_step_start = 150_000
    learning_rate_cooldown_step_end = 300_000

    dropout = 0.0
    n_epochs = 1

    draft_model = nn.ModuleList(
        [nn.Linear(10, 10), nn.Linear(10, 10)]
    )
    optimizer = Adafactor(
        draft_model.parameters(),
        lr=learning_rate,
        relative_step=False,
    )
    warmup_scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=learning_rate_warmup_steps
    )
    cooldown_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=learning_rate_cooldown_step_end - learning_rate_cooldown_step_start,
        eta_min=0.1 * learning_rate,
        last_epoch=-1
    )
    scheduler = SequentialLR(
        optimizer,
        [warmup_scheduler, cooldown_scheduler],
        [learning_rate_cooldown_step_start]
    )
    lrs = []
    for epoch in range(300_000):
        print(epoch, optimizer.param_groups[0]['lr'])
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    plt.plot(lrs)
    plt.show()
