import torch
import torch.nn.functional as F

from .constants import (
    NUM_AA,
    HP_LABEL_SMOOTHING_WEIGHT,
    HP_LOSS_SMOOTHED_NORM,
    HP_FOCAL_GAMMA,
)


def loss_nll(S, log_probs, mask):
    criterion = torch.nn.NLLLoss(reduction="none")
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)),
        S.contiguous().view(-1),
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs, -1)
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask).clamp(min=1.0)
    return loss, loss_av, true_false


def loss_smoothed(S, log_probs, mask, weight: float = HP_LABEL_SMOOTHING_WEIGHT):
    S_onehot = F.one_hot(S, NUM_AA).float()
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)
    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / HP_LOSS_SMOOTHED_NORM
    return loss, loss_av
