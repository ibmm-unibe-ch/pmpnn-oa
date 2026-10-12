import torch

from .constants import (
    HP_NOAM_FACTOR,
    HP_NOAM_WARMUP,
    HP_ADAM_BETAS,
    HP_ADAM_EPS,
)


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer  = optimizer
        self._step      = step
        self.warmup     = warmup
        self.factor     = factor
        self.model_size = model_size

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        self.optimizer.zero_grad()


def _make_optimizer(parameters, hidden_dim, step):
    return NoamOpt(
        hidden_dim, HP_NOAM_FACTOR, HP_NOAM_WARMUP,
        torch.optim.Adam(parameters, lr=0, betas=HP_ADAM_BETAS, eps=HP_ADAM_EPS),
        step,
    )
