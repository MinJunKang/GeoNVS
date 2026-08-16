
import math
import numpy as np
from typing import Optional, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from diffusers.optimization import get_scheduler



class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """

    def __init__(
        self,
        warm_up_steps,
        lr_min,
        lr_max,
        lr_start,
        max_decay_steps,
        verbosity_interval=0,
    ):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.0
        self.verbosity_interval = verbosity_interval

    def schedule(self, n, **kwargs):
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")
        if n < self.lr_warm_up_steps:
            lr = (
                self.lr_max - self.lr_start
            ) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            t = (n - self.lr_warm_up_steps) / (
                self.lr_max_decay_steps - self.lr_warm_up_steps
            )
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + np.cos(t * np.pi)
            )
            self.last_lr = lr
            return lr

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaWarmUpCosineScheduler2:
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
    """

    def __init__(
        self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0
    ):
        assert (
            len(warm_up_steps)
            == len(f_min)
            == len(f_max)
            == len(f_start)
            == len(cycle_lengths)
        )
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.0
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(
                    f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                    f"current cycle {cycle}"
                )
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[
                cycle
            ] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            t = (n - self.lr_warm_up_steps[cycle]) / (
                self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle]
            )
            t = min(t, 1.0)
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (
                1 + np.cos(t * np.pi)
            )
            self.last_f = f
            return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)


class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):
    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(
                    f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                    f"current cycle {cycle}"
                )

        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[
                cycle
            ] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (
                self.cycle_lengths[cycle] - n
            ) / (self.cycle_lengths[cycle])
            self.last_f = f
            return f


# overwrite diffusers get_scheduler to add custom schedulers
def get_scheduler_(
    name, 
    optimizer, 
    step_rules: Optional[str] = None,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    num_cycles: Optional[int] = None,
    power: float = 1.0,
    last_epoch: int = -1
):
    """`diffusers.get_scheduler` plus the SEVA lambda schedulers.

    `num_cycles` shapes the curve rather than counting steps (cosine reads it as
    0.5 for a single half-period decay, cosine_with_restarts as the number of
    restarts), so it is left unset by default and each schedule applies its own.
    """
    if name == 'lambdalinear':
        scheduler_ = LambdaLinearScheduler(
            warm_up_steps=[num_warmup_steps],
            f_min=[1.0],
            f_max=[1.0],
            f_start=[1.e-6],
            cycle_lengths=[num_training_steps]
        )
        scheduler = LambdaLR(optimizer, lr_lambda=scheduler_.schedule, last_epoch=last_epoch)
    elif name == 'warmup_cosine2':
        scheduler_ = LambdaWarmUpCosineScheduler2(
            warm_up_steps=[num_warmup_steps],
            f_min=[1.0],
            f_max=[1.0],
            f_start=[1.e-6],
            cycle_lengths=[num_training_steps]
        )
        scheduler = LambdaLR(optimizer, lr_lambda=scheduler_.schedule, last_epoch=last_epoch)
    elif name == 'warmup_cosine':
        # unlike the cycle-based schedulers below, this one indexes nothing and
        # compares its arguments to the step count directly, so they are scalars
        scheduler_ = LambdaWarmUpCosineScheduler(
            warm_up_steps=num_warmup_steps,
            lr_min=1.0,
            lr_max=1.0,
            lr_start=1.e-6,
            max_decay_steps=int(num_training_steps * 0.85)
        )
        scheduler = LambdaLR(optimizer, lr_lambda=scheduler_.schedule, last_epoch=last_epoch)
    else:
        extra = {} if num_cycles is None else {"num_cycles": num_cycles}
        scheduler = get_scheduler(
            name=name,
            optimizer=optimizer,
            step_rules=step_rules,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=power,
            last_epoch=last_epoch,
            **extra,
        )
    return scheduler