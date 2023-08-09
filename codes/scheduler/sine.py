import torch
from torch.optim.lr_scheduler import LambdaLR
import math

class SineAnnealingScheduler(LambdaLR):
    def __init__(self, optimizer, T_max, last_epoch, verbose=False):
        self.T_max = T_max
        super().__init__(optimizer, lambda epoch: math.sin(math.pi * epoch / self.T_max), last_epoch=last_epoch, verbose=verbose)
