import torch
from torch.optim.lr_scheduler import LambdaLR
import math

class SineAnnealingScheduler(LambdaLR):
    def __init__(self, optimizer, T_max, last_epoch, verbose=False):
        self.T_max = T_max
        super().__init__(optimizer, lambda epoch: 1 / 2 - math.cos(2*math.pi * epoch / self.T_max) / 2, last_epoch=last_epoch, verbose=verbose)
