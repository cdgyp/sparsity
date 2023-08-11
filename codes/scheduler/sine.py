import torch
from torch.optim.lr_scheduler import LambdaLR
import math

class SineAnnealingScheduler(LambdaLR):
    def full_cosine(self, epoch):
        lr = max(1 / 2 - math.cos(2*math.pi * min(epoch / self.T_max, 1)) / 2, self.min_eta_ratio)
        print(lr)
        return lr
    def __init__(self, optimizer, T_max, last_epoch, verbose=False, min_eta_ratio=1e-2):
        self.T_max = T_max
        self.min_eta_ratio=min_eta_ratio
        super().__init__(optimizer, lr_lambda=self.full_cosine , last_epoch=last_epoch, verbose=verbose)
