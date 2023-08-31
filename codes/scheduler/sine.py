import torch
from torch.optim.lr_scheduler import LRScheduler
import math
import warnings

class SineAnnealingScheduler(LRScheduler):
    def full_cosine(self, epoch):
        r = self.T_max * self.warmup
        phase = 2 * math.pi * min(epoch, r) / self.T_max
        lr_ratio_base = 1 / 2 - math.cos(phase) / 2

        lr_ratio = lr_ratio_base * (self.gamma ** max(0, epoch - r)) * (1 / self.max_lr_ratio if self.max_lr_ratio else 1)
        if epoch < r:
            lr_ratio = max(lr_ratio, self.min_eta_ratio)
        return lr_ratio
    
    def __init__(self, optimizer, T_max, last_epoch, verbose=False, min_eta_ratio=1e-4, gamma=0.5, warmup_phase=1.0):
        self.T_max = T_max
        self.min_eta_ratio=min_eta_ratio
        self.gamma = gamma
        self.warmup = warmup_phase
        self.max_lr_ratio = None
        max_lr_ratio = 0
        for t in range(0, self.T_max + 1):
            max_lr_ratio = max(max_lr_ratio, self.full_cosine(t))
        self.max_lr_ratio = max_lr_ratio
        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)
    
    def get_lr(self) -> float:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return [base_lr * self.full_cosine(self.last_epoch)
                for base_lr in self.base_lrs]
        

if __name__ == '__main__':
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=3e-3)
    scheduler = SineAnnealingScheduler(optimizer, 10, -1, True)
    for i in range(0, 15):
        print(f"epoch {i}", scheduler.get_last_lr())
        loss = model.weight.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
