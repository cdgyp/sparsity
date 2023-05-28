import torch
from torch import nn

class CustomizedActivation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class SymmetricReLU(CustomizedActivation):
    def __init__(self, half_interval=1) -> None:
        super().__init__()
        self.half_interval = half_interval
        self.relu = nn.ReLU()
    def forward(self, x: torch.Tensor):
        return self.relu(x.abs() - self.half_interval)

class SReLU(CustomizedActivation):
    def __init__(self, interval=1) -> None:
        super().__init__()
        self.interval = interval
    def forward(self, x: torch.Tensor):
        return x * (x > 0) + (x + self.interval) * (x + self.interval < 0)
    def new(interval=1):
        return lambda: SReLU(interval=interval)

class WeirdLeakyReLU(CustomizedActivation):
    def __init__(self, alpha_positive=1, alpha_negative=0.001) -> None:
        super().__init__()
        self.alpha_positive = alpha_positive
        self.alpha_negative = alpha_negative
    def forward(self, x: torch.Tensor):
        return self.alpha_positive * x * (x > 0) + self.alpha_negative * x * (x <= 0)
    def get_constructor(alpha_positive=1, alpha_negative=0.001):
        return lambda: WeirdLeakyReLU(alpha_positive=alpha_positive, alpha_negative=alpha_negative)

class DenseReLU(CustomizedActivation):
    def __init__(self, half_interval=1) -> None:
        super().__init__()
        self.half_interval = half_interval
        self.relu = nn.ReLU()
    def forward(self, x: torch.Tensor):
        return self.relu(-x.abs() + self.half_interval)
    def get_constructor(half_interval=1):
        return lambda: DenseReLU(half_interval=half_interval)