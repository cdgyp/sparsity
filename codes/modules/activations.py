import torch
from torch import nn
from .relu_vit import MLPBlock

class CustomizedActivation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def get_habitat(self) -> 'dict[str, torch.Tensor]':
        pass

class CustomizedReLU(CustomizedActivation):
    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.ReLU()
    def forward(self, x):
        return self.inner(x)
    
    def get_habitat(self):
        return {
            "x": torch.tensor([[-1e32, 0]]),
            "y": torch.tensor([[-1e-6, 1e-6]])
        }

class SymmetricReLU(CustomizedActivation):
    def __init__(self, half_interval=1) -> None:
        super().__init__()
        self.half_interval = half_interval
        self.relu = nn.ReLU()
    def forward(self, x: torch.Tensor):
        return self.relu(x.abs() - self.half_interval)

class SReLU(CustomizedActivation):
    def __init__(self, half_interval=0.5) -> None:
        super().__init__()
        self.half_interval = abs(half_interval)
    def forward(self, x: torch.Tensor):
        return (x - self.half_interval) * (x - self.half_interval > 0) + (x + self.half_interval) * (x + self.half_interval < 0)
    def new(half_interval=0.5):
        return lambda: SReLU(half_interval=half_interval)
    def get_habitat(self):
        return {
            'x': torch.tensor([[-self.half_interval, self.half_interval]]),
            'y': torch.tensor([[-1e-6, 1e-6]])
        }

class Shift(CustomizedActivation):
    def __init__(self, inner: nn.Module, shift_x=0, shift_y=0) -> None:
        super().__init__()
        self.inner = inner
        self.shift_x = shift_x
        self.shift_y = shift_y
    def forward(self, x):
        return self.inner(x - self.shift_x) + self.shift_y
    def get_habitat(self):
        inner_habitat = self.inner.get_habitat()
        return {
            'x': inner_habitat['x'] + self.shift_x,
            'y': inner_habitat['y'] + self.shift_y
        }

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
    

class ActivationPosition(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
    def forward(self, x):
        return self.inner(x)
    def get_habitat(self):
        return self.inner.get_habitat()


def careful_bias_initialization(module: nn.Module, shift_x: float):
    with torch.no_grad():
        for name, m in module.named_modules():
            if isinstance(m, MLPBlock):
                keys = m[0]
                assert isinstance(keys, nn.Linear)
                keys.bias.add_(shift_x)
