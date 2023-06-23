import torch
from torch import nn

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
            "y": torch.tensor([[-1e-6, 1e-6]]),
            "view_x": torch.tensor([[-5, 5]]),
            "view_y": torch.tensor([[-5, 5]])
        }

class CustomizedGELU(CustomizedActivation):
    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.GELU()
    def forward(self, x):
        return self.inner(x)
    
    def get_habitat(self):
        return {
            "x": torch.tensor([[-1e32, -1]]),
            "y": torch.tensor([[-1e-6, 1e-6]]),
            "view_x": torch.tensor([[-5, 5]]),
            "view_y": torch.tensor([[-5, 5]])
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
        return torch.relu(x - self.half_interval) - torch.relu(-x - self.half_interval)
    def new(half_interval=0.5):
        return lambda: SReLU(half_interval=half_interval)
    def get_habitat(self):
        return {
            'x': torch.tensor([[-self.half_interval, self.half_interval]]),
            'y': torch.tensor([[-1e-6, 1e-6]])
        }

class Shift(CustomizedActivation):
    def __init__(self, inner: nn.Module, shift_x=0, shift_y=0, alpha_x=1.0, alpha_y=1.0) -> None:
        super().__init__()
        self.inner = inner
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y
    def forward(self, x):
        return  self.alpha_y * self.inner(self.alpha_x * (x - self.shift_x)) + self.shift_y
    def get_habitat(self):
        inner_habitat = self.inner.get_habitat()
        return {
            'x': inner_habitat['x'] + self.shift_x,
            'y': inner_habitat['y'] + self.shift_y,
            'view_x': inner_habitat['view_x'] + self.shift_x,
            'view_y': inner_habitat['view_y'] + self.shift_y
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
    
class SquaredReLU(CustomizedActivation):
    def forward(self, x: torch.Tensor):
        return torch.relu(x) ** 2
    def get_habitat(self):
        return {
            "x": torch.tensor([[-1e32, 0]]),
            "y": torch.tensor([[-1e-6, 1e-6]]),
            "view_x": torch.tensor([[-5, 5]]),
            "view_y": torch.tensor([[-5, 5]])
        }

class JumpingSquaredReLU(CustomizedActivation):
    def forward(self, x):
        return (x > 0) * ((x + 1)**2 - 1) / 2
    
    def get_habitat(self):
        return {
            "x": torch.tensor([[-1e32, 0]]),
            "y": torch.tensor([[-1e-6, 1e-6]]),
            "view_x": torch.tensor([[-5, 5]]),
            "view_y": torch.tensor([[-5, 5]])
        }
    
class SShaped(CustomizedActivation):
    def __init__(self, inner: CustomizedActivation, half_interval=0.5) -> None:
        super().__init__()
        self.inner = inner
        self.half_interval = abs(half_interval)
    def forward(self, x: torch.Tensor):
        return self.inner(x - self.half_interval) - self.inner(-x - self.half_interval)
    def get_habitat(self):
        inner_habitat = self.inner.get_habitat()
        inner_x_habitat = inner_habitat['x']
        assert len(inner_x_habitat) == 1, inner_x_habitat
        assert inner_x_habitat[0, 0] <= -100 and inner_x_habitat[0, 1] >= 0
        x_habitat = torch.tensor([[-(inner_x_habitat[0, 1] + self.half_interval), inner_x_habitat[0, 1] + self.half_interval]])
        inner_y_habitat = inner_habitat['y']
        y_habitat = torch.cat([inner_y_habitat, -inner_y_habitat], dim=0)

        view_y = torch.cat([inner_habitat['view_y'], -inner_habitat['view_y']])
        return {
            'x': x_habitat,
            'y': y_habitat,
            'view_x': x_habitat * 3,
            'view_y': torch.tensor([[view_y.min().item(), view_y.max().item()]])
        }

class ActivationPosition(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
    def forward(self, x):
        return self.inner(x)
    def get_habitat(self):
        return self.inner.get_habitat()

try:
    from .relu_vit import MLPBlock
    def careful_bias_initialization(module: nn.Module, shift_x: float):
        with torch.no_grad():
            for name, m in module.named_modules():
                if isinstance(m, MLPBlock):
                    keys = m[0]
                    assert isinstance(keys, nn.Linear)
                    keys.bias.add_(shift_x)
except: pass