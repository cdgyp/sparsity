import sys
import os
for version in os.listdir(os.path.abspath('./extensions/lib')):
    version_numer = version[version.find('python') + len('python'):]
    _relative_extensions_path = f'./extensions/lib/{version}/site-packages/jsrelu_ext-0.0-py{version_numer}-linux-x86_64.egg'
    sys.path.append(os.path.abspath(_relative_extensions_path))
import torch
from torch import nn
import jsrelu_ext
from ..base import BaseModule, Plugin, ModuleReference

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
    
    def derivative(self, x: torch.Tensor):
        return (x >= 0).type(x.dtype, non_blocking=True)

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

class _SparseJumpingSquaredReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.float()
        nonzeros = (x > 0).to_sparse_csr()
        larger = (x > 4).to_sparse_csr()
        middle = nonzeros * (~larger.to_dense())


        middle_x = torch.masked.masked_tensor(x * middle, middle, requires_grad=True)
        large_x = torch.masked.masked_tensor(x * larger, larger, requires_grad=True)

        ctx.save_for_backward(middle_x)
        ctx.save_for_backward(larger)
        
        middle_z = ((middle_x + 1) ** 2 - 1) / 2
        if larger.sum() > 0:
            large_z = (large_x - 4 * larger)
            large_z = large_z * 5 + 12
            return middle_z.get_data() + large_z.get_data()
        else:
            return middle_z.get_data()
    @staticmethod
    def backward(ctx, grad_output):
        middle_x, larger = ctx.saved_tensors
        derivative = (middle_x + 1).get_data() + larger * 5
        return grad_output * derivative


import jsrelu_ext
class _JumpingSquaredReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        return jsrelu_ext.forward(x)
        y = x + 1
        y.square_().add_(-1).div_(2)
        return y * (x >= 0)
    @staticmethod
    def backward(ctx, grad_output):
        x,  = ctx.saved_tensors
        return grad_output * jsrelu_ext.derivative(x)
        return grad_output * (x + 1) * (x >= 0)
    
class _NumericalControlledJumpingSquaredReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        nonzeros = (x > 0)
        larger = (x > 4)
        ctx.save_for_backward(x)
        return nonzeros * ((~larger.to_dense()) * ((x + 1)**2 - 1) / 2 +  larger * (5 * (x - 4) + 12))
    @staticmethod
    def backward(ctx, grad_output):
        x,  = ctx.saved_tensors
        nonzeros = x > 0
        large = x > 4
        middle = nonzeros * ~large

        return grad_output * (middle * (x + 1) + large * 5)

class JumpingSquaredReLU(CustomizedActivation):
    def __init__(self):
        super().__init__()
        self.inner = _JumpingSquaredReLU.apply

    def forward(self, x):
        return self.inner(x)
    def derivative(self, x):
        return (x >= 0) * (x + 1)
    
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
    def derivative(self, x):
        return self.inner.derivative(x)
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



class MixedActivation(CustomizedActivation):
    def __init__(self, *activations: CustomizedActivation) -> None:
        super().__init__()
        self.activations = nn.ModuleList(activations)
        self.ks = [1.0] + [0.0] * (len(self.activations) - 1)
    def forward(self, x):
        res = 0
        for k, activation in zip(self.ks, self.activations):
            k = max(min(k, 1), 0)
            if k != 0.0:
                res = res + k * activation(x)
        return res
    def _insert(self, res: 'dict[str, torch.Tensor]', key, interval):
        assert len(interval) == 1
        interval = interval[0]
        if key not in res:
            res[key] = interval.unsqueeze(dim=0)
        else:
            res[key][0, 0] = max(res[key][0, 0], interval[0])
            res[key][0, 1] = min(res[key][0, 1], interval[1])
        
    def _intersection(self, a: 'dict[str, torch.Tensor]', b: 'dict[str, torch.Tensor]'):
        res = {}
        for key, interval in a.items():
            self._insert(res, key, interval)
        for key, interval in b.items():
            self._insert(res, key, interval)
        return res

    def get_habitat(self) -> 'dict[str, torch.Tensor]':
        res = {}
        for k, act in zip(self.ks, self.activations):
            if k != 0.0:
                res = self._intersection(res, act.get_habitat())
        return res
    def derivative(self, x):
        res = 0
        for (k, act) in zip(self.ks, self.activations):
            k = max(min(k, 1), 0)
            if k != 0.0:
                res = res + k * act.derivative(x)
        return res

class ActivationMixingScheduler(Plugin):
    def _compute_ks(self):
        pass
    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        self.main = ModuleReference(main)
    def prepare(self, *args, **lwargs):
        ks = self._compute_ks()
        for m in self.main.modules():
            if isinstance(m, MixedActivation):
                m.ks = ks
        
class LinearActivationMixing(ActivationMixingScheduler):
    def __init__(self, max_epoch=None, max_iteration=None):
        super().__init__()
        self.max_epoch = max_epoch
        self.max_iteration = max_iteration
        assert (self.max_epoch is None and self.max_iteration is not None) or (self.max_epoch is not None and self.max_iteration is None)

    def _compute_ks(self):
        if self.max_epoch is not None:
            k = self.epoch / self.max_epoch
        else:
            k = self.iteration / self.max_iteration
        k = min(k, 1)
        k = max(k, 0)
        self.losses.observe(k, 'portion_of_jsrelu')
        return [1 - k, k]

if __name__ == "__main__":
    sjsrelu = _SparseJumpingSquaredReLU.apply
    X = torch.randn([128, 128], device="cuda", requires_grad=True)
    Y = sjsrelu(X).sum()
    Y.backward()