import torch
from torch import nn
from ..base import BaseModule, Plugin, ModuleReference

class ZerothBias(nn.Module):
    def __init__(self, clipping: float=None, shape=None, layer_norm: nn.LayerNorm=None) -> None:
        """
        :param float clipping: absolute maximum magnitude of entries; but when `layer_norm` is not None, this clipping become relative to the magnitude of entries in `layer_norm`'s scaling factor. Defaults to None meaning no upperbound
        :param shape: shape of feature map of **individual** samples, defaults to None meaning automatic adaptation at the first sample
        :param nn.LayerNorm layer_norm: the `LayerNorm` proceeding the zeroth bias, defaults to None
        """
        super().__init__()
        if shape is not None:
            self.biases = nn.Parameter(torch.zeros(shape), requires_grad=True)
        else:
            self.biases: torch.Tensor = None
        self.clipping = abs(clipping) if clipping is not None else clipping
        self.layer_norm = ModuleReference(layer_norm)
    def forward(self, x: torch.Tensor):
        if self.biases is None:
            self.biases = nn.Parameter(torch.zeros(x.shape[1:], device=x.device))
        return x + self.biases[..., :x.shape[1], :]
    def clip(self):
        if self.clipping is not None and self.biases is not None:
            with torch.no_grad():
                if self.layer_norm is not None:
                    absolute_clipping = self.clipping * (self.layer_norm.weight.abs().unsqueeze(dim=-2) if hasattr(self.layer_norm, 'weight') and self.layer_norm.weight is not None else 1)
                else:
                    absolute_clipping = self.clipping
                if self.training: assert self.biases.requires_grad
                min_tensor, max_tensor = -absolute_clipping, absolute_clipping
                self.biases.copy_(torch.maximum(min_tensor, torch.minimum(self.biases, max_tensor)))

class DoublyBiased(nn.Module):
    def __init__(self, linear: nn.Linear=None, clipping: float=None, shape=None, layer_norm: nn.LayerNorm=None) -> None:
        """
        :param float clipping: absolute maximum magnitude of entries; but when `layer_norm` is not None, this clipping become relative to the magnitude of entries in `layer_norm`'s scaling factor. Defaults to None meaning no upperbound
        :param shape: shape of feature map of **individual** samples, defaults to None meaning automatic adaptation at the first sample
        :param nn.LayerNorm layer_norm: the `LayerNorm` proceeding the zeroth bias, defaults to None
        """
        super().__init__()
        self.linear = linear
        self.zeroth_bias = ZerothBias(clipping=clipping, shape=shape, layer_norm=layer_norm)
    def forward(self, x: torch.Tensor):
        y = self.linear(x)
        z = self.zeroth_bias(y)
        return z
    
try:
    class ZerothBiasPlugin(Plugin):
        def __init__(self, clipping=None, log_per_step=1) -> None:
            super().__init__()
            self.clipping = abs(clipping) if clipping is not None else clipping
            self.log_per_step = log_per_step
        def register(self, main: BaseModule, plugins: 'list[Plugin]'):
            self.main = ModuleReference(main)
            for module in self.main.modules():
                if isinstance(module, ZerothBias):
                    module.clipping = self.clipping
        def _clip(self):
            for module in self.main.modules():
                if isinstance(module, ZerothBias):
                    module.clip()
        def prepare(self,X, Y: torch.Tensor, labeled: torch.Tensor, D: torch.Tensor):
            self._clip()
        def after_backward(self):
            with torch.no_grad():
                self._clip()
                if self.iteration % self.log_per_step == 0:
                    for m in self.main.modules():
                        if isinstance(m, ZerothBias):
                            self.losses.observe(m.biases.abs().mean(), 'zeroth_biases')

    class RestrictAffinePlugin(Plugin):
        """
            restrict the scaling factors of LayerNorm layers preceeding zeroth biases
        """
        def __init__(self, log_per_step=1, finetuning=False, uplift_iterations=None):
            super().__init__()
            self.log_per_step = log_per_step
            self.finetuning = finetuning
            self.uplift_iteration = uplift_iterations
        def register(self, main: BaseModule, plugins: 'list[Plugin]'):
            self.main = ModuleReference(main)
            count = 0
            for m in self.main.modules():
                if isinstance(m, ZerothBias):
                    ln: nn.LayerNorm = m.layer_norm.main
                    if 'bias' in ln._parameters:
                        if not self.finetuning:
                            del ln._parameters['bias']
                            ln.register_parameter('bias', None)
                        else:
                            ln.bias.requires_grad = False
                    count += 1
            print(f"RestrictedAffine: {count} LayerNorm layers are restricted")
        def clamp(self, ln: nn.LayerNorm):
            ln.weight.clamp_(min=1)
        def uplift(self, ln: nn.LayerNorm):
            delta = (1 - ln.weight.abs() > 0).float().div_(self.uplift_iteration)
            sign = (ln.weight.sign() >= 0).float()

            ln.weight.add_(sign * delta)

        def after_backward(self):
            with torch.no_grad():
                for m in self.main.modules():
                    if not isinstance(m, ZerothBias):
                        continue
                    ln: nn.LayerNorm = m.layer_norm.main
                    if hasattr(ln, 'weight') and ln.weight is not None:
                        if not self.finetuning:
                            self.clamp(ln)
                        else:
                            self.uplift(ln)
                        if self.iteration % self.log_per_step == 0:
                            self.losses.observe(ln.weight.abs().mean(), 'restricted_LayerNorm_scaling_factors')
        
except ImportError as e:
    if e.name == '..base':
        pass
    else:
        raise e
