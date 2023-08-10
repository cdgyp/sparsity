import torch
from torch import nn
from ..base import BaseModule, Plugin, ModuleReference

class ImplicitAdversarialSample(nn.Module):
    def __init__(self, clipping=None, shape=None) -> None:
        super().__init__()
        if shape is not None:
            self.biases = nn.Parameter(torch.zeros(shape), requires_grad=True)
        else:
            self.biases: torch.Tensor = None
        self.clipping = abs(clipping) if clipping is not None else clipping
    def forward(self, x: torch.Tensor):
        if self.biases is None:
            self.biases = nn.Parameter(torch.zeros(x.shape[1:], device=x.device))
        return x + self.biases
    def clip(self):
        if self.clipping is not None and self.biases is not None:
            with torch.no_grad():
                self.biases.clamp_(min=-self.clipping, max=self.clipping)

class WrappedImplicitAdversarialSample(BaseModule):
    def __init__(self, clipping=None, shape=None) -> None:
        super().__init__()
        self.inner = ImplicitAdversarialSample(clipping=clipping, shape=shape)
    def forward(self, x: torch.Tensor):
        self.losses.observe(self.inner.biases.abs().mean(), 'implicit_adversarial_samples')
        return self.inner(x)
    def clip(self):
        self.inner.clip()

class ImplicitAdversarialSamplePlugin(Plugin):
    def __init__(self, clipping=None) -> None:
        super().__init__()
        self.clipping = abs(clipping) if clipping is not None else clipping
    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        self.main = ModuleReference(main)
        for module in self.main.modules():
            if isinstance(module, WrappedImplicitAdversarialSample):
                module.clipping = self.clipping
    def _clip(self):
        for module in self.main.modules():
            if isinstance(module, WrappedImplicitAdversarialSample):
                module.clip()
    def prepare(self, Y: torch.Tensor, labeled: torch.Tensor, D: torch.Tensor):
        self._clip()
    def after_backward(self):
        self._clip()
