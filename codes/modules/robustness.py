import torch
from torch import nn
from ..base import BaseModule, Plugin, ModuleReference

class ImplicitAdversarialSample(BaseModule):
    def __init__(self, clipping=None) -> None:
        super().__init__()
        self.biases: torch.Tensor = None
        self.clipping = abs(clipping) if clipping is not None else clipping
    def forward(self, x: torch.Tensor):
        if self.biases is None:
            self.biases = nn.Parameter(torch.zeros(x.shape[1:], device=x.device))
        self.losses.observe(self.biases.abs().mean(), 'implicit_adversarial_samples')
        return x + self.biases
    def clip(self):
        if self.clipping is not None and self.biases is not None:
            with torch.no_grad():
                self.biases.clamp_(min=-self.clipping, max=self.clipping)

class ImplicitAdversarialSamplePlugin(Plugin):
    def __init__(self, clipping=None) -> None:
        super().__init__()
        self.clipping = abs(clipping) if clipping is not None else clipping
    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        self.main = ModuleReference(main)
        for module in self.main.modules():
            if isinstance(module, ImplicitAdversarialSample):
                module.clipping = self.clipping
    def _clip(self):
        for module in self.main.modules():
            if isinstance(module, ImplicitAdversarialSample):
                module.clip()
    def prepare(self, Y: torch.Tensor, labeled: torch.Tensor, D: torch.Tensor):
        self._clip()
    def after_backward(self):
        self._clip()
