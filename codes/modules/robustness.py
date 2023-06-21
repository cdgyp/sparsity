import torch
from torch import nn
from ..base import BaseModule

class ImplicitAdversarialSample(BaseModule):
    def __init__(self) -> None:
        super().__init__()
        self.biases: torch.Tensor = None
    def forward(self, x: torch.Tensor):
        if self.biases is None:
            self.biases = nn.Parameter(torch.zeros(x.shape[1:], device=x.device))
        self.losses.observe(self.biases.abs().mean(), 'implicit_adversarial_samples')
        return x + self.biases