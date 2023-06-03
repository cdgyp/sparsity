import torch
from torch import nn

class ImplicitAdversarialSample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device_tester = nn.Parameter(torch.tensor([0.0]))
        self.biases: torch.Tensor = None
    def forward(self, x: torch.Tensor):
        if self.biases is None:
            self.biases = nn.Parameter(torch.zeros(x.shape[1:], device=self.device_tester.device))
        return x + self.biases