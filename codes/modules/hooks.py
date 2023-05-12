import torch
from torch import nn
from codes.base.base import BaseModule, Plugin
from einops import einsum

from ..base import ForwardHook, BackwardHook, Hook, Plugin, replace_config
from vit_pytorch.vit import ViT


class ActivationHook(Hook):
    def hook_on_all(vit: ViT, *args, **kwargs):
        hook_type = kwargs['type']  if 'type'  in kwargs and kwargs['type'] is not None else ActivationHook

        res = []
        for name, module in vit.named_modules():
            if not '1.fn' in name:
                continue
            if isinstance(module, nn.GELU) or isinstance(module, nn.ReLU):
                h = hook_type()
                h.hook_on(module, name)
                res.append(h)
        return res


class ActivationMapHook(ForwardHook):
    def __init__(self) -> None:
        super().__init__()
        self.activations: torch.Tensor = None
    def __call__(self, module: nn.Module, input, output):
        assert isinstance(output, torch.Tensor), output
        self.activations = output

    def hook_on_all(module: nn.Module, *args, **kwargs):
        return ActivationHook.hook_on_all(module, *args, **replace_config(kwargs, type=ActivationMapHook))

class ActivationGradientHook(BackwardHook):
    def __init__(self) -> None:
        super().__init__()
        self.gradients: torch.Tensor = None

    def __call__(self, module: nn.Module, grad_input, grad_output):
        grad_input = grad_input[0]
        assert isinstance(grad_input, torch.Tensor), grad_input
        self.gradients = grad_input
    def hook_on_all(module: nn.Module, *args, **kwargs):
        return ActivationHook.hook_on_all(module, *args, **replace_config(kwargs, type=ActivationGradientHook))
    

class ActivationObservationPlugin(Plugin):
    def __init__(self):
        super().__init__()
        self.activation_hooks: list[ActivationMapHook] = []
        self.gradient_hooks: list[ActivationGradientHook] = []
    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        self.activation_hooks = ActivationMapHook.hook_on_all(main)
        self.gradient_hooks = ActivationGradientHook.hook_on_all(main)
        print(len(self.activation_hooks), len(self.gradient_hooks))
    def forward(self, res: torch.Tensor, *args, **kwargs):

        if not self.training:
            return res
        if self.iteration < 20:
            return res
        
        with torch.no_grad():
            assert self.training
            for i, h in enumerate(self.activation_hooks):
                assert h.handle is not None
                a = h.activations
                assert len(a.shape) == 3
                great_than_zero = (a > 0).float()
                self.losses.observe(great_than_zero.mean(), 'activation', 'ratio', str(i))
                self.losses.observe((a.abs() * great_than_zero).sum(dim=[-1, -2]) / great_than_zero.sum(dim=[-1, -2]), 'activation', 'amplitude', str(i))

        return res
    def after_backward(self):

        if not self.training:
            return
        if self.iteration < 20:
            return

        with torch.no_grad():
            assert self.training
            for i,  h in enumerate(self.gradient_hooks):
                assert h.handle is not None
                g = h.gradients

                ggT = einsum(
                    g,          g,
                    'b k1 d,    b k2 d  ->  b k1 k2'
                )

                identity = torch.eye(ggT.shape[-1], device=ggT.device).unsqueeze(dim=0)

                diagonal = (ggT * identity).norm(dim=[-1, -2], p=2) ** 2
                non_diagonal = (ggT * (1 - identity)).norm(dim=[-1, -2], p=2) ** 2
                assert len(diagonal.shape) == 1 and len(non_diagonal.shape) == 1
                self.losses.observe(diagonal, 'gradient', 'diagonal', str(i))
                self.losses.observe(non_diagonal, 'gradient', 'non-diagonal', str(i))


    def clean(self):
        for h in self.activation_hooks:
            h.activations = None
        for h in self.gradient_hooks:
            h.gradients = None