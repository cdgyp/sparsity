import torch
from torch import nn
from codes.base.base import BaseModule, Plugin
from einops import einsum

from ..base import ForwardHook, BackwardHook, Hook, Plugin, replace_config
from .vit import ViT, FeedForward


class ActivationHook(Hook):
    def hook_on_all(vit: ViT, depth, *args, **kwargs):
        hook_type = kwargs['type']  if 'type'  in kwargs and kwargs['type'] is not None else ActivationHook
        module_types = kwargs['module_types']

        res = []
        for name, module in vit.named_modules():
            if not isinstance(module, FeedForward):
                continue
            if str(depth - 1) in name:
                continue
            for submodule in module.modules():
                for type in module_types:
                    if isinstance(submodule, type):
                        h = hook_type()
                        h.hook_on(submodule)
                        res.append(h)
                        print(f'hook {hook_type} at module {submodule} as {type}')
                        break
        
        return res


class ActivationMapHook(ForwardHook):
    def __init__(self) -> None:
        super().__init__()
        self.activations: torch.Tensor = None
    def __call__(self, module: nn.Module, input, output):
        assert isinstance(output, torch.Tensor), output
        self.activations = output

    def hook_on_all(module: nn.Module, depth, *args, **kwargs):
        return ActivationHook.hook_on_all(module, depth, *args, **replace_config(kwargs, type=ActivationMapHook, module_types=[nn.ReLU, nn.GELU]))

class MlpGradientHook(BackwardHook):
    def __init__(self) -> None:
        super().__init__()
        self.gradients: torch.Tensor = None

    def __call__(self, module: nn.Module, grad_input, grad_output):
        assert isinstance(grad_output, tuple) and len(grad_output) == 1
        grad = grad_output[0]
        assert isinstance(grad, torch.Tensor), grad
        self.gradients = grad
    def hook_on_all(module: nn.Module, depth, *args, **kwargs):
        return ActivationHook.hook_on_all(module, depth, *args, **replace_config(kwargs, type=MlpGradientHook, module_types=[FeedForward]))
    
class GradientRecorder(BaseModule):
    def __init__(self, p=1, beta=0.9, label=''):
        super().__init__()
        self.label = label
        self.gradient_ratios = 1
        self.diagonal_gradients = 0
        self.layerwise_gradient_ratios = []
        self.layerwise_diagonal_gradients = []
        self.p = p
        self.beta = beta

        self.layerwise_pn_gradient_ratios = []
        self.pn_gradient_ratios = 0



    def forward(self, g: torch.Tensor, i: int):
        ggT = einsum(
            g,          g,
            '... k1 d,    ... k2 d  ->  ... k1 k2'
        )

        identity = torch.eye(ggT.shape[-1], device=ggT.device)
        if len(ggT.shape) > len(identity.shape):
            identity = identity.unsqueeze(dim=0)

        diagonal: torch.Tensor = (ggT * identity).norm(p=self.p, dim=[-1, -2])
        non_diagonal: torch.Tensor = (ggT * (1 - identity)).norm(p=self.p, dim=[-1, -2])
        assert len(diagonal.shape) in {0, 1} and len(non_diagonal.shape) in {0, 1}
        self.losses.observe(diagonal, self.label + '_gradient', 'diagonal', str(i))
        self.losses.observe(non_diagonal, self.label + '_gradient', 'non-diagonal', str(i))
        self.losses.observe((diagonal / (non_diagonal + 1e-32)).log10(), self.label + '_gradient_ratio', str(i))

        self.layerwise_gradient_ratios.append((diagonal / (non_diagonal + 1e-32)).log10().mean())
        self.layerwise_diagonal_gradients.append(diagonal.mean())

        positive = (ggT * (ggT > 0)).norm(p=self.p, dim=[-1, -2])
        negative = (ggT * (ggT < 0)).norm(p=self.p, dim=[-1, -2])
        pn_ratio = (positive / (negative + 1e-32))
        self.layerwise_pn_gradient_ratios.append(pn_ratio.log10().mean())
        self.losses.observe(pn_ratio.log10(), self.label + '_positive-negative_gradient_ratio', str(i))


    
    def summarize(self):
        gradient_ratios = torch.stack(self.layerwise_gradient_ratios).flatten()
        diagonal_gradients = torch.stack(self.layerwise_diagonal_gradients).flatten()
        pn_gradient_ratios = torch.stack(self.layerwise_pn_gradient_ratios).flatten()
        self.gradient_ratios = self.beta * self.gradient_ratios + (1 - self.beta) * gradient_ratios
        self.diagonal_gradients = self.beta * self.diagonal_gradients + (1 - self.beta) * diagonal_gradients
        self.pn_gradient_ratios = self.beta * self.pn_gradient_ratios + (1 - self.beta) * pn_gradient_ratios

        assert self.gradient_ratios.requires_grad == False
        assert self.diagonal_gradients.requires_grad == False

        self.layerwise_diagonal_gradients = []
        self.layerwise_gradient_ratios = []
        self.layerwise_pn_gradient_ratios = []
    
    def get_results(self):
        return {
            **{f'hparam/{self.label}_diagonal_gradients/mean': float(self.diagonal_gradients.mean())},
            **{f'hparam/{self.label}_diagonal_gradients/{i}': float(dg) for i, dg in enumerate(self.diagonal_gradients)},
            **{f'hparam/{self.label}_gradient_ratios/mean': float(self.gradient_ratios.mean())},
            **{f'hparam/{self.label}_gradient_ratios/{i}': r for i, r in enumerate(self.gradient_ratios)},
            **{f'hparam/{self.label}_positive-negative_gradient_ratios/mean': float(self.pn_gradient_ratios.mean())},
            **{f'hparam/{self.label}_positive-negative_gradient_ratios/{i}': r for i, r in enumerate(self.pn_gradient_ratios)},
        }
    
class CorrelationRecorder(BaseModule):
    def __init__(self, label='', beta=0.8):
        super().__init__()
        self.beta = beta
        self.layerwise_correlations = []
        self.layerwise_means = []
        self.correlations = 0
        self.means = 0
        self.label = label

    def forward(self, g: torch.Tensor, i: int):
        assert len(g.shape) == 3

        mean = g.mean(dim=0)
        self.layerwise_means.append(mean.abs().mean())

        sigma = ((g - mean.unsqueeze(dim=0))**2).mean(dim=0).sqrt()
        e_xy = einsum(
            g,      g,
            'b i d, b j d   ->  i j d'
        ) / g.shape[0]
        ex_ey = einsum(
            mean,   mean,
            'i d,   j d     ->  i j d'
        )
        cov = e_xy - ex_ey

        assert len(sigma.shape) == 2
        correlation = (cov / sigma.unsqueeze(dim=0) / sigma.unsqueeze(dim=1)).abs()

        self.layerwise_correlations.append(correlation.mean())

        self.losses.observe(mean, self.label + '_gradient_means', str(i))
        self.losses.observe(correlation, self.label + '_gradient_pearson_correlations', str(i))

    def summarize(self):
        assert len(self.layerwise_correlations[0].shape) in {0, 1}, self.layerwise_correlations[0].shape
        assert len(self.layerwise_means[0].shape) in {0, 1}, self.layerwise_means[0].shape
        means = torch.stack(self.layerwise_means).flatten()
        correlations = torch.stack(self.layerwise_correlations).flatten()

        self.means = self.beta * self.means + (1 - self.beta) * means
        self.correlations = self.beta * self.correlations  + (1 - self.beta) * correlations

        self.layerwise_correlations = []
        self.layerwise_means = []


    def get_results(self):
        return {
            **{f'hparam/{self.label}_person_corelation/mean': float(self.correlations.mean())},
            **{f'hparam/{self.label}_person_corelation/{i}': c for i, c in enumerate(self.correlations)}
        }


class ActivationObservationPlugin(Plugin):
    def __init__(self, p=1, depth=12, beta=0.9, batchwise_reported=False):
        super().__init__()
        self.activation_hooks: list[ActivationMapHook] = []
        self.gradient_hooks: list[MlpGradientHook] = []
        self.beta = beta
        self.p = p
        self.sizes = []
        self.samplewise_recorder = GradientRecorder(p=self.p, beta=self.beta, label='sample')
        self.batchwise_reported = batchwise_reported
        self.batchwise_recorder = GradientRecorder(p=self.p, beta=self.beta, label='batch')

        self.samplewise_correlation_recorder = CorrelationRecorder(label='sample', beta=self.beta)

        self.depth = depth

    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        self.activation_hooks = ActivationMapHook.hook_on_all(main, self.depth)
        self.gradient_hooks = MlpGradientHook.hook_on_all(main, self.depth)
        print(len(self.activation_hooks), len(self.gradient_hooks))
    def forward(self, res: torch.Tensor, *args, **kwargs):

        if not self.training:
            return res
        if self.iteration < 0:
            return res
        
        with torch.no_grad():
            assert self.training
            for i, h in enumerate(self.activation_hooks):
                assert h.handle is not None
                a = h.activations
                assert len(a.shape) > 1
                great_than_zero = (a > 0).float()
                self.losses.observe(great_than_zero.mean(), 'activation', 'ratio', str(i))
                dims = list(range(len(a.shape)))[1:]
                self.losses.observe((a.abs() * great_than_zero).sum(dim=dims) / great_than_zero.sum(dim=dims), 'activation', 'amplitude', str(i))

                if i >= len(self.sizes):
                    self.sizes.append(a.shape[1])
                self.sizes[i] = a.shape[1]
                assert len(self.sizes) > i

        return res
    def after_backward(self):

        if not self.training:
            return
        if self.iteration < 0:
            return

        with torch.no_grad():
            assert self.training
            for i,  h in enumerate(self.gradient_hooks):
                assert h.handle is not None
                g = h.gradients
                assert g.shape[1] == self.sizes[i]

                self.samplewise_recorder(g, i)
                if self.batchwise_reported:
                    self.batchwise_recorder(g.flatten(0, 1), i)
                self.samplewise_correlation_recorder(g, i)
            
            self.samplewise_recorder.summarize()
            if self.batchwise_reported:
                self.batchwise_recorder.summarize()
            self.samplewise_correlation_recorder.summarize()
    def get_results(self):
        if self.batchwise_reported:
            return {
                **self.samplewise_recorder.get_results(),
                **self.batchwise_recorder.get_results(),
                **self.samplewise_correlation_recorder.get_results()
            }
        else:
            return {
                **self.samplewise_recorder.get_results(),
                **self.samplewise_correlation_recorder.get_results(),
            }



    def clean(self):
        for h in self.activation_hooks:
            h.activations = None
        for h in self.gradient_hooks:
            h.gradients = None