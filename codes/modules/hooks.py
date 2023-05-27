import torch
from torch import nn
from codes.base.base import BaseModule, Plugin, ModuleReference
from einops import einsum

from ..base import ForwardHook, BackwardHook, Hook, Plugin, replace_config
from .vit import ViT, FeedForward
from .relu_vit import MLPBlock, CustomizedActivation


class ActivationHook(Hook):
    def hook_on_all(vit: ViT, depth, *args, **kwargs):
        hook_type = kwargs['type']  if 'type'  in kwargs and kwargs['type'] is not None else ActivationHook
        module_types = kwargs['module_types']

        res = []
        for name, module in vit.named_modules():
            if not isinstance(module, FeedForward) and not isinstance(module, MLPBlock):
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
        self.pre_activations: torch.Tensor = None
    def __call__(self, module: nn.Module, input, output):
        assert isinstance(output, torch.Tensor), output
        self.activations = output
        assert isinstance(input, tuple) and len(input) == 1
        self.pre_activations = input[0]

    def hook_on_all(module: nn.Module, depth, *args, **kwargs):
        return ActivationHook.hook_on_all(module, depth, *args, **replace_config(kwargs, type=ActivationMapHook, module_types=[nn.ReLU, nn.GELU, CustomizedActivation, nn.LeakyReLU]))

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
        return ActivationHook.hook_on_all(module, depth, *args, **replace_config(kwargs, type=MlpGradientHook, module_types=[FeedForward, MLPBlock]))
    
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



    def forward(self, g: torch.Tensor, i: int, a: torch.Tensor):
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

        aaT = einsum(
            a,          a,
            '... k1 d, ... k2 d -> ... k1 k2'
        )
        inner_products = aaT * ggT
        diagonal_inner_products = (inner_products * identity).sum(dim=[-1, -2])
        non_diagonal_inner_products = (inner_products * (1 - identity)).sum(dim=[-1, -2])
        self.losses.observe(diagonal_inner_products, self.label + '_matrix_inner_product', 'diagonal', i)
        self.losses.observe(non_diagonal_inner_products, self.label + '_matrix_inner_product', 'non-diagonal', i)
        self.losses.observe((diagonal_inner_products / (non_diagonal_inner_products.abs() + 1e-32)).log10(), self.label + '_matrix_inner_product_ratio', i)


    
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

class IdleRecorder(BaseModule):
    def __init__(self):
        super().__init__()
    def forward(*args, **kwargs):
        pass
    def summarize(self):
        pass
    def get_results(self):
        return dict()

class ActivationObservationPlugin(Plugin):
    def __init__(self, p=1, depth=12, beta=0.9, batchwise_reported=False, log_per_step=1):
        super().__init__()
        self.activation_hooks: list[ActivationMapHook] = []
        self.gradient_hooks: list[MlpGradientHook] = []
        self.beta = beta
        self.p = p
        self.sizes = []
        self.samplewise_recorder = GradientRecorder(p=self.p, beta=self.beta, label='sample')
        self.batchwise_reported = batchwise_reported
        self.batchwise_recorder = GradientRecorder(p=self.p, beta=self.beta, label='batch')

        #self.samplewise_correlation_recorder = CorrelationRecorder(label='sample', beta=self.beta)
        self.samplewise_correlation_recorder = IdleRecorder()

        self.depth = depth

        self.main = None

        self.log_per_step = log_per_step

    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        self.activation_hooks = ActivationMapHook.hook_on_all(main, self.depth)
        self.gradient_hooks = MlpGradientHook.hook_on_all(main, self.depth)
        print(len(self.activation_hooks), len(self.gradient_hooks))
        self.main: BaseModule = ModuleReference(main)
    
    def forward(self, res: torch.Tensor, *args, **kwargs):

        if not self.training:
            return res
        if self.iteration < 0:
            return res
        if self.iteration % self.log_per_step != 0:
            return res
        
        with torch.no_grad():
            assert self.training
            for i, h in enumerate(self.activation_hooks):
                assert h.handle is not None
                a = h.activations
                assert len(a.shape) > 1
                great_than_zero = (a.abs() > 0).float()
                self.losses.observe(great_than_zero.mean(), 'activation', 'ratio', i)
                self.losses.observe((a > 0).float().mean(), 'activation', 'positive_ratio', i)
                self.losses.observe((a < 0).float().mean(), 'activation', 'negative_ratio', i)
                dims = list(range(len(a.shape)))[1:]
                self.losses.observe((a.abs() * great_than_zero).sum(dim=dims) / great_than_zero.sum(dim=dims), 'activation_amplitude', str(i))
                self.losses.observe(a.norm(p='fro', dim=[-1, -2]).mean(), 'activation_norms', i)

                self.losses.observe((a.abs() * (a > 0)).sum(), 'activation_amplitude', 'positive', i)
                self.losses.observe((a.abs() * (a < 0)).sum(), 'activation_amplitude', 'negative', i)

                self.losses.observe(h.pre_activations, 'pre_activation', i)

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
        if self.iteration % self.log_per_step != 0:
            return

        with torch.no_grad():
            assert self.training
            for i,  h in enumerate(self.gradient_hooks):
                assert h.handle is not None
                g = h.gradients
                assert g.shape[1] == self.sizes[i], (g.shape[1], self.sizes[i], i, self.sizes)

                assert self.activation_hooks[i].activations is not None
                self.samplewise_recorder(g, i, self.activation_hooks[i].activations)
                if self.batchwise_reported:
                    self.batchwise_recorder(g.flatten(0, 1), i)
                self.samplewise_correlation_recorder(g, i)
            
            self.samplewise_recorder.summarize()
            if self.batchwise_reported:
                self.batchwise_recorder.summarize()
            self.samplewise_correlation_recorder.summarize()

            grad_norm = 0
            for m in self.main.modules():
                if isinstance(m, FeedForward) or isinstance(m, MLPBlock):
                    layers = None 
                    if isinstance(m, FeedForward):
                        layers = m.net
                    elif isinstance(m, MLPBlock):   
                        layers = m
                    key_linear = layers[3]
                    assert isinstance(key_linear, nn.Linear)


                    if key_linear.weight.grad is not None:
                        grad_norm += (key_linear.weight.grad**2).sum()
            self.losses.observe(grad_norm.sqrt(), 'grad norm wrt keys')
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
        if self.gradient_hooks[0].gradients is not None:
            for h in self.activation_hooks:
                h.activations = None
                h.pre_activations = None
        for h in self.gradient_hooks:
            h.gradients = None


class GradientNoisePlugin(Plugin):
    def __init__(self, beta=0.9, log_per_step=1):
        super().__init__()
        self.beta = beta
        self.gradient_average = []
        self.gradient_variance = []
        self.main = None
        self.log_per_step = log_per_step
    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        self.main = ModuleReference(main)
    def after_backward(self):
        if self.iteration % self.log_per_step != 0:
            return

        for i, p in enumerate(self.main.parameters()):
            if len(self.gradient_average) <= i:
                self.gradient_average.append(0)
                self.gradient_variance.append(0)
            if p.grad is not None:
                self.gradient_average[i] = self.beta * self.gradient_average[i] + (1 - self.beta) * p.grad.flatten()
                self.gradient_variance[i] = self.beta * self.gradient_variance[i] + (1 - self.beta) * (p.grad.flatten() - self.gradient_average[i]) ** 2
                
        average = torch.cat(self.gradient_average).norm(p=2)
        std = torch.cat(self.gradient_variance).sum().sqrt()
        self.losses.observe(average, 'gradient_noise', 'norm')
        self.losses.observe(std, 'gradient_noise', 'std')
        self.losses.observe(std / (average + 1e-32), 'gradient_noise', 'ratio')
    

class SimilarityPlugin(Plugin):
    def __init__(self, log_per_step=1):
        super().__init__()
        self.log_per_step = log_per_step
        self.main = None
    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        self.main = ModuleReference(main)
    
    def retranspose(self, weight: torch.Tensor):
            if weight.shape[0] < weight.shape[1]:
                weight = weight.transpose(0, 1)
            return weight
    def calc(self, key_linear: nn.Linear, value_linear: nn.Linear, mlp_index: int):
        with torch.no_grad():
            key = self.retranspose(key_linear.weight)
            value = self.retranspose(value_linear.weight)

            kkT = einsum(
                key, key,
                'i d,   j d     ->  i j'
            ).flatten()

            vvT = einsum(
                value, value,
                'i d,   j d     ->  i j'
            ).flatten()
            
            pearson = torch.corrcoef(
                torch.stack([kkT, vvT], dim=0)
            )[0, 1]

            self.losses.observe(pearson, 'kkTvvT', 'pearson', mlp_index)
            self.losses.observe((kkT.sign() == vvT.sign()).float().mean(), 'kkTvvT', 'sign', mlp_index)

            product = kkT * vvT
            self.losses.observe((product.abs() * (product > 0)).mean(), 'kkTvvT', 'positive', mlp_index)
            self.losses.observe((product.abs() * (product < 0)).mean(), 'kkTvvT', 'negative', mlp_index)
            self.losses.observe(((product.abs() * (product > 0)).mean() / (product.abs() * (product < 0) + 1e-32).mean()).log10(), 'kkTvvT', 'log_ratio', mlp_index)



    def forward(self, *args):
        if self.iteration % self.log_per_step != 0:
            return
        
        main = self.main
        mlp_index = 0
        for module in main.modules():
            if isinstance(module, MLPBlock):
                linears = []
                for submodule in module.modules():
                    if isinstance(submodule, nn.Linear):
                        linears.append(submodule)
                assert len(linears) == 2
                self.calc(*linears, mlp_index=mlp_index)
                mlp_index += 1
                        

        return args[0]
    