import os
import gc
import torch
from torch import nn
from codes.base import Plugin
from codes.base.base import BaseModule, Plugin, ModuleReference
from torch import einsum

from ..base import ForwardHook, BackwardHook, Hook, Plugin, replace_config
# from .vit import ViT, FeedForward
# from .relu_vit import MLPBlock
from .activations import CustomizedActivation, ActivationPosition
from .magic import MagicSynapse

def _is_instance(obj, types):
    if not isinstance(types, list):
        return isinstance(obj, types)
    if not len(types) == 1:
        return isinstance(obj, types[0])
    for t in types:
        if isinstance(obj, t):  return True
    return False

class ActivationHook(Hook):
    max_batch_size: int = 1024
    def hook_on_all(vit, mlp_types, target_module_types, type=None, msg=None):
        hook_type = type if type is not None else ActivationHook

        res = []
        count = 0
        if 'RANK' in os.environ:
            rank = f"rank {os.environ['RANK']}: "
        else:
            rank= ""
        print(f"ActivationHook: {rank}Deploying {hook_type}")
        for name1, module in vit.named_modules():
            if not _is_instance(module, mlp_types):
                continue
            for name2, submodule in module.named_modules():
                if _is_instance(submodule, target_module_types):
                    count += 1
                    h = hook_type()
                    h.hook_on(submodule)
                    h.msg = msg
                    h.index = count-1
                    if hasattr(h, 'max_batch_size'):
                        h.max_batch_size = ActivationHook.max_batch_size
                    res.append(h)
                    print(f'\t{rank}{name1}{name2}: {submodule.__class__}')
                    break
        print(f"ActivationHook: {rank}{count}s {hook_type} hooked")
        
        return res


class ActivationMapHook(ForwardHook):
    def __init__(self) -> None:
        super().__init__()
        self.activations: torch.Tensor = None
        self.pre_activations: torch.Tensor = None
        self.module = None
        self.active = False
        self.gradient_checkpointing = False
        self.max_batch_size = 1024
    def __call__(self, module: nn.Module, input, output):
        if not self.active:
            return
        if self.module is None:
            self.module = ModuleReference(module)
        assert isinstance(output, torch.Tensor), output
        if self.activations is None:
            self.activations = []
        if self.pre_activations is None:
            self.pre_activations = []
        if self.gradient_checkpointing and (len(self.activations) > 0 or len(self.pre_activations) > 0):
            return
        
        with torch.no_grad():
            if output.is_sparse or output.is_sparse_csr:
                self.activations.append(output[:self.max_batch_size].to_dense())
            else:
                self.activations.append(output[:self.max_batch_size])
            assert isinstance(input, tuple) and len(input) == 1
            self.pre_activations.append(input[0][:self.max_batch_size])
    def get(self):
        try:
            return torch.cat(self.activations, dim=0), torch.cat(self.pre_activations, dim=0)
        except Exception as e:
            print(self.activations, self.pre_activations)
            raise e
    
    def clean(self):
        self.activations = None
        self.pre_activations = None

    def hook_on_all(module: nn.Module, mlp_types, msg=None):
        return ActivationHook.hook_on_all(module, mlp_types, ActivationPosition, type=ActivationMapHook, msg=msg)


class MlpGradientHook(BackwardHook):
    def __init__(self) -> None:
        super().__init__()
        self.gradients: torch.Tensor = None
        self.active = False
        self.max_batch_size = 1024

    def __call__(self, module: nn.Module, grad_input, grad_output):
        if not self.active:
            return
        assert (isinstance(grad_output, tuple) and len(grad_output) == 1) or isinstance(grad_output, torch.Tensor), grad_output
        grad = grad_output[0].to_dense().float() if isinstance(grad_output, tuple) else grad_output.to_dense().float()
        assert isinstance(grad, torch.Tensor), grad
        if self.gradients is None:
            self.gradients = []
        self.gradients.append(grad)
    def clean(self):
        self.gradients = None
    def hook_on_all(module: nn.Module, mlp_types, msg=None):
        return ActivationHook.hook_on_all(module, mlp_types, mlp_types, type=MlpGradientHook, msg=msg)
    def get(self):
        return torch.cat([g[:self.max_batch_size] for g in self.gradients], dim=0)
    
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
            'b i d, b j d   ->  i j d',
            g,      g,
        ) / g.shape[0]
        ex_ey = einsum(
            'i d,   j d     ->  i j d',
            mean,   mean,
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

class HookingPlugin(Plugin):
    def is_hook_active(self):
        return (self.iteration % self.log_per_step == 0) or not self.training
    def prepare(self, *args, **kwargs):
        for h in self.hooks:
            h.active = self.is_hook_active()
    def clean(self):
        for h in self.hooks:
            h.clean()
    def gradient_checkpointing_enable(self):
        for h in self.hooks:
            if hasattr(h, 'gradient_checkpointing'):
                h.gradient_checkpointing = True
    def gradient_checkpointing_disable(self):
        for h in self.hooks:
            if hasattr(h, 'gradient_checkpointing'):
                h.gradient_checkpointing = False

class ActivationObservationPlugin(HookingPlugin):
    def __init__(self, mlp_types, extract_linear_layers, p=1, beta=0.9, batchwise_reported=False, log_per_step=1, pre_activation_only=False, gradient_checkpointing=False):
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

        self.mlp_types = mlp_types
        self.extract_linear_layers = extract_linear_layers

        self.main = None

        self.log_per_step = log_per_step
        self.pre_activation_only = pre_activation_only
    
    def prepare(self, *args, **kwargs):
        for h in self.activation_hooks:
            h.active = self.is_hook_active()
        for h in self.gradient_hooks:
            h.active = self.is_hook_active()

    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        self.activation_hooks = ActivationMapHook.hook_on_all(main, self.mlp_types)
        self.gradient_hooks = MlpGradientHook.hook_on_all(main, self.mlp_types)
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
                a, pre_a = h.get()
                self.losses.observe(pre_a.mean(), 'pre_activation', i)
                if not self.pre_activation_only:
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


                    if i >= len(self.sizes):
                        self.sizes.append(a.shape[1])
                    self.sizes[i] = a.shape[1]
                    assert len(self.sizes) > i
                h.clean()

        return res
    def after_minibatch_backward(self):
        if self.pre_activation_only:
            return
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
                h.get_ready()
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
                if _is_instance(m, self.mlp_types):
                    value_linear = self.extract_linear_layers(m)['values']
                    assert isinstance(value_linear, nn.Linear)


                    if value_linear.weight.grad is not None:
                        grad_norm += (value_linear.weight.grad**2).sum()
            self.losses.observe(grad_norm.sqrt(), 'grad norm wrt values')
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
                h.clean()
        for h in self.gradient_hooks:
            h.clean()


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
        
        with torch.no_grad():
            for i, p in enumerate(self.main.parameters()):
                if len(self.gradient_average) <= i:
                    self.gradient_average.append(torch.tensor([0], device=p.device))
                    self.gradient_variance.append(torch.tensor([0], device=p.device))
                if p.grad is not None:
                    self.gradient_average[i] = self.beta * self.gradient_average[i] + (1 - self.beta) * p.grad.flatten()
                    self.gradient_variance[i] = self.beta * self.gradient_variance[i] + (1 - self.beta) * (p.grad.flatten() - self.gradient_average[i]) ** 2
                
            average = torch.cat(self.gradient_average).norm(p=2)
            std = torch.cat(self.gradient_variance).sum().sqrt()
            self.losses.observe(average, 'gradient_noise', 'norm')
            self.losses.observe(std, 'gradient_noise', 'std')
            self.losses.observe(std / (average + 1e-32), 'gradient_noise', 'ratio')
    

class SimilarityPlugin(Plugin):
    def __init__(self, mlp_types, log_per_step=1):
        super().__init__()
        self.log_per_step = log_per_step
        self.main = None
        self.mlp_types = mlp_types
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
                'i d,   j d     ->  i j',
                key, key,
            ).flatten()

            vvT = einsum(
                'i d,   j d     ->  i j',
                value, value,
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
            if _is_instance(module, self.mlp_types):
                linears = []
                for submodule in module.modules():
                    if isinstance(submodule, nn.Linear):
                        linears.append(submodule)
                assert len(linears) == 2
                self.calc(*linears, mlp_index=mlp_index)
                mlp_index += 1
                        

        return args[0]
    

class ParameterChangePlugin(Plugin):
    def __init__(self, log_per_step=10):
        super().__init__()
        self.log_per_step = log_per_step
        self.main = None
        self.initial_parameters = None


    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        self.main = ModuleReference(main)
    
    def after_backward(self):
        if self.iteration % self.log_per_step != 0 :
            return

        with torch.no_grad():
            new_parameters = torch.nn.utils.convert_parameters.parameters_to_vector(self.main.parameters())
            if self.initial_parameters is None:
                self.initial_parameters = new_parameters.clone()
        
            self.losses.observe((self.initial_parameters.sign() == new_parameters.sign()).float().mean(), 'parameter_changes', 'sign')
            self.losses.observe((self.initial_parameters - new_parameters).abs().mean(), 'parameter_changes', 'mean_absolute')
            self.losses.observe((self.initial_parameters - new_parameters).abs() / (self.initial_parameters.abs() + 1e-32), 'parameter_changes', 'l1_relative')

            if self.training and self.iteration % (100 * self.log_per_step) == 0:
                self.losses.histogram(self.initial_parameters - new_parameters, 'parameter_changes', 'absolute', bins=200)
        
class ActivationDistributionPlugin(HookingPlugin):
    def __init__(self, mlp_types, log_per_step=10, eps=1e-5):
        super().__init__()
        self.log_per_step = log_per_step
        self.main = None
        self.hooks: list[ActivationMapHook] = None
        self.mlp_types = mlp_types
        self.eps = eps
        self.logged = False
    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        self.main = ModuleReference(main)
        self.hooks = ActivationMapHook.hook_on_all(main, self.mlp_types)
    def fall_within(self, values: torch.Tensor, ranges: torch.Tensor):
        res = torch.zeros_like(values, dtype=torch.bool)
        for range in ranges:
            res = res.logical_or_((range[0] <= values) & (values <= range[1]))
        return res.float().mean()
    def do_logs(self):
        if self.logged:
            return
        for i, h in enumerate(self.hooks):
            activations, pre_activations = h.get()
            habitat = h.module.get_habitat()
            # if self.training and self.iteration % (self.log_per_step * 100) == 0:
                # self.losses.histogram(activations.flatten().clamp(min=habitat['view_y'][0, 0].item(), max=habitat['view_y'][0, 1].item()), 'activation_distribution', i, bins=200)
                # self.losses.histogram(pre_activations.flatten().clamp(min=habitat['view_x'][0, 0].item(), max=habitat['view_x'][0, 1].item()), 'pre_activation_distribution', i, bins=200)
            
            if self.iteration % self.log_per_step == 0 or not self.training:
                status = 'train' if self.training else 'test'
                self.losses.observe(self.fall_within(pre_activations.flatten(), habitat['x']).mean(), f'activation_concentration_{status}', 'pre_activation', i)
                self.losses.observe(self.fall_within(activations.flatten(), habitat['y']).mean(), f'activation_concentration_{status}', 'activation', i)
        self.logged = True
        self.clean()
        for h in self.hooks:
            h.activations = [None]

    def clean(self):
        for h in self.hooks:
            h.clean()
        self.logged = False
        gc.collect()
    def after_minibatch_backward(self, *args, **kwargs):
        # When gradient checkpointing is used, we want the hooks not to record the activations in second forward propogation
        # Therefore, we keep the activations and `logged` flag until minibatch backward. If it is cleaned as early as at the end of `forward()`, then activations will be registered again

        self.clean()
    
    def after_testing_step(self):
        self.clean()
            
    def is_hook_active(self):
        return not (self.iteration % self.log_per_step != 0 and self.training)
    
    def forward(self, *args, **kwargs):
        if self.iteration % self.log_per_step != 0 and self.training:
            return
            
        with torch.no_grad():
            self.do_logs()

class ActivationGradientHook(MlpGradientHook):
    def hook_on_all(module: nn.Module, mlp_types):
        return ActivationHook.hook_on_all(module, mlp_types, ActivationPosition, type=MlpGradientHook)


class MkkTPlugin(HookingPlugin):
    def __init__(self, mlp_types, extract_linear_layers, log_per_step=10, eps=1e-5):
        super().__init__()
        self.log_per_step = log_per_step
        self.eps = eps
        self.mlp_types = mlp_types
        self.extract_linear_layers = extract_linear_layers
    def get_weight_matrix(self, mlp, which: str):
        layer = self.extract_linear_layers(mlp)[which]
        assert isinstance(layer, torch.nn.Linear) or isinstance(layer, MagicSynapse), (layer.__class__, which)
        layer = layer if isinstance(layer, torch.nn.Linear) else layer.linear
        return layer.weight
    
    def is_hook_active(self):
        return super().is_hook_active() and self.training
    
    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        self.main = ModuleReference(main)
        self.hooks = ActivationGradientHook.hook_on_all(main, self.mlp_types)

    def entrysum(self, x: torch.Tensor, M: torch.Tensor=None):
        """compute $sum_(i, j) (x x^T hadamard M)_(i, j) = trace(x x^T M) = trace(x^T M x)$
            
        :param torch.Tensor x: batched vectors
        :param torch.Tensor M: symmetric, defaults to all-1 matrix
        """
        if M is None:
            return x.sum(dim=-1) ** 2
        else:
            return einsum(
            '... i, i j, ... j  -> ...',
            x, M, x,
        )
    def diagonalsum(self, x: torch.Tensor, M: torch.Tensor=None):
        if M is None:
            return (x ** 2).sum(dim=-1)
        else:
            return ((x ** 2) * torch.diagonal(M)).sum(dim=-1)
    
    def after_minibatch_backward(self, *args, **kwargs):
        if self.iteration % self.log_per_step == 0 and self.training:
            with torch.no_grad():
                self.do_logs()
            self.clean()

class DiagonalityPlugin(MkkTPlugin):
    def single_log(self, name, p, i, sum_diagonals, sum_nondiagonals):
        self.losses.observe(sum_diagonals.mean(), f'verification_norm{p}_{name}', 'diagonals', i)
        self.losses.observe(sum_nondiagonals.mean(), f'verification_norm{p}_{name}', 'non-diagonals', i)
        self.losses.observe(
            ((sum_diagonals) / (sum_nondiagonals + self.eps)).log10().mean(),
            f'verification_norm{p}_{name}', 'ratio', i
        )


    def do_logs(self):
        layers = [m for m in self.main.modules() if _is_instance(m, self.mlp_types)]
        assert len(layers) == len(self.hooks), (len(layers), len(self.hooks))
        for i, (mlp, h) in enumerate(zip(layers, self.hooks)):
            g = h.get()
            key = self.get_weight_matrix(mlp, 'key')
            kkT = einsum(
                'i d,   j d     ->  i j',
                key, key,
            )
            
            for p in [1, 2]:
                matrix = kkT.abs() ** p
                sum_diagonals = torch.diagonal(matrix).sum()
                sum_nondiagonals = matrix.sum() - sum_diagonals
                self.single_log('kkT', p, i, sum_diagonals, sum_nondiagonals)

            for p in [1, 2]:
                for name, another_p in [('M', None), ('hadamard', kkT.abs() ** p)]:
                    x = g.abs() ** p
                    sum_diagonals = self.diagonalsum(x, another_p) # trace(abs(x x^T hadamard another) ** p) = trace((abs(x x^T) ** p hadamard abs(another) ** p)) = trace((abs(x**p) abs(x**p)^T) hadamard abs(another ** p))
                    sum_all = self.entrysum(x, another_p) # sum abs((x x^T) hadamard another)**p_(i, j) = sum abs(abs(x x^T)**p hadamard abs(another) **p)_(i, j) = sum (abs(x)**p abs(x)**p^T hadamard abs(another)**p)_(i, j)
                    sum_nondiagonals = sum_all - sum_diagonals
                    assert sum_diagonals.shape == sum_nondiagonals.shape
                    assert len(sum_diagonals.shape) in [1, 2], sum_diagonals.shape
                    self.single_log(name, p, i, sum_diagonals, sum_nondiagonals)

class SpectralIncreasePlugin(MkkTPlugin):
    def do_logs(self):
        layers = [m for m in self.main.modules() if _is_instance(m, self.mlp_types)]
        assert len(layers) == len(self.hooks), (len(layers), len(self.hooks))
        
        for i, (mlp, h) in enumerate(zip(layers, self.hooks)):
            g = h.get()
            key = self.get_weight_matrix(mlp, 'key')

            self.losses.observe(key.square().sum(), 'spectral_increase', 'kkT', i) # trace(K K^T) = || K ||_2^2
            self.losses.observe(g.square().sum(dim=-1).mean(), 'spectral_increase', 'M', i) # trace(M) = trace(g g^T) = || g ||_2^2


from typing import Union
class SpectralObservationPlugin(MkkTPlugin):
    def eigenvalues(X) -> torch.Tensor:
        """eigenvalues of X
        """
        return torch.real(torch.linalg.eigvals(X))
    
    def spectral_properties(g, K, ratio_threshold=0.1, eps=1e-32) -> 'dict[str, Union[torch.Tensor, dict[str, torch.Tensor]]]':
        kkT = einsum(
            '... i d, ... j d -> ... i j',
            K.float(), K.float(),
        )
        trace = (kkT * torch.eye(kkT.shape[-1], device=kkT.device).unsqueeze(dim=0)).sum(dim=(-1, -2))
        eigenvalues_kkT = SpectralObservationPlugin.eigenvalues(kkT).clamp(min=0)
        threshold: torch.Tensor = eigenvalues_kkT.mean(dim=-1) * ratio_threshold
        non_zero = eigenvalues_kkT > threshold.unsqueeze(dim=-1)
        filtered_eigenvalues_kkT: torch.Tensor = eigenvalues_kkT * non_zero
        max_eigenvalues_kkT = filtered_eigenvalues_kkT.max(dim=-1)[0]
        min_nonzero_eigenvalues_kkT = (filtered_eigenvalues_kkT + max_eigenvalues_kkT.unsqueeze(dim=-1) * (~non_zero)).min(dim=-1)[0]
    
        min_eigenvalues_kkT = eigenvalues_kkT.min(dim=-1)[0]
        l, r = (min_eigenvalues_kkT+1e-3).log10(), (max_eigenvalues_kkT/3).log10()
        bins = [
            10**((r - l) / 10 * i + l) for i in range(11)
        ]
        bins =  bins + [max_eigenvalues_kkT]
        h = torch.histogram(eigenvalues_kkT.to('cpu'), torch.tensor(bins))
        h_near_zero = torch.histogram(eigenvalues_kkT.to('cpu'), torch.tensor([1e-9, 1e-6, 1e-3]))

        g2 = g**2
        min_g2, max_g2 = g2.min(dim=-1)[0], g2.max(dim=-1)[0]
        return {
            'kkT': {
                'average_eigenvalues': (eigenvalues_kkT.mean(dim=-1), trace / kkT.shape[-1]),
                'zero_rate': (~non_zero).float().mean(dim=-1),
                'max_pseudo_zeros': (eigenvalues_kkT * (~non_zero)).max(dim=-1)[0],
                'average_nonzero_eigenvalues': (eigenvalues_kkT * non_zero).sum(dim=-1) / non_zero.sum(dim=-1),
                'ratio': max_eigenvalues_kkT / (min_nonzero_eigenvalues_kkT + eps),
                'max_eigenvalue':  max_eigenvalues_kkT,
                'min_nonzero_eigenvalue': min_nonzero_eigenvalues_kkT,
                'normal_histogram': {
                    'probabilities': h[0] / h[0].sum(dim=-1),
                    'bin_edges': h[1]
                },
                'near_zero_histogram': {
                    'probabilities': h_near_zero[0] / h_near_zero[0].sum(dim=-1),
                    'bin_edges': h_near_zero[1]
                },
                'eigenvalues': eigenvalues_kkT,
            },
            'g': {
                'min': min_g2,
                'max': max_g2,
            },
            'combined': max_eigenvalues_kkT * max_g2 / (min_nonzero_eigenvalues_kkT * min_g2 + eps) 
        }
    def do_logs(self):
        layers = [m for m in self.main.modules() if _is_instance(m, self.mlp_types)]
        assert len(layers) == len(self.hooks), (len(layers), len(self.hooks))
        for i, (mlp, h) in enumerate(zip(layers, self.hooks)):
            g = h.get()

            g2: torch.Tensor = g**2
            min_g2, max_g2 = g2.min(dim=-1)[0], g2.max(dim=-1)[0]
            mean_g2 = g2.mean(dim=-1)

            self.losses.observe(min_g2.mean(), 'spectral_g2', 'min', i)
            self.losses.observe(max_g2.mean(), 'spectral_g2', 'max', i)
            self.losses.observe(g2.mean(), 'spectral_g2', 'mean', i)

            self.losses.observe((max_g2 / (min_g2 + self.eps)).log10().mean(), 'spectral_g2', 'log_ratio', i)
            self.losses.observe((max_g2 / (min_g2 + self.eps)).mean(), 'spectral_g2', 'ratio', i)

            self.losses.observe((mean_g2 / (min_g2 + self.eps)).log10().mean(), 'spectral_g2', 'log_ratio_with_mean', i)
            self.losses.observe((mean_g2 / (min_g2 + self.eps)).mean(), 'spectral_g2', 'ratio_with_mean', i)

            if int(self.iteration // self.log_per_step) % 10 == 0:
                sorted_g2 = g2.sort(dim=-1)[0]
                for p in [0.1, 0.2, 0.3]:
                    lowerbound = sorted_g2[..., int(p/2*sorted_g2.shape[-2])]
                    upperbound = sorted_g2[..., -int(p/2*sorted_g2.shape[-2])]
                    self.losses.observe(lowerbound.mean(), 'spectral_g2_bounds', 'lower', 1 - p, i)
                    self.losses.observe(upperbound.mean(), 'spectral_g2_bounds', 'upper', 1 - p, i)

                    ratio = upperbound / (lowerbound + self.eps)
                    self.losses.observe(ratio.mean(), 'spectral_g2_bounds', 'ratio', 1 - p, i)
                    self.losses.observe(ratio.log10().mean(), 'spectral_g2_bounds', 'log_ratio', 1 - p, i)

from torch.autograd.functional import jacobian
class EffectiveGradientSparsity(MkkTPlugin):
    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        super().register(main, plugins)
        self.activation_hooks = ActivationMapHook.hook_on_all(main, self.mlp_types, 'EffectiveGradient')
    def prepare(self, *args, **kwargs):
        super().prepare(*args, **kwargs)
        for h in self.activation_hooks:
            h.active = self.is_hook_active()
    def clean(self):
        super().clean()
        for h in self.activation_hooks:
            h.clean()
    def do_logs(self):
        layers = [m for m in self.main.modules() if _is_instance(m, self.mlp_types)]
        assert len(layers) == len(self.hooks), (len(layers), len(self.hooks))
        for i, (hook_g, hook_a) in enumerate(zip(self.hooks, self.activation_hooks)):
            g = hook_g.get()
            _, pre_activations = hook_a.get()
            assert g.shape == pre_activations.shape, (g.shape, pre_activations.shape)
            activation_function = hook_a.module.main
            gamma = activation_function.derivative(pre_activations)
            assert g.shape == gamma.shape, (g.shape, gamma.shape)
            eta = g * gamma
            for p in [1e-9, 1e-6, 1e-3]:
                self.losses.observe((g.abs() < (p * 1e9)).float().mean(), 'concentration_g', p, i)
                self.losses.observe((eta.abs() < p).float().mean(), 'effective_concentration', p, i)
            for p in [1, 2]:
                self.losses.observe((eta.abs()**p).sum(dim=-1).mean(), 'effective_sparsity_norm', p, i)
            
            for sign, name in zip([+1, -1], ['beneficial', 'detrimental']):
                selected = (sign * eta < 0)
                count = selected.sum()
                self.losses.observe(sign * count.float(), 'activation_effect', 'count', name, i)

                sum = (eta * selected).abs_().sum()
                self.losses.observe(sign * sum, 'activation_effect', 'sum', name, i)
                self.losses.observe(sign * sum /  count, 'activation_effect', 'average', name, i)
    def gradient_checkpointing_enable(self):
        super().gradient_checkpointing_enable()
        for h in self.activation_hooks:
            if hasattr(h, 'gradient_checkpointing'):
                h.gradient_checkpointing = True
    def gradient_checkpointing_disable(self):
        super().gradient_checkpointing_disable()
        for h in self.activation_hooks:
            if hasattr(h, 'gradient_checkpointing'):
                h.gradient_checkpointing = False


class VGradientObservationPlugin(HookingPlugin):
    def __init__(self, mlp_types, log_per_step=1):
        super().__init__()
        self.mlp_types = mlp_types
        self.log_per_step = log_per_step
        self.main: BaseModule = None
        self.hooks: 'list[MlpGradientHook]' = None
    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        self.main = ModuleReference(main)
        self.hooks = MlpGradientHook.hook_on_all(main, mlp_types=self.mlp_types, msg='VGradient')

    def after_minibatch_backward(self):
        if self.iteration % self.log_per_step == 0:
            with torch.no_grad():
                self.do_logs()

    def do_logs(self):
        for i, h in enumerate(self.hooks):
            g_V = h.get()
            self.losses.observe(g_V.norm(p=2, dim=-1).mean(), 'norm_g_V', i)

    def clean(self):
        for h in self.hooks:
            h.clean()