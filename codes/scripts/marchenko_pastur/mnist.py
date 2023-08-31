import math
import torch
from torch import Tensor, nn
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.optim.lr_scheduler import LinearLR
from torchvision import transforms
import argparse
import einops

from torch.utils.tensorboard import SummaryWriter
from math import sqrt
from codes.base import BaseModule, Plugin

from codes.base.base import BaseModule, Plugin

from ...base import BaseModule, Model, Plugin, InputHook, IndexedHook, Training, device, new_experiment, ERM, start_tensorboard_server, DeviceSetter, WrapperDataset, ModuleReference
from ...base import expand_groundtruth
from ...modules.hooks import MlpGradientHook

class MixingInputHook(InputHook):
    def __init__(self) -> None:
        super().__init__()
        self.tokens = []
    def __call__(self, model, input, output=None):
        self.tokens.append(input[0])
    def get(self):
        res = self.tokens
        self.tokens = []
        return torch.cat(res, dim=0)



class MnistMLP(BaseModule):
    class HiddenLayer(nn.Module):
        def __init__(self, dim_hidden, index, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.network = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU()
            )
            self.index = index
        def forward(self, x):
            return self.network(x)
    class UnitNorm(nn.Module):
        def __init__(self, hidden_dim) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.scale = sqrt(self.hidden_dim)
        def forward(self, x):
            return x / self.scale
    def __init__(self, image_size=64, n_hidden_layer=4, dim_hidden=128, n_class=10):
        super().__init__()
        self.initial = nn.Linear(image_size ** 2, dim_hidden)
        self.hidden = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim_hidden),
                # MnistMLP.UnitNorm(dim_hidden),
                self.HiddenLayer(dim_hidden, index=i)
            )   
            for i in range(n_hidden_layer)
        ])
        self.classifier = nn.Linear(dim_hidden, n_class)
    def forward(self, x: torch.Tensor):
        if len(x.shape) > 2:
            x = x.flatten(1)
        z = self.initial(x)
        for layer in self.hidden:
            z = layer(z) + z
        return self.classifier(z)
    
class CovariancePlugin(Plugin):
    def __init__(self, controller_identifier, label=None, clipped=True, activated_on_training=False, log_per_step=1, centered=False, args=None):
        super().__init__()
        self.centered = centered
        self.covariances = None
        self.covariances_of_epochs = 0
        self.means = None
        self.means_of_epochs = 0
        self.n_epochs = 0
        self.label = label
        self.hooks = None
        self.clipped = clipped
        self.controller_identifier = controller_identifier
        self.activated_on_training = activated_on_training
        self.log_per_step = log_per_step
        self.decay = 1 - args.weight_decay
        self.effective_T = args.effective_T
        self.hidden_dim = args.dim_hidden
        self.batch_size = args.batch_size
        

    def suitable(self):
        return self.controller_identifier is None or self.current_executer_id == self.controller_identifier

    def hook_type(self):
        pass
    def register_hook(self, layer: nn.Module, hook):
        pass
    
    def hook_all(self, model: MnistMLP):
        res = []
        count = 0
        for layer in model.modules():
            if isinstance(layer, MnistMLP.HiddenLayer):
                if self.hooks is None or len(self.hooks) == 0:
                    h = IndexedHook(self.hook_type())
                else:
                    h = self.hooks[count]
                    count += 1
                self.register_hook(layer, h)
                res.append(h)
        return res
    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        super().register(main, plugins)
        if self.controller_identifier is None or self.current_executer_id != self.controller_identifier:
            self.hooks = self.hook_all(main)
        
    def collect(self):
        with torch.no_grad():
            if self.covariances is None:
                self.covariances = []
                self.means = []
            
            covariances_layerwise = []
            means_layerwise = []
            for index, inputs in [h.get() for h in self.hooks]:
                if inputs is None or len(inputs) == 0:
                    continue
                outer: torch.Tensor = einops.einsum(
                    inputs, inputs,
                    '... i, ... j -> ... i j'
                ).flatten(start_dim=0, end_dim=-3)
                covariance = outer.mean(dim=-3) # [i, j]
                mean = inputs.mean(dim=-2)  # [i]

                covariances_layerwise.append(covariance)    # [layer, i, j]
                means_layerwise.append(mean) # [layer, i]
            
            self.covariances.append(
                torch.stack(covariances_layerwise) # [layer, i, j]
            )   # [batch, layer, i, j]
            self.means.append(
                torch.stack(means_layerwise)    # [batch, layer, i]
            )  
    
    def forward(self, res, Y, labeled, D):
        if self.activated_on_training:
            if not self.training:
                return
        else:
            if self.training:
                return
        
        if not self.suitable():
            return
        
        self.collect()
    
    def scale(self, covariances: torch.Tensor, name: str=None):
        """scale `covariances` by a scalar to minimize anisotropy without affecting spectral distribution
        """
        assert len(covariances.shape) == 3, covariances.shape
        identity_shape = [1] * (len(covariances.shape) - 2) + list(covariances.shape[-2:])
        numerator = (covariances * torch.eye(covariances.shape[-1], device=covariances.device).reshape(identity_shape)).sum(dim=[-1, -2])
        dominator = (covariances ** 2).sum(dim=[-1, -2])
        a = numerator / dominator 
        scaled = a.unsqueeze(dim=-1).unsqueeze(dim=-1) * covariances
        if name is not None:
            for l, U in enumerate(scaled):
                alpha = torch.trace(U)
                self.losses.observe(alpha, f'{name}_alpha', l)
                hidden_dim = covariances.shape[-1]
                self.losses.observe(alpha / hidden_dim, 'ratio', f'{name}_alpha', l)
                # self.losses.observe(covariances.shape[-1] ** 0.75 / aa, 'ratio_p0.75', f'{name}_a', l)

        
        return scaled

    def anisotropy(self, covariances: torch.Tensor, name: str=None, means=None):
        hidden_dim = covariances.shape[-1]
        if self.centered:
            covariances = covariances - einops.einsum(
                    means,      means,
                    '... i ,    ... j  ->   ... i j'
                )
        covariances = self.scale(covariances, name)
        divergence = covariances - torch.eye(covariances.shape[-1], device=covariances.device)
        anisotropy = divergence.flatten(-2).norm(2, dim=-1)
        return anisotropy * sqrt(hidden_dim)
    def process(self):
        if self.covariances is None:
            return
        with torch.no_grad():
            covariances = torch.stack(self.covariances).transpose(0, 1) # [layer, batch, i, j]
            means = torch.stack(self.means).transpose(0, 1) # [layer, batch, i]
            hidden_dim = covariances.shape[-1]
            covariances_layerwise = covariances.mean(dim=1) # [layer, i, j]
            means_layerwise = means.mean(dim=1)             # [layer, i]
            anisotropies = self.anisotropy(covariances_layerwise, self.label + '_epoch', means_layerwise)

            self.covariances_of_epochs = self.covariances_of_epochs * self.decay + covariances_layerwise  # [layer, i, j]
            if self.centered:
                if self.decay != 1:
                    raise NotImplemented("centered under weight decay")
                self.means_of_epochs = self.means_of_epochs + means_layerwise
                means_epochs_layerwise = self.means_of_epochs / self.n_epochs                         # [layer, i]  
            else:
                means_epochs_layerwise = None
            covariances_epochs_layerwise = self.covariances_of_epochs   # [layer, i, j]
            global_anisotropies = self.anisotropy(covariances_epochs_layerwise, self.label + '_global', means_epochs_layerwise)
            inv_c = self.batch_size * min(self.iteration, self.effective_T) / self.hidden_dim


            if self.iteration % self.log_per_step == 0:
                self.losses.observe(inv_c, 'inv_c')
                self.losses.observe(covariances.shape[-1], self.label + '_epoch_anisotropy', 'd')
                self.losses.observe(covariances_epochs_layerwise.shape[-1], self.label + '_global_anisotropy', 'd')
                for l, (anisotropy, global_anisotropy) in enumerate(zip(anisotropies, global_anisotropies)):
                    self.losses.observe(anisotropy, self.label + '_epoch_anisotropy', l)
                    self.losses.observe(global_anisotropy, self.label + '_global_anisotropy', l)

                    # E[x^T x] = E[trace(x^T x)] = trace(E[x x^T]) = trace(covariance)
                    self.losses.observe(torch.trace(covariances_layerwise[l]), self.label + '_epoch_norm', l) 
                    self.losses.observe(torch.trace(covariances_epochs_layerwise[l]), self.label + '_global_norm', l)
            
                    self.losses.observe(covariances.shape[-1], self.label + '_epoch_norm', 'd') 
                    self.losses.observe(covariances.shape[-1], self.label + '_global_norm', 'd') 

                    # ratios
                    self.losses.observe(anisotropy / hidden_dim, 'ratio', self.label + '_epoch_anisotropy', l)
                    self.losses.observe(global_anisotropy / hidden_dim, 'ratio', self.label + '_global_anisotropy', l)
                    # self.losses.observe(hidden_dim ** 0.75 / anisotropy, 'ratio_p0.75', self.label + '_epoch_anisotropy', l)
                    # self.losses.observe(hidden_dim ** 0.75 / global_anisotropy, 'ratio_p0.75', self.label + '_global_anisotropy', l)
                    if not self.clipped:
                        self.losses.observe(hidden_dim / torch.trace(covariances_layerwise[l]), 'ratio', self.label + '_epoch_norm', l) 
                        self.losses.observe(hidden_dim / torch.trace(covariances_epochs_layerwise[l]), 'ratio', self.label + '_global_norm', l)
            
            self.covariances = None

    def after_testing(self):
        if self.covariances is None:
            return
        if not self.suitable():
            return
        if not self.activated_on_training:
            self.process()
    def after_backward(self):
        if not (self.activated_on_training and self.training):
            return
        self.process()

    def clean(self):
        pass
    
class SampleCovariancePlugin(CovariancePlugin):
    def __init__(self, controller_identifier, clipped=True, activated_on_training=False, log_per_step=1, centered=False, args=None):
        super().__init__(controller_identifier=controller_identifier, label='sample', clipped=clipped, activated_on_training=activated_on_training, log_per_step=log_per_step, centered=centered, args=args)
    def hook_type(self):
        return InputHook()
    def register_hook(self, layer: nn.Module, hook):
        layer.register_forward_hook(hook)

class MixingMlpGradientHook(MlpGradientHook):
    def __init__(self) -> None:
        super().__init__()
        self.gradients: 'list[torch.Tensor]' = []

    def __call__(self, module: nn.Module, grad_input, grad_output):
        assert (isinstance(grad_output, tuple) and len(grad_output) == 1) or isinstance(grad_output, torch.Tensor), grad_output
        grad = grad_output[0].to_dense().float() if isinstance(grad_output, tuple) else grad_output.to_dense().float()
        assert isinstance(grad, torch.Tensor), grad
        self.gradients.append(grad)

    def get(self):
        res = self.gradients
        self.gradients = []
        return torch.cat(res, dim=0)
    
class GradientCovariancePlugin(CovariancePlugin):
    def __init__(self, controller_identifier, clipped=True, activated_on_training=False, log_per_step=1, centered=False, args=None):
        assert activated_on_training
        super().__init__(controller_identifier=controller_identifier, label='gradient', clipped=clipped, activated_on_training=activated_on_training, log_per_step=log_per_step, centered=centered, args=args)
        self.grad_switch: torch.set_grad_enabled = None
        self.model: BaseModule = None
    def hook_type(self):
        return MlpGradientHook()
    def register_hook(self, layer: nn.Module, hook):
        layer.register_full_backward_hook(hook)

    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        if self.suitable():
            self.model = ModuleReference(main)
        return super().register(main, plugins)

    def prepare(self, Y, label, D):
        if self.activated_on_training:
            return
        if not self.activated_on_training and self.training:
            return
        if self.suitable() and not self.activated_on_training:
            self.grad_switch = torch.set_grad_enabled(True)
            self.grad_switch.__enter__()
        
    def forward(self, res, Y, labeled, D):
        if self.activated_on_training and not self.training:
            return
        if not self.activated_on_training and self.training:
            return
        assert self.activated_on_training
        if not self.activated_on_training:
            if self.suitable():
                loss = self.losses.sum_losses()
                loss.backward()
                self.model.zero_grad()
                self.grad_switch.__exit__(None, None, None)
                self.grad_switch = None
                return super().forward(res, Y, labeled, D)
        
    def after_backward(self):
        if not (self.activated_on_training and self.training):
            return
        if not self.suitable():
            return
        
        self.collect()
        return super().after_backward()
    
class ForcedERM(ERM):
    """compute ERM loss even in testing
    """
    def forward(self, pred, Y, labeled, D):
        Y = expand_groundtruth(pred, Y)
        labeled_pred_loss: torch.Tensor = self.pred_loss(pred[labeled].flatten(0, -2), Y[labeled].flatten())

        self.losses.set(labeled_pred_loss, 1, 'pred_loss', 'labeled')
            
        if (~labeled).sum() > 0:
            unlabeled_pred_loss: torch.Tensor = self.pred_loss(pred[~labeled].flatten(0, -2), Y[~labeled].flatten())
            self.losses.observe(unlabeled_pred_loss.mean(), 'pred_loss', 'unlabeled')

        return super().forward(pred, Y, labeled, D)
    
class ParallelModule(BaseModule):
    def __init__(self, modules: 'list[BaseModule]', stack_dim=None, cat_dim=None, auto=False, collate_fn=None, independent_training=True):
        super().__init__()
        self.mains = nn.ModuleList(modules)
        self.stack_dim = stack_dim
        self.cat_dim = cat_dim
        self.collate = collate_fn
        self.auto = auto
        assert (self.stack_dim is not None) + (self.cat_dim is not None) + (collate_fn is not None) <= 1 + (auto), 'multiple ways to collate parallel outputs'
        self.independent_training = independent_training
    def _output(self, outputs: 'list[torch.Tensor]'):
        if len(outputs) == 0:
            return None
        if self.collate is not None:
            return self.collate(outputs, self)
        if not all([isinstance(r, torch.Tensor) for r in outputs]):
            return  outputs
        if self.auto:
            if self.training and self.independent_training:
                return torch.cat(outputs, dim=0)
            else:
                return torch.stack(outputs, dim=-2)
        if self.stack_dim is not None:
            return torch.stack(outputs, dim=self.stack_dim)
        elif self.cat_dim is not None:
            return torch.cat(outputs, dim=self.cat_dim)
        else:
            return outputs
    def forward(self, X: torch.Tensor, *args, **kwargs):
        res = []
        for m, x in zip(self.mains, torch.chunk(X, len(self.mains))):
            sample = x if self.independent_training and self.training else X
            res.append(m(sample, *args, **kwargs))
        return self._output(outputs=res)
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except:
            def wrapped(*args, **kwargs):
                res = []
                for m in self.mains:
                    if hasattr(m, name):
                        res.append(getattr(m, name)(*args, **kwargs))
                return self._output(outputs=res)
            return wrapped
    

class ParallelModel(Model):
    def __init__(self, models: 'list[BaseModule]', *plugins: Plugin, writer: SummaryWriter = None, identifier=None):
        super().__init__(ParallelModule(models, auto=True), *plugins, writer=writer, identifier=identifier)

class ZipCollator:
    class _Iterator:
        def __init__(self, iter) -> None:
            self.iter = iter
        def __next__(self):
            X = next(self.iter)
            if not isinstance(X, torch.Tensor):
                if isinstance(X[0], torch.Tensor):
                    return torch.cat(X)
                res = []
                for j in range(len(X[0])):
                    res.append(torch.cat([x[j].to(device) for x in X]))
                return tuple(res)
            else:
                return X
    
    def __init__(self, *iterables) -> None:
        self.iterables = iterables
    def __iter__(self): 
        return self._Iterator(zip(*self.iterables))
    def __len__(self):
        try:
            return len(self.main)
        except:
            return None

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', type=str, default='sample_gradient_covariance')
    parser.add_argument('--n-hidden-layer', type=int, default=4)
    parser.add_argument('--dim-hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument('--centered', action='store_true')
    parser.add_argument('--n-epochs', type=int, default=100)
    parser.add_argument('--dont-clip-vectors', action='store_true')
    parser.add_argument('--n-parallel-models', type=int, required=True)
    parser.add_argument('--centralized', action='store_true')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--log-per-step', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=1e-6)
    return parser

def effective_T(weight_decay, threshold):
    if weight_decay <= 0.0:
        return float('inf')
    r = math.sqrt(1 - weight_decay)
    """
        the total weight of the tail after k steps is given by sum_{i=k}^inf r^i = r^k (1 - r^inf) / (1 - r)
        so to make tail smaller than the threshold, one needs r^k (1 - r^inf) / (1 - r) <= threshold <== r^k <= threshold <=> k >= log(threshold) / log(r)
    """
    eff_T = math.log(threshold * (1-r)) / math.log(r) - 1
    print(
        f'weight_decay: {weight_decay}',
        f'threshold: {threshold}',
        f'effective_T: {eff_T}'
    )
    return int(math.ceil(eff_T))


def main():
    parser = get_parser()
    all_args = parser.parse_known_args()
    args = all_args[0]
    unknown_args = all_args[1]
    assert len(unknown_args) == 2 and unknown_args[0] == '--device', unknown_args
    setattr(args, 'effective_T', effective_T(args.weight_decay, args.threshold))
    
    writer, _ = new_experiment(args.title + '/' + f'wd{args.weight_decay}' + '/' + str(args.dim_hidden), args=args, dir_to_runs='runs/marchenko_pastur')
    start_tensorboard_server(writer.get_logdir())

    controller_id = 'control'

    def new_covariance(plugin_type, controller_id):
        sample_covariance = plugin_type(controller_id, clipped=~args.dont_clip_vectors, activated_on_training=args.training, log_per_step=args.log_per_step, centered=args.centered, args=args)
        return sample_covariance

    sample_covariance = new_covariance(SampleCovariancePlugin, controller_id)
    gradient_covariance = new_covariance(GradientCovariancePlugin, controller_id)
    def make_single_model(identifier=None, centralized=True):
        return Model(
            MnistMLP(
                image_size=args.image_size,
                n_hidden_layer=args.n_hidden_layer,
                dim_hidden=args.dim_hidden
            ),
            sample_covariance if centralized else new_covariance(SampleCovariancePlugin, None),
            gradient_covariance if centralized else new_covariance(GradientCovariancePlugin, None),
            writer=writer,
            identifier=identifier
        )


    # model = ParallelModel(
        # [make_single_model(i, args.centralized) for i in range(args.n_parallel_models)],
        # ForcedERM() if not args.training else ERM(),
        # *([   sample_covariance,
            # gradient_covariance,
        # ] if args.centralized else []),
        # writer=writer,
        # identifier=controller_id
    # )

    model = Model(
        MnistMLP(
            image_size=args.image_size,
            n_hidden_layer=args.n_hidden_layer,
            dim_hidden=args.dim_hidden
        ),
        ERM(),
        new_covariance(SampleCovariancePlugin, None),
        new_covariance(GradientCovariancePlugin, None),
        writer=writer,
    ).to(device)

    train_transform = transforms.Compose([
        transforms.RandomCrop(args.image_size, 16),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
    ])
    test_transform = transforms.ToTensor()

    train_dataset = WrapperDataset(MNIST('data/mnist', True, transform=train_transform, download=True))
    # train_dataloader = ZipCollator(
            # *[DataLoader(train_dataset, args.batch_size, True, num_workers=8, drop_last=True, pin_memory=True) for i in range(args.n_parallel_models)]
    # )
    train_dataloader = DataLoader(train_dataset, args.batch_size, True, num_workers=8, drop_last=True, pin_memory=True)
    
    # test_dataset = MNIST('data/mnist', False, transform=test_transform, download=True)
    # test_dataloader = DataLoader(test_dataset, args.batch_size, True, num_workers=8)
    
    def linear_scheduler_maker(optimizer):
        return LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=0.01, total_iters=args.n_epochs)

    trainer = Training(
        n_epoch=args.n_epochs,
        model=DeviceSetter(model.to(device)),
        train_dataloader=train_dataloader,
        test_dataloader=None, # testing in `Training` includes training set
        lr=args.lr,
        test_batch=100000000,
        save_every_epoch=None,
        optimizer='SGD',
        weight_decay=args.weight_decay,
        writer=writer,
        scheduler_maker=linear_scheduler_maker,
        initial_test=False
    )

    trainer.run()

if __name__ == '__main__':
    main()
