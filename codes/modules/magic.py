from typing import Union
import torch
from torch import Tensor, nn
from torch.nn.modules.module import Module
from ..base import ModuleReference
from math import sqrt

try:
    import loralib as lora
except ImportError:
    print("Warning: Cannot import loralib")

class MagicSynapse(nn.Module):
    def __init__(self, rho: float=0.1, linear: nn.Linear=None, layer_norm: nn.LayerNorm=None, losses=None, skip_connection=False) -> None:
        """
            rho: approximately the ratio between noise std and entry norms in weight matrices
        """
        super().__init__()
        self.rho = rho
        self.linear = linear
        self.ln: nn.LayerNorm = ModuleReference(layer_norm) if layer_norm is not None else None
        self._cache: dict[str, torch.Tensor] = {}
        self._cache_gaussian: dict[str, torch.Tensor] = {}
        self.losses = losses
        self.skip_connection = skip_connection
    def get_sigma(self, module: torch.nn.Linear, x: torch.Tensor=None):
        if self.skip_connection:
            sigma = self.rho / sqrt(x.shape[-1]) / 10
        else:
            if hasattr(module, 'merge_weights'):
                # dealing with LoRAed linear layers
                assert isinstance(module, lora.Linear) or isinstance(module, lora.MergedLinear)
                old_training = bool(module.training)
                module.train(False)
                weight = module.weight + (module.weight_attaching_to.main if hasattr(module, 'weight_attaching_to') else 0)
                module.train(old_training)
            else:
                weight = module.weight
            sigma = weight.norm().div_(sqrt(weight.numel())).mul_(self.rho)
        return sigma
    def get_norms(self, input: torch.Tensor):
        """
            use pre-allocated memory when computing norms
        """
        device = str(input.device)
        if device not in self._cache or self._cache[device].shape != input.shape[:-1] or self._cache[device].dtype != input.dtype:
            self._cache[device] = torch.empty(input.shape[:-1], dtype=input.dtype, device=input.device, requires_grad=False)
        torch.norm(input, p=2, dim=-1, out=self._cache[device])
        return self._cache[device].unsqueeze(dim=-1)
    def get_standard_gaussian(self, output: torch.Tensor):
        device = str(output.device)
        if device not in self._cache_gaussian or self._cache_gaussian[device].shape != output.shape or self._cache_gaussian[device].dtype != output.dtype:
            self._cache_gaussian[device] = torch.empty_like(output, requires_grad=False)
        self._cache_gaussian[device].normal_()
        return self._cache_gaussian[device]
    
    def get_noises(self, input: torch.Tensor, output: torch.Tensor, module: torch.nn.Linear):
        with torch.no_grad():
            sigma = self.get_sigma(module, x=input)
            norms = self.get_norms(input=input)
            gaussians = self.get_standard_gaussian(output=output)
            return gaussians.mul_(sigma).mul_(norms)
        
    def perturb(self, module: nn.Linear, input: torch.Tensor, output: torch.Tensor):
        if self.rho == 0.0 or not self.training:
            return output
        return output + self.get_noises(input, output, module)
    
    def forward(self, input: torch.Tensor, output: torch.Tensor=None, module: nn.Linear=None):
        if module is None:
            module = self.linear
        if output is None:
            output = module(input)
        return self.perturb(module, input, output)
    
    def plug_in(model: nn.Module, rho: float=0.1, filter=None, path='model', skip_connections=False, skip_connection_filer=None):
        for name, module in model.named_children():
            p = '.'.join([path, name])
            if isinstance(module, nn.Linear) and (filter is None or filter(p, module)):
                setattr(model, name, MagicSynapse(rho=rho, linear=module))
                print(f'\t {p}')
            elif skip_connections and ('res' in name.lower() or 'res' in str(module.__class__.__name__).lower() or 'skip' in name.lower() or 'skip' in str(module.__class__.__name__).lower()) and (skip_connection_filer is None or skip_connection_filer(p, module)):
                setattr(model, name, MagicSynapse(rho=rho, linear=module, skip_connection=True))
                print(f'\t {p}')
            else:
                MagicSynapse.plug_in(module, rho=rho, filter=filter, path=p, skip_connections=skip_connections, skip_connection_filer=skip_connection_filer)
        return model
    
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.linear, name)

try:
    from ..base import InputHook
    class MagicSynapseHook(InputHook):
        def __init__(self, rho: float=0.1) -> None:
            super().__init__()
            self.magic_synapse = MagicSynapse(rho)
        def __call__(self, module, input, output=None):
            return self.magic_synapse(input, output, module)
        def hook_on(model: nn.Module, rho: float=0.1):
            """
                Note `torch.compile()` in current versions does **not** support hooks. So use`MagicSynapse.plug_in()` if the model will be compiled
            """
            handles: 'list[torch.utils.hooks.RemovableHandle]' = []
            count = 0
            for name, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    h = m.register_forward_hook(MagicSynapseHook(rho))
                    handles.append(h)
                    print(f'\t {name}')
                    count += 1
            print(MagicSynapse.__class__, 'replace {count} Linear layers')
            return model, handles
except ImportError as e:
    if e.name == '..base':
        pass
    else:
        raise e
