import torch
from torch import nn

class MagicSynapse(nn.Module):
    def __init__(self, rho: float=0.1, linear: nn.Linear=None) -> None:
        """
            rho: approximately the ratio between noise std and entry norms in weight matrices
        """
        super().__init__()
        self.rho = rho
        self.linear = linear
    def get_sigma(self, module: torch.nn.Linear):
        weight = module.weight
        norm = weight.abs().mean()
        return norm * self.rho
    def perturb(self, module: nn.Linear, input: torch.Tensor, output: torch.Tensor):
        if self.rho == 0.0 or not self.training:
            return output
        sigma = self.get_sigma(module)
        norms = input.norm(2, dim=-1, keepdim=True)
        gaussians = torch.randn_like(output, requires_grad=False)
        return output + sigma * norms * gaussians
    def forward(self, input: torch.Tensor, output: torch.Tensor=None, module: nn.Linear=None):
        if module is None:
            module = self.linear
        if output is None:
            output = module(input)
        return self.perturb(module, input, output)
    def _default_filter(*args, **kwargs):
        return True
    def plug_in(model: nn.Module, rho: float=0.1, filter=None):
        print("MagicSynapse: plugging in")
        for name, module in model.named_children():
            module = MagicSynapse.replace(module, rho=rho)
            if isinstance(module, nn.Linear) and (filter is None or filter(name, module)):
                setattr(model, name, MagicSynapse(rho=rho, linear=module))
                print(f'\t\t {name}')
                count += 1
        print("MagicSynapse: totally {count} modules are perturbed")
        return model

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
                    print(f'\t\t {name}')
                    count += 1
            print(MagicSynapse.__class__, 'replace {count} Linear layers')
            return model, handles
except ImportError as e:
    if e.name == '..base':
        pass
    else:
        raise e