from typing import Any
import torch
from torch  import nn
from torch.utils.data import DataLoader
from ..base import BaseModule, Plugin
from tqdm.auto import tqdm
from math import ceil, sqrt

class AdversarialExample:
    def __init__(self, dataloader: DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module, n_iterations: int, test_fn=None):
        self.dataloader = dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.n_iterations = n_iterations
        self.test_fn = test_fn
    def step(self, adversarial_X: torch.nn.Parameter, X: torch.Tensor, Y: torch.Tensor):
        pass
    def generate_for_batch(self, X: torch.Tensor, Y: torch.Tensor):
        adversarial_X = torch.nn.Parameter(X.clone().detach(), requires_grad=True)
        for t in range(self.n_iterations):
            pred = self.model(adversarial_X)
            loss = self.loss_fn(pred, Y)
            (-loss).backward()

            self.step(adversarial_X, X, Y)
            adversarial_X.grad = None

        return adversarial_X.data

    def run(self, n_samples):
        self.model.eval()
        self.model.requires_grad_(False)
        
        remained_samples = n_samples
        res = []
        baseline = []
        Ys = []
        baseline_test = []
        adversarial_test = []
        device = next(iter(self.model.parameters())).device

        for (X, Y) in tqdm(self.dataloader, desc="Generating Adversarial Examples", total=min(ceil(n_samples / self.dataloader.batch_size), len(self.dataloader))):
            if remained_samples <= 0:
                break
            X = X[:remained_samples].to(device)
            Y = Y[:remained_samples].to(device)
            remained_samples -= len(X)
            
            adversarial_X = self.generate_for_batch(X, Y)
            res.append(adversarial_X)
            baseline.append(X)
            Ys.append(Y)
            if self.test_fn is not None:
                baseline_test.append(torch.tensor(self.test_fn(self.model(X), Y)))
                adversarial_test.append(torch.tensor(self.test_fn(self.model(adversarial_X), Y)))
        if self.test_fn:
            return torch.cat(baseline), torch.cat(res), torch.cat(Ys), torch.cat(baseline_test), torch.cat(adversarial_test)
        else:
            return torch.cat(baseline), torch.cat(res), torch.cat(Ys)
    
class FGSMExample(AdversarialExample):
    def __init__(self, dataloader: DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module, epsilon: float, test_fn=None):
        super().__init__(dataloader, model, loss_fn, 1, test_fn=test_fn)
        self.epsilon = abs(epsilon)

    def step(self, adversarial_X: torch.nn.Parameter, X: torch.Tensor, Y: torch.Tensor):
        with torch.no_grad():
            adversarial_X.add_(self.epsilon * adversarial_X.grad.sign()).clamp_(0, 1)


class AdversarialObservation(Plugin):
    class LinearHook:
        def __init__(self, name=None) -> None:
            self.inputs: torch.Tensor = None
            self.outputs: torch.Tensor = None
            self.weight: torch.Tensor = None
            self.name = name
        def clean(self):
            self.inputs = None
            self.outputs = None
            self.weight = None
        def __call__(self, module: nn.Linear, args, output) -> Any:
            self.weight = module.weight
            assert isinstance(args, tuple) and len(args) == 1
            self.inputs = args[0]
            self.outputs = output

    def __init__(self, pairing_dimension='outer'):
        super().__init__()
        self.hooks: 'list[AdversarialObservation.LinearHook]' = []
        self.paring_dimension = pairing_dimension
    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        for name, m in main.named_modules():
            if isinstance(m, nn.Linear):
                h = self.LinearHook(name)
                m.register_forward_hook(h)
                self.hooks.append(h)
    def entropy(self, X: torch.Tensor):
        X = X.abs()
        X = X / X.sum(dim=-1, keepdim=True)
        return -(X * X.log2()).sum(dim=-1)
    def forward(self, res: torch.Tensor, Y: torch.Tensor, labeled: torch.Tensor, D: torch.Tensor):
        with torch.no_grad():
            for i, h in enumerate(self.hooks):
                if h.outputs is None:
                    assert 'out_proj' in h.name
                    continue
                if self.paring_dimension.lower() == 'outer':
                    outputs = h.outputs.unflatten(0, [2, -1])
                    inputs = h.inputs.unflatten(0, [2, -1])
                    WDeltaX = outputs[1] - outputs[0] # [b, ..., k, n]
                    X = inputs[0]   # [b, ..., k, d]
                elif self.paring_dimension.lower() == 'inner':
                    outputs = h.outputs.unflatten(-3, [-1, 2])
                    inputs = h.inputs.unflatten(-3, [-1, 2])
                    WDeltaX = outputs[..., 1, :, :] - outputs[..., 0, :, :] # [b, ..., k, n]
                    X = inputs[..., 0, :, :]   # [b, ..., k, d]
                else:
                    raise NotImplemented()
                XXT = torch.matmul(X, X.transpose(-1, -2))
                XXTXXT = torch.matmul(XXT, XXT.transpose(-1, -2))
                if i < len(self.hooks) - 1:
                    # print(h.name, ":", torch.trace(XXTXXT.mean(dim=0)) / torch.trace(XXT.mean(dim=0))**2)
                    pass

                Xp = torch.linalg.pinv(X)

                DeltaW = torch.matmul(Xp, WDeltaX)

                self.losses.observe(DeltaW.norm(dim=[-1, -2]).mean(), 'norm', i)
                self.losses.observe(h.weight.norm(dim=[-1, -2]).mean(), 'W_norm', i)
                self.losses.observe(DeltaW.norm(dim=[-1, -2]).mean() / h.weight.norm(dim=[-1, -2]).mean(), 'norm_ratio', i)
                self.losses.observe(((DeltaW**2).max(dim=-1)[0] / (DeltaW**2).sum(dim=-1)).mean() * DeltaW.shape[-1], 'max_value_ratio', i)

                self.losses.observe((WDeltaX.norm(dim=-1) / X.norm(dim=-1)).mean()/ h.weight.norm(dim=[-1, -2]).mean(), 'tokenwise_noise_norm_ratio', i)

                continue



                LHS = torch.matmul(XXT, WDeltaX) # [b, ..., k, n]
                assert LHS.shape == WDeltaX.shape, (LHS.shape, WDeltaX.shape)
                inner_product = (LHS * WDeltaX).sum(dim=-1).clamp(min=0)
                cosine_similarity = inner_product / (LHS.norm(dim=-1) * WDeltaX.norm(dim=-1))
                self.losses.observe(cosine_similarity.mean(), 'cosine_similarity', 'layerwise', i)
                self.losses.observe(cosine_similarity.mean(), 'cosine_similarity', 'average')

            for h in self.hooks:
                h.clean()
            








