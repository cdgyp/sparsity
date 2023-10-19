from typing import Any
import torch
from torch  import nn
from torch.utils.data import DataLoader
from ..base import BaseModule, Plugin, LossManager
from tqdm.auto import tqdm
from math import ceil, sqrt

class AdversarialExample:
    def __init__(self, dataloader: DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module, n_iterations: int, test_fn=None, tqdm=True):
        self.dataloader = dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.n_iterations = n_iterations
        self.test_fn = test_fn
        self.tqdm = tqdm
    def step(self, adversarial_X: torch.nn.Parameter, X: torch.Tensor, Y: torch.Tensor):
        pass
    def generate_for_batch(self, X: torch.Tensor, Y: torch.Tensor):
        adversarial_X = torch.nn.Parameter(X.clone().detach(), requires_grad=True)
        optimizer = torch.optim.SGD([adversarial_X], lr=0)
        for t in range(self.n_iterations):
            pred = self.model(adversarial_X)
            loss = self.loss_fn(pred, Y)
            (-loss).backward()

            self.step(adversarial_X, X, Y)
            optimizer.zero_grad()

        return adversarial_X.data, pred

    def run(self, n_samples, generator=False, data=None):
        self.model.eval()
        require_grads = [bool(p.requires_grad) for p in self.model.parameters()]
        self.model.requires_grad_(False)
        
        remained_samples = n_samples
        res = []
        baseline = []
        Ys = []
        self.baseline_test = []
        self.adversarial_test = []
        device = next(iter(self.model.parameters())).device


        if data is not None:
            iterator = data
        else:
            iterator = tqdm(self.dataloader, desc="Generating Adversarial Examples", total=min(ceil(n_samples / self.dataloader.batch_size), len(self.dataloader))) if not generator and self.tqdm else self.dataloader

        for (X, Y) in iterator:
            if remained_samples <= 0:
                break
            X = X[:remained_samples].to(device)
            Y = Y[:remained_samples].to(device)
            remained_samples -= len(X)
            
            adversarial_X, pred = self.generate_for_batch(X, Y)
            if hasattr(self.model, 'after_testing_step'):
                self.model.after_testing_step()
            
            bl_test = self.test_fn(pred, Y)
            with torch.inference_mode():
                adv_test = self.test_fn(self.model(adversarial_X), Y)
            if not isinstance(bl_test, torch.Tensor): bl_test = torch.tensor(bl_test)
            if not isinstance(adv_test, torch.Tensor): adv_test = torch.tensor(adv_test)

            if generator:
                yield X, adversarial_X, Y, bl_test, adv_test
            else:
                res.append(adversarial_X)
                baseline.append(X)
                Ys.append(Y)
            if self.test_fn is not None:
                self.baseline_test.append(bl_test)
            self.adversarial_test.append(adv_test)

        for rg, p in zip(require_grads, self.model.parameters()):
            p.requires_grad = rg
            
        if not generator:
            if self.test_fn:
                return torch.cat(baseline), torch.cat(res), torch.cat(Ys), torch.cat(self.baseline_test), torch.cat(self.adversarial_test)
            else:
                return torch.cat(baseline), torch.cat(res), torch.cat(Ys)
        else:
            return
    
class FGSMExample(AdversarialExample):
    def __init__(self, dataloader: DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module, epsilon: float, test_fn=None, tqdm=True):
        super().__init__(dataloader, model, loss_fn, 1, test_fn=test_fn, tqdm=tqdm)
        self.epsilon = abs(epsilon)

    def step(self, adversarial_X: torch.nn.Parameter, X: torch.Tensor, Y: torch.Tensor):
        with torch.no_grad():
            adversarial_X.add_(self.epsilon * adversarial_X.grad.sign()).clamp_(0, 1)


class AdversarialObservation(Plugin):
    class LinearHook:
        def __init__(self, name=None, filter_threshold=None, losses:LossManager=None) -> None:
            self.inputs: torch.Tensor = None
            self.outputs: torch.Tensor = None
            self.weight: torch.Tensor = None
            self.name = name
            self.pairing_dimension = None
            self.filter_threshold = filter_threshold
            self.activated = True
            self.losses = losses
        def clean(self):
            self.inputs = None
            self.outputs = None
            self.weight = None
        def set_filter(self, pairing_dimension=None):
            self.pairing_dimension = pairing_dimension

        def hard_filter(self, eigen_val: torch.Tensor):
            sorted_eigen_val = eigen_val.sort(dim=-1, descending=True)[0]
            eigen_sum = torch.zeros(eigen_val.shape[: -1], device=eigen_val.device)
            eigen_threshold = torch.zeros(eigen_val.shape[: -1], device=eigen_val.device)
            for i in range(sorted_eigen_val.shape[-1]):
                eigen_threshold = eigen_threshold.maximum(sorted_eigen_val[..., i] * (eigen_sum > self.filter_threshold * eigen_val.sum(dim=-1)))
                eigen_sum += sorted_eigen_val[..., i]
            # print(((eigen_val >= eigen_threshold.unsqueeze(dim=-1)) * eigen_val).sum(dim=-1).mean(), eigen_val.sum(dim=-1).mean(),.float().mean())
            assert all((((eigen_val >= eigen_threshold.unsqueeze(dim=-1)) * eigen_val).sum(dim=-1) >= self.filter_threshold * eigen_val.sum(dim=-1)))
            return (eigen_val >= eigen_threshold.unsqueeze(dim=-1)).float()
        def soft_filter(self, eigen_val: torch.Tensor):
            return eigen_val.sqrt()
        def __call__(self, module: nn.Linear, args, output) -> Any:
            if not self.activated:
                return
            
            self.weight = module.weight
            assert isinstance(args, tuple) and len(args) == 1
            self.inputs = args[0]
            self.outputs = output

            if self.pairing_dimension is not None:
                if self.pairing_dimension == 'outer':
                    X: torch.Tensor = self.inputs.unflatten(0, [2, -1])[0]
                    Zs: torch.Tensor = self.outputs.unflatten(0, [2, -1]) 
                    DeltaZ = Zs[1] - Zs[0]  # since substracted, biases are canceled
                else:
                    raise NotImplemented()
                assert not X.requires_grad
                XXT = torch.matmul(X, X.transpose(-1, -2))
                eigen_val, eigen_vec = torch.linalg.eig(XXT)
                eigen_val: torch.Tensor
                eigen_vec: torch.Tensor
                eigen_val = eigen_val.real.clamp(min=0)
                eigen_vec = eigen_vec.real
                if len(eigen_val.shape) == 1:
                    return
                if self.filter_threshold is not None:
                    filtered_eigen_val = self.hard_filter(eigen_val)
                else:
                    filtered_eigen_val = self.soft_filter(eigen_val)
                
                diagonal_eigen_vec = torch.eye(eigen_vec.shape[-1], device=eigen_vec.device).unsqueeze(dim=0) * filtered_eigen_val.unsqueeze(dim=-1)
                filter_XXT = torch.matmul(torch.matmul(eigen_vec, diagonal_eigen_vec), eigen_vec.transpose(-1, -2))
                NewDeltaZ = torch.matmul(filter_XXT, DeltaZ)
                self.losses.observe((NewDeltaZ.norm(dim=[-1, -2]) / DeltaZ.norm(dim=[-1, -2])).mean(), 'norm_change_by_filtering')

                if self.filter_threshold is not None:
                    scaled_NewDeltaZ = NewDeltaZ
                    # print(NewDeltaZ.norm(dim=[-1, -2]).mean(), DeltaZ.norm(dim=[-1, -2]).mean())
                else:
                    scaled_NewDeltaZ = NewDeltaZ * (DeltaZ.norm(dim=[-1, -2]) / NewDeltaZ.norm(dim=[-1, -2])).unsqueeze(dim=-1).unsqueeze(dim=-1)
                if self.pairing_dimension == 'outer':
                    self.outputs[len(self.outputs) // 2:] = self.outputs[:len(self.outputs) // 2] + scaled_NewDeltaZ
                    return self.outputs
                else:
                    raise NotImplemented()




    def __init__(self, pairing_dimension='outer', do_filtering=False, filter_threshold=None):
        super().__init__()
        self.hooks: 'list[AdversarialObservation.LinearHook]' = []
        self.paring_dimension = pairing_dimension
        self.do_filtering = do_filtering
        self.filter_threshold = filter_threshold
        self.activated = True
    def set_filter(self, do_filtering: bool):
        self.do_filtering = do_filtering
    def register(self, main: BaseModule, plugins: 'list[Plugin]'):
        for name, m in main.named_modules():
            if isinstance(m, nn.Linear):
                h = self.LinearHook(name, filter_threshold=self.filter_threshold)
                m.register_forward_hook(h)
                self.hooks.append(h)
    def prepare(self, *args, **kwargs):
        for h in self.hooks:
            h.set_filter(self.paring_dimension if self.do_filtering else None)
            h.filter_threshold = self.filter_threshold
            h.activated = self.activated
            h.losses = self.losses
    def entropy(self, X: torch.Tensor):
        X = X.abs()
        X = X / X.sum(dim=-1, keepdim=True)
        return -(X * X.log2()).sum(dim=-1)
    def forward(self, res: torch.Tensor, Y: torch.Tensor, labeled: torch.Tensor, D: torch.Tensor):
        if not self.activated:
            return
        with torch.no_grad():
            for i, h in enumerate(self.hooks):
                if h.outputs is None:
                    assert 'out_proj' in h.name
                    continue
                if self.paring_dimension.lower() == 'outer':
                    outputs = h.outputs.unflatten(0, [2, -1])
                    inputs = h.inputs.unflatten(0, [2, -1])
                    WDeltaX = outputs[1] - outputs[0] # [b, ..., k, n]; since substracted, biases are canceled
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