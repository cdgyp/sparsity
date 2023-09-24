import torch
import loralib as lora
class LoRAfy:
    def __init__(self, r=16) -> None:
        self.linears = []
        self.r = r
    def replace_linears(self, model: torch.nn.Module, path='model'):
        for name, module in model.named_children():
            p = '.'.join([path, name])
            if isinstance(module, torch.nn.Linear) and not isinstance(module, lora.LoRALayer) and 'mlp' in p.lower():
                lora_linear = lora.Linear(module.in_features, module.out_features, r=self.r, bias=(module.bias is not None))
                setattr(model, name, lora_linear)
                self.linears.append(p + ': ' + str(module.__class__))
            else:
                self.replace_linears(module, path=p)
        return model
    def __call__(self, model: torch.nn.Module):
        from .robustness import ZerothBias
        from .magic import MagicSynapse
        print("LoRA: Replacing linear layers")
        model = self.replace_linears(model)
        for m in self.linears:
            print(f'\t{m}')
        print(f"LoRA: {len(self.linears)} linear layers")

        # freeze parameters, except those in biases and zeroth-biases
        lora.mark_only_lora_as_trainable(model, bias='all')
        for m in model.modules():
            if isinstance(m, MagicSynapse) or isinstance(m, ZerothBias):
                for p in m.parameters():
                    p.requires_grad = True

        return model