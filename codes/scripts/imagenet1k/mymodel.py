import torch
import os
from functools import partial
from torch.utils.data import DataLoader

from ...base import Model, Wrapper

from ...modules.sparsify import Sparsify
from ...modules.relu_vit import relu_vit_b_16, ViT_B_16_Weights, MLPBlock, EncoderBlock
from ...modules.robustness import DoublyBiased

from torch.distributed import get_rank


class ImageNet1kSparsify(Sparsify):
    def __init__(self, db_mlp: bool, jsrelu: bool, magic_synapse: bool, restricted_affine: bool = None, zeroth_bias_clipping=0.1, db_mlp_shape=None, rho=0.1, log_per_step=10, mixed_scheduling={'max_epoch': None, 'max_iteration': None}, lora_r=None, model_type=Model, wrapper_type=Wrapper) -> None:
        super().__init__(db_mlp, jsrelu, magic_synapse, restricted_affine, zeroth_bias_clipping, db_mlp_shape, rho, log_per_step, mixed_scheduling, lora_r, model_type, wrapper_type)
        self.mlp_types = [MLPBlock]
    def extract_linear_layers(self, mlp: MLPBlock) -> 'dict[str, torch.nn.Linear]':
        linears = {
            'key': mlp[0],
            'value': mlp[3]
        }
        assert all([isinstance(m, torch.nn.Linear) for m in linears.values()]), {key: m.__class__ for key, m in linears.items()}
        return linears
    
    def is_MLP(self, name: str, module: torch.nn.Module):
        return isinstance(module, EncoderBlock)
    def wrap_MLP(self, path, name: str, model: torch.nn.Module, module: EncoderBlock, clipping, shape):
        db_mlp = DoublyBiased(module.mlp, clipping=clipping, shape=shape, layer_norm=module.ln_2)
        setattr(module, 'mlp', db_mlp)
        return model
    def magic_synapse_filter(self, name: str, module: torch.nn.Module):
        return '.mlp.' in name

def get_imagenet1k_model(model_type: str, dataloader: DataLoader, args=None, epoch_size=0, start_epoch=1, max_epoch_mixing_activations=10):
    if model_type not in ['vanilla', 'sparsified']:
        raise NotImplemented(model_type)
    sparsified = (model_type == 'sparsified')
    
    if sparsified:
        args.restricted_affine=True

    vit = relu_vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if not args.from_scratch and not args.finetune else None, progress=True, wide=args.wide, **{
        'num_classes': 1000,
        'rezero': False,
        'norm_layer': partial(torch.nn.LayerNorm, eps=1e-6),
    })

    
    model, _, output_dir = ImageNet1kSparsify(
        db_mlp=sparsified,
        jsrelu=sparsified,
        magic_synapse=args.magic_synapse,
        restricted_affine=args.restricted_affine,
        zeroth_bias_clipping=args.zeroth_bias_clipping,
        rho=args.magic_synapse_rho,
        log_per_step=args.log_per_step,
        mixed_scheduling={'max_iteration': args.activation_mixing_epoch * len(dataloader)},
        lora_r=args.lora_r
    )(
            'imagenet1k', 
            args.title + '/' + model_type,
            vit,
            resume=args.resume,
            finetuning=args.finetune,
            epoch_size=epoch_size,
            start_epoch=start_epoch,
            dataloader=dataloader,
            device=args.gpu if args.distributed else "cuda",
        )
    
    args.output_dir = output_dir

    return model