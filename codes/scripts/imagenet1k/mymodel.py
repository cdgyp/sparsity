import torch
import os
from functools import partial
from torch.utils.data import DataLoader

from ...base import Model, Wrapper

from ...modules.sparsify import Sparsify
from ...modules.relu_vit import relu_vit_b_16, ViT_B_16_Weights, MLPBlock, EncoderBlock, ResidualConnection
from ...modules.robustness import DoublyBiased, RestrictAffinePlugin
from ...modules.hooks import GradientDensityPlugin, CoefficientPlugin
from ...modules.activations import ActivationPosition

from torch.distributed import get_rank


class ImageNet1kSparsify(Sparsify):
    def __init__(self, db_mlp: bool, jsrelu: bool, magic_synapse: bool, restricted_affine: bool = None, zeroth_bias_clipping=0.1, db_mlp_shape=None, rho=0.1, log_per_step=10, scheduling={'activation_mixing_iteration': None, 'layernorm_uplifting_iteration': None}, lora_r=None, model_type=Model, wrapper_type=Wrapper, magic_residual=False, manual_plugins: 'list[Plugin]'=None,) -> None:
        super().__init__(db_mlp, jsrelu, magic_synapse, restricted_affine, zeroth_bias_clipping, db_mlp_shape, rho, log_per_step, scheduling, lora_r, model_type, wrapper_type, magic_residual, manual_plugins=manual_plugins)
        self.mlp_types = [MLPBlock]
    def extract_linear_layers(self, mlp: MLPBlock) -> 'dict[str, torch.nn.Linear]':
        linears = {
            'key': mlp[0],
            'value': mlp[3]
        }
        try:
            assert all([isinstance(m, torch.nn.Linear) for m in linears.values()]), {key: m.__class__ for key, m in linears.items()}
        except AssertionError:
            from ...modules.magic import MagicSynapse
            for k, v in linears.items():
                if isinstance(v, MagicSynapse):
                    assert not hasattr(v.linear, 'weight_attaching_to')
                    linears[k] = v.linear
        assert all([isinstance(m, torch.nn.Linear) for m in linears.values()]), {key: m.__class__ for key, m in linears.items()}

        return linears
    
    def is_MLP(self, name: str, module: torch.nn.Module):
        return isinstance(module, EncoderBlock)
    def wrap_MLP(self, path, name: str, model: torch.nn.Module, module: EncoderBlock, clipping, shape):
        db_mlp = DoublyBiased(module.mlp, clipping=clipping, shape=shape, layer_norm=module.ln_2)
        setattr(module, 'mlp', db_mlp)
        return model
    def magic_synapse_filter(self, name: str, module: torch.nn.Module):
        if '.mlp.' in name:
            return True
        try:
            import loralib as lora
            return isinstance(module, lora.Linear) or isinstance(module, lora.MergedLinear)
        except: pass
        return False
    def skip_connection_filter(self, path: str, module: torch.nn.Module):
        return isinstance(module, ResidualConnection)
        

def get_imagenet1k_model(model_type: str, dataloader: DataLoader, args=None, epoch_size=0, start_epoch=1, max_epoch_mixing_activations=10):
    if model_type not in ['vanilla', 'sparsified']:
        raise NotImplemented(model_type)
    sparsified = (model_type == 'sparsified')
    
    if sparsified:
        args.restricted_affine=True

    vit = relu_vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if not args.from_scratch and not args.finetune else None, progress=True, wide=args.wide, **{
        'num_classes': 10 if 'cifar10' in args.data_path else 1000,
        'rezero': False,
        'norm_layer': partial(torch.nn.LayerNorm, eps=1e-6),
    })

    if args.post_training_only:
        os.makedirs('runs/imagenet1k/' + args.title + '/' + model_type, exist_ok=True)

    if args.post_training_only:
        if args.gradient_density_only:
            manual_plugins = [
                RestrictAffinePlugin(
                    log_per_step=args.log_per_step, 
                    finetuning=False, 
                    uplift_iterations=10000
                ) if sparsified else None,
                GradientDensityPlugin([MLPBlock]),
            ] 
        if args.augmented_flatness_only:
            manual_plugins = [
                RestrictAffinePlugin(
                    log_per_step=args.log_per_step, 
                    finetuning=False, 
                    uplift_iterations=10000
                ) if sparsified else None,
                CoefficientPlugin(
                    lambda name,m: isinstance(m, MLPBlock),
                    partial(ImageNet1kSparsify.extract_linear_layers, None),
                    lambda path, name, m: isinstance(m, ActivationPosition)
                ),
            ] 
    else:
        manual_plugins = None


    model, _, output_dir = ImageNet1kSparsify(
        db_mlp=sparsified,
        jsrelu=sparsified,
        magic_synapse=args.magic_synapse,
        restricted_affine=args.restricted_affine,
        zeroth_bias_clipping=args.zeroth_bias_clipping,
        rho=args.magic_synapse_rho,
        log_per_step=args.log_per_step,
        scheduling={'activation_mixing_iteration': args.activation_mixing_epoch * len(dataloader), 'layernorm_uplifting_iteration': args.layernorm_uplifting_epoch * len(dataloader)},
        lora_r=args.lora_r,
        magic_residual=args.magic_residual,
        manual_plugins=manual_plugins
    )(
            'imagenet1k', 
            args.title + '/' + model_type,
            vit,
            resume=args.resume if not args.post_training_only else 'runs/imagenet1k/' + args.title + '/' + model_type,
            finetuning=args.finetune,
            epoch_size=epoch_size,
            start_epoch=start_epoch,
            dataloader=dataloader,
            device=args.gpu if args.distributed else "cuda",
            mixed_activation=args.mixed_activation
        )
    
    args.output_dir = output_dir

    return model
