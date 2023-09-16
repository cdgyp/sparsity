import os
import torch
from ..base import new_experiment, Model, Wrapper, ERM, start_tensorboard_server, replace_config, SpecialReplacement, LossManager
from .hooks import ActivationObservationPlugin, GradientNoisePlugin, SimilarityPlugin, ParameterChangePlugin, ActivationDistributionPlugin, DiagonalityPlugin, SpectralObservationPlugin, EffectiveGradientSparsity, SpectralIncreasePlugin, VGradientObservationPlugin
from .activations import JumpingSquaredReLU, CustomizedReLU, ActivationPosition, CustomizedGELU
from .robustness import RestrictAffinePlugin, DoublyBiased, ZerothBiasPlugin
from .magic import MagicSynapse
from torch.distributed import get_rank

class Sparsify:
    def __init__(self, model_type=Model, wrapper_type=Wrapper) -> None:
        self.activations = []
        self.mlps = []
        self.mlp_types = []
        self.model_type = model_type
        self.wrapper_type = wrapper_type
    
    def extract_linear_layers(self, mlp) -> 'dict[str, torch.nn.Linear]':
        pass

    def replace_activations(self, model: torch.nn.Module, jsrelu, path='model'):
        for name, module in model.named_children():
            p = '.'.join([path, name])
            if isinstance(module, torch.nn.ReLU):
                if jsrelu:
                    setattr(model, name, ActivationPosition(JumpingSquaredReLU()))
                else:
                    setattr(model, name, ActivationPosition(CustomizedReLU()))
                self.activations.append(p + ': ' + str(module.__class__))
            else:
                self.replace_activations(module, jsrelu=jsrelu, path=p)
        return model
    
    def is_MLP(self, name: str, module: torch.nn.Module):
        pass
    def wrap_MLP(self, path: str, name: str, model: torch.nn.Module, module: torch.nn.Module, clipping, shape):
        pass

    def replace_MLPs(self, model: torch.nn.Module, clipping=None, shape=None, path='model'):
        for name, module in model.named_children():
            p = '.'.join([path, name])
            if self.is_MLP(p, module):
                self.wrap_MLP(path, name, model, module, clipping=clipping, shape=shape)
                self.mlps.append(p + ': ' + str(module.__class__))
            else:
                self.replace_MLPs(module, clipping, shape, p)
        return model

    def magic_synapse_filter(self, name: str, module: torch.nn.Module):
        return True

    
    def __call__(self, 
        name: str,
        title: str,
        model: torch.nn.Module,
        db_mlp: bool,
        jsrelu: bool,
        magic_synapse: bool,
        restricted_affine: bool=None,
        zeroth_bias_clipping=0.1,
        db_mlp_shape=None,
        rho=0.1,
        log_per_step=10,
        device='cuda',
        epoch_size=0, start_epoch=0, steps=None,
        resume=None,
        dataloader=None,
        physical_batch_size=None,
    ):
        if restricted_affine is None:
            restricted_affine = db_mlp

        if resume is not None and len(resume) > 0:
            if 'save' in resume:
                dir = resume[:resume.find('save')]
            else:
                dir = resume
        else:
            dir = name + '/' + title + '/'

        writer, _ = new_experiment(dir, None, dir_to_runs='runs', resume=resume is not None and len(resume) > 0, device=device)

        if db_mlp:
            print("Sparsify: replacing singly biased MLPs")
            model = self.replace_MLPs(model, zeroth_bias_clipping, db_mlp_shape)
            for m in self.mlps:
                print(f'\t{m}')
            print(f'Sparsify: {len(self.mlps)} MLP blocks replaced')

        print("Sparsify: replacing activation functions")
        model = self.replace_activations(model, jsrelu=jsrelu)
        for a in self.activations:
            print(f'\t{a}')
        print(f'Sparsify: {len(self.activations)} activations replaced')
        
        if magic_synapse:
            print("MagicSynapse: Plugging in")
            model = MagicSynapse.plug_in(model=model, rho=rho, filter=self.magic_synapse_filter)
            print("MagicSynapse: Finished")

        model = self.model_type(
            self.wrapper_type(model),
            RestrictAffinePlugin(log_per_step=log_per_step) if restricted_affine else None,
            ActivationDistributionPlugin(self.mlp_types, log_per_step),
            ZerothBiasPlugin(zeroth_bias_clipping, log_per_step=log_per_step) if db_mlp else None,
            SpectralIncreasePlugin(self.mlp_types, self.extract_linear_layers, log_per_step=log_per_step),
            EffectiveGradientSparsity(self.mlp_types, self.extract_linear_layers, log_per_step=log_per_step),
            VGradientObservationPlugin(mlp_types=self.mlp_types, log_per_step=log_per_step),
        ).to(device)

        model.iteration = steps if steps is not None else epoch_size * start_epoch 
        model.epoch = start_epoch
        model.losses = LossManager(writer=writer)
        start_tensorboard_server(writer.logdir)

        if db_mlp and db_mlp_shape is None:
            # make parameters of dynamic modules of implicit adversarial samples ready
            assert dataloader is not None
            with torch.no_grad():
                X, Y = next(iter(dataloader))
                pred = model(X.to(device)[:physical_batch_size])
                if hasattr(model, 'clean'):
                    model.clean()
        return model, writer, writer.logdir