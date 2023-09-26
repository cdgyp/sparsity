import os
from typing import Any
import torch
from ..base import new_experiment, Model, Wrapper, ERM, start_tensorboard_server, replace_config, SpecialReplacement, LossManager
from .hooks import ActivationObservationPlugin, GradientNoisePlugin, SimilarityPlugin, ParameterChangePlugin, ActivationDistributionPlugin, DiagonalityPlugin, SpectralObservationPlugin, EffectiveGradientSparsity, SpectralIncreasePlugin, VGradientObservationPlugin
from .activations import JumpingSquaredReLU, CustomizedReLU, ActivationPosition, CustomizedGELU, MixedActivation, LinearActivationMixing
from .robustness import RestrictAffinePlugin, DoublyBiased, ZerothBiasPlugin
from .magic import MagicSynapse
from torch.distributed import get_rank

class Sparsify:
    def __init__(self, 
        db_mlp: bool,
        jsrelu: bool,
        magic_synapse: bool,
        restricted_affine: bool=None,
        zeroth_bias_clipping=0.1,
        db_mlp_shape=None,
        rho=0.1,
        log_per_step=10,
        mixed_scheduling={'max_epoch': None, 'max_iteration': None},
        lora_r=None,
        model_type=Model, 
        wrapper_type=Wrapper,
    ) -> None:
        self.activations = []
        self.mlps = []
        self.mlp_types = []
        self.model_type = model_type
        self.wrapper_type = wrapper_type

        self.db_mlp = db_mlp

        self.jsrelu = jsrelu
        self.magic_synapse = magic_synapse
        self.restricted_affine = restricted_affine
        self.zeroth_bias_clipping = zeroth_bias_clipping
        self.db_mlp_shape = db_mlp_shape
        self.rho = rho
        self.log_per_step = log_per_step
        self.mixed_scheduling = mixed_scheduling
        self.lora_r = lora_r

        if self.restricted_affine is None:
            self.restricted_affine = self.db_mlp

        print(self.mixed_scheduling)


    
    def extract_linear_layers(self, mlp) -> 'dict[str, torch.nn.Linear]':
        pass

    def activation_function_filter(self, path, name, module):
        return ('relu' in module.__class__.__name__.lower()) or isinstance(module, ActivationPosition)

    def replace_activations(self, model: torch.nn.Module, jsrelu, path='model'):
        if isinstance(jsrelu, str):
            jsrelu = (jsrelu.lower() == 'jsrelu' or jsrelu.lower() == 'jumpring_squared_relu')
        if isinstance(jsrelu, bool):
            if jsrelu:
                make_act = lambda: ActivationPosition(JumpingSquaredReLU())
            else:
                make_act = lambda: ActivationPosition(CustomizedReLU())
        else:
            make_act = jsrelu
            
        
        for name, module in model.named_children():
            p = '.'.join([path, name])
            if self.activation_function_filter(path, name, module):
                setattr(model, name, make_act())
                self.activations.append(f"{p}: {module.__class__}")
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

    def _make_model(self, model, finetuning, has_obs=True):
        obs: 'list[torch.nn.Module]' = [
            RestrictAffinePlugin(log_per_step=self.log_per_step, finetuning=(finetuning is not None)) if self.restricted_affine else None,
            ActivationDistributionPlugin(self.mlp_types, self.log_per_step),
            ZerothBiasPlugin(self.zeroth_bias_clipping, log_per_step=self.log_per_step) if self.db_mlp else None,
            SpectralIncreasePlugin(self.mlp_types, self.extract_linear_layers, log_per_step=self.log_per_step),
            EffectiveGradientSparsity(self.mlp_types, self.extract_linear_layers, log_per_step=self.log_per_step),
            VGradientObservationPlugin(mlp_types=self.mlp_types, log_per_step=self.log_per_step),
            LinearActivationMixing(**self.mixed_scheduling) if self.jsrelu == 'mixed' or (isinstance(self.jsrelu, bool) and self.jsrelu and finetuning) else None,
        ]
        if not has_obs:
            for ob in obs:
                if ob is not None:
                    assert len(list(ob.parameters())) == 0
            obs = []
        
        model = self.model_type(
            self.wrapper_type(model),
            *obs
        )

        return model

    def load_checkpoint(self, model: torch.nn.Module, path: str, strict: bool):
        checkpoint = torch.load(path, map_location="cpu")
        matching_status = model.load_state_dict(checkpoint["model"], strict=strict)
        assert len(matching_status.unexpected_keys) == 0, (matching_status.unexpected_keys, matching_status.missing_keys)
        assert all(['lora' in key for key in matching_status.missing_keys]), (matching_status.unexpected_keys, matching_status.missing_keys)

        return model
    
    def __call__(self, 
        name: str,
        title: str,
        model: torch.nn.Module,
        resume=None,
        finetuning=None,
        epoch_size=0, start_epoch=0, steps=None,
        dataloader=None,
        tensorboard_server=True,
        device='cuda',
    ):

        if resume is not None and len(resume) > 0:
            if 'save' in resume:
                dir = resume[:resume.find('save')]
            else:
                dir = resume
        else:
            dir = name + '/' + title + '/'

        main = model
        del model

        writer, _ = new_experiment(dir, None, dir_to_runs='runs', resume=resume is not None and len(resume) > 0, device=device)

        if finetuning is not None:
            model = self._make_model(main, finetuning, has_obs=False)

            if self.lora_r is not None:
                from .lora import LoRAfy
                model = LoRAfy(self.lora_r)(model)
                strict = False
            else:
                strict = True

            print(f"finetuning for sparsity from {finetuning}")
            model = self.load_checkpoint(model, finetuning, strict)
            main = model.main.model
            del model

        if self.db_mlp:
            print("Sparsify: replacing singly biased MLPs")
            main = self.replace_MLPs(main, self.zeroth_bias_clipping, self.db_mlp_shape)
            for m in self.mlps:
                print(f'\t{m}')
            print(f'Sparsify: {len(self.mlps)} MLP blocks replaced')

        print("Sparsify: replacing activation functions")
        jsrelu = self.jsrelu
        if finetuning and isinstance(jsrelu, bool) and jsrelu:
            jsrelu = 'mixed'
        if isinstance(jsrelu, str) and jsrelu == 'mixed':
            mixed_activation_maker = lambda: ActivationPosition(MixedActivation(CustomizedReLU(), JumpingSquaredReLU()))
            jsrelu = mixed_activation_maker
        else:
            jsrelu = self.jsrelu
        main = self.replace_activations(main, jsrelu=jsrelu)
        for a in self.activations:
            print(f'\t{a}')
        print(f'Sparsify: {len(self.activations)} activations replaced to {self.jsrelu}')
        
        if self.magic_synapse:
            print("MagicSynapse: Plugging in")
            main = MagicSynapse.plug_in(model=main, rho=self.rho, filter=self.magic_synapse_filter)
            print("MagicSynapse: Finished")
        
        model = self._make_model(main, finetuning, has_obs=True).to(device)

        model.iteration = steps if steps is not None else epoch_size * start_epoch 
        model.epoch = start_epoch
        model.losses = LossManager(writer=writer)
        if tensorboard_server:
            start_tensorboard_server(writer.logdir)

        if self.db_mlp and self.db_mlp_shape is None:
            # make parameters of dynamic modules of implicit adversarial samples ready
            assert dataloader is not None
            with torch.no_grad():
                X, Y = next(iter(dataloader))
                pred = model(X.to(device)[:1])
                if hasattr(model, 'clean'):
                    model.clean()
        return model, writer, writer.logdir