import torch
import os
from torch.utils.data import DataLoader
def get_model(model_type: str, dataloader: DataLoader, args=None, epoch_size=0, start_epoch=1):
    from ...base import new_experiment, Model, Wrapper, ERM, start_tensorboard_server, replace_config, SpecialReplacement, LossManager
    from ...modules.hooks import ActivationObservationPlugin, GradientNoisePlugin, SimilarityPlugin, ParameterChangePlugin, ActivationDistributionPlugin, DiagnoalityPlugin
    from ...modules.relu_vit import relu_vit_b_16, ViT_B_16_Weights, MLPBlock, ImplicitAdversarialSample
    from ...modules.activations import JumpingSquaredReLU, CustomizedReLU, ActivationPosition, CustomizedGELU
    from ...modules.robustness import ImplicitAdversarialSamplePlugin
    sparsified = (model_type == 'sparsified')
    if args.resume is not None and len(args.resume) > 0:
        assert 'save' in args.resume, args.resume
        dir = args.resume[:args.resume.find('save')]
    else:
        dir = 'imagenet1k/' + args.title + '/' + f'{(model_type)}'

    writer, _ = new_experiment(dir, None, dir_to_runs='runs', resume=args.resume is not None and len(args.resume) > 0)

    if sparsified:
        default_activation_layer = JumpingSquaredReLU
    else:
        default_activation_layer = CustomizedReLU
    MLPBlock.default_activation_layer = lambda: ActivationPosition(default_activation_layer())
    model = Model(
        Wrapper(
            relu_vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if not args.from_scratch else None, progress=True, **{
                'num_classes': 1000,
                'rezero': False,
                'implicit_adversarial_samples': sparsified
            })
        ),
        # ActivationObservationPlugin(p=1, depth=12, batchwise_reported=False, log_per_step=args.log_per_step, pre_activation_only=True),
        # GradientNoisePlugin(log_per_step=args.log_per_step),
        # SimilarityPlugin(log_per_step=args.log_per_step),
        # ParameterChangePlugin(log_per_step=args.log_per_step),
        ActivationDistributionPlugin(12, log_per_step=args.log_per_step),
        ImplicitAdversarialSamplePlugin(args.implicit_adversarial_samples_clipping),
        DiagnoalityPlugin(12, log_per_step=args.log_per_step),
    ).to(args.device)

    model.iteration = epoch_size * start_epoch
    model.epoch = start_epoch
    model.losses = LossManager(writer=writer)
    start_tensorboard_server(writer.log_dir)
    args.output_dir = os.path.join(writer.log_dir, 'save')

    if sparsified:
        # make parameters of dynamic modules of implicit adversarial samples ready
        with torch.no_grad():
            X, Y = next(iter(dataloader))
            pred = model(X.to(args.device)[:args.physical_batch_size])

    return model