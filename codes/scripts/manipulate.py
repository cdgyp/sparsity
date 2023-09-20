import argparse
import torch
from torchvision.datasets import ImageFolder, CIFAR10, ImageNet
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from torch  import nn
from torchvision import transforms
from functools import partial

from ..base import new_experiment, Training, Model, Wrapper, device,  WrapperDataset, ERM, DeviceSetter, start_tensorboard_server, replace_config, SpecialReplacement
from ..modules.hooks import ActivationObservationPlugin, GradientNoisePlugin, SimilarityPlugin, ParameterChangePlugin, ActivationDistributionPlugin, DiagonalityPlugin, EffectiveGradientSparsity
from ..modules.relu_vit import relu_vit_b_16, ViT_B_16_Weights, MLPBlock
from ..modules.activations import SymmetricReLU, SReLU, WeirdLeakyReLU, Shift, ActivationPosition, careful_bias_initialization, CustomizedReLU, SquaredReLU, SShaped, JumpingSquaredReLU
from ..modules.robustness import ZerothBiasPlugin
from ..data.miniimagenet import MiniImagenet
from torchvision.datasets import ImageNet, ImageFolder
from .imagenet1k.mymodel import ImageNet1kSparsify



parser = argparse.ArgumentParser()
parser.add_argument('--title', type=str, default='vit')
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'RMSprop'])
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--n_epoch', type=int, default=10)
parser.add_argument('--grad_clipping', type=float, default=None)
parser.add_argument('--p', type=float, default=1)
parser.add_argument('--batchwise_reported', type=int, default=0)
parser.add_argument('--warmup_epoch', type=int, default=5)
parser.add_argument('--initial_lr_ratio', type=float, default=1e-1)
parser.add_argument('--activation_layer', type=str, default='relu')
parser.add_argument('--pretrained', type=int, default=1)
parser.add_argument('--log_per_step', type=int, default=10)
parser.add_argument('--half_interval', type=float, default=0.5)
parser.add_argument('--shift_x', type=float, default=0)
parser.add_argument('--shift_y', type=float, default=0)
parser.add_argument('--alpha_x', type=float, default=1)
parser.add_argument('--alpha_y', type=float, default=1)
parser.add_argument('--careful_bias_initialization', type=int, default=0)
parser.add_argument('--rezero', type=int, default=0)
parser.add_argument('--zeroth-bias', type=int, default=0)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--mixed_precision', action='store_true')
parser.add_argument('--resume', type=str, default=None)

all_args = parser.parse_known_args()
args = all_args[0]


print('not known params', all_args[1])
writer, ref_hash = new_experiment(args.title + '/' + str(replace_config(args, title=SpecialReplacement.DELETE)), args)

if args.activation_layer == 'relu':
    make_activation_layer = CustomizedReLU
elif args.activation_layer == 'symmetric_relu':
    make_activation_layer = SymmetricReLU
elif args.activation_layer == 's_relu':
    make_activation_layer = SReLU.new(0.5)
elif args.activation_layer == 'leaky_relu':
    make_activation_layer = lambda: nn.LeakyReLU(0.1)
elif args.activation_layer == 'weird_leaky_relu':
    make_activation_layer = WeirdLeakyReLU.get_constructor(0.1, 1)
elif args.activation_layer == 'weird':
    make_activation_layer = lambda: SShaped(CustomizedReLU(), args.half_interval)
elif ('relu2' in args.activation_layer or 'squared_relu' in args.activation_layer) and 'jump' not in args.activation_layer:
    make_activation_layer = SquaredReLU
    if 'weird' in args.activation_layer:
        make_activation_layer = lambda: SShaped(SquaredReLU(), args.half_interval)
elif 'jumping' in args.activation_layer:
    make_activation_layer = JumpingSquaredReLU
    if 'weird' in args.activation_layer:
        make_activation_layer = lambda: SShaped(JumpingSquaredReLU(), args.half_interval)
else:
    raise NotImplemented(args.activation_layer)


train_transforms = transforms.Compose([
    transforms.Resize(args.image_size + 32),
    transforms.RandomCrop(args.image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.ToTensor()
])

if args.dataset == 'cifar10':
    train_dataset = CIFAR10('/data/cifar10', True, transform=train_transforms, download=True)
    test_dataset = CIFAR10('/data/cifar10', False, transform=test_transforms, download=True)
elif args.dataset == 'imagenet1k':
    train_dataset = ImageFolder('data/imagenet1k256/ILSVRC/Data/CLS-LOC/train', transform=train_transforms)
    test_dataset = ImageFolder('data/imagenet1k256/ILSVRC/Data/CLS-LOC/val', transform=test_transforms)

# train_dataset = MiniImagenet('/data/datasets/miniimagenet', 'train', args.image_size)
# test_dataset = MiniImagenet('/data/datasets/miniimagenet', 'test', args.image_size)


def make_dataloaders(dataset: ImageFolder):
    # indices = [i for i, c in enumerate(dataset.targets) if c < args.num_classes]
    # subset = WrapperDataset(Subset(dataset, indices))
    subset = WrapperDataset(dataset)

    dataloader = DataLoader(subset, args.batch_size, True, num_workers=8, drop_last=True)

    return dataloader

train_dataloader = make_dataloaders(train_dataset)
test_dataloader = make_dataloaders(test_dataset)

observation = ActivationObservationPlugin(p=args.p, depth=12, batchwise_reported=bool(args.batchwise_reported), log_per_step=args.log_per_step)

def make_model(
    model_type: str, 
    args,
    dataloader
):
    vit = relu_vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if not args.from_scratch else None, progress=True, wide=args.wide, **{
        'num_classes': 1000,
        'rezero': False,
        'norm_layer': partial(torch.nn.LayerNorm, eps=1e-6),
    })
    
    model, _, output_dir = ImageNet1kSparsify()(
            args.dataset,
            args.title + '/' + model_type,
            vit,
            args.zeroth_bias,
            lambda: ActivationPosition(make_activation_layer()),
            False,
            args.zeroth_bias,
            log_per_step=args.log_per_step,
            device="cuda",
            resume=args.resume,
            dataloader=dataloader,
            physical_batch_size=args.batch_size
        )
    
    args.output_dir = output_dir

    return model

if args.zeroth_bias and args.activation_layer == 'jumping_squared_relu':
    model_type = 'Full'
elif args.zeroth_bias:
    model_type = 'DB-MLP_only'
elif args.activation_layer == 'jumping_squared_relu':
    model_type = 'JSReLU_only'
else:
    model_type = 'Vanilla'


vit = make_model(model_type=model_type, args=args,dataloader=train_dataloader)
    

if args.careful_bias_initialization:
    careful_bias_initialization(vit, args.shift_x)

def make_scheduler(optim):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, 
            T_max=args.n_epoch, 
            eta_min=0.0, 
            verbose=True
        )

training = Training(
    n_epoch=args.n_epoch,
    model=DeviceSetter(vit),
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    lr=args.lr,
    writer=vit.writer,
    test_every_epoch=1,
    save_every_epoch=20,
    optimizer=args.optimizer,
    weight_decay=args.weight_decay,
    gradient_clipping=args.grad_clipping,
    log_per_step=args.log_per_step,
    mixed_precision=args.mixed_precision,
    scheduler_maker=make_scheduler
)

training.run()