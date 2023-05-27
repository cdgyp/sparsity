import argparse
from torchvision.datasets import ImageFolder, CIFAR10
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LambdaLR
from torch  import nn
from torchvision import transforms

from ..base import new_experiment, Training, Model, Wrapper, device,  WrapperDataset, ERM, DeviceSetter, start_tensorboard_server, replace_config, SpecialReplacement, Confidence
from ..modules.hooks import ActivationObservationPlugin
from ..modules.relu_vit import relu_vit_b_16, ViT_B_16_Weights, MLPBlock, SymmetricReLU, VisionTransformer
from ..data.miniimagenet import MiniImagenet
from torchvision.datasets import ImageFolder



parser = argparse.ArgumentParser()
parser.add_argument('--title', type=str, default='vit')
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'RMSprop'])
parser.add_argument('--num_classes', type=int, default=31)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--n_epoch', type=int, default=10)
parser.add_argument('--grad_clipping', type=float, default=None)
parser.add_argument('--p', type=float, default=1)
parser.add_argument('--batchwise_reported', type=int, default=0)
parser.add_argument('--warmup_epoch', type=int, default=5)
parser.add_argument('--initial_lr_ratio', type=float, default=1e-1)
parser.add_argument('--activation_layer', type=str, default='relu')
parser.add_argument('--pretrained', type=int, default=1)
parser.add_argument('--no_noise', type=int, default=0)
all_args = parser.parse_known_args()
args = all_args[0]

print('not known params', all_args[1])
writer, ref_hash = new_experiment(args.title + '_' + str(replace_config(args, title=SpecialReplacement.DELETE)), args)

if args.activation_layer == 'relu':
    MLPBlock.default_activation_layer = nn.ReLU
elif args.activation_layer == 'symmetric_relu':
    MLPBlock.default_activation_layer = SymmetricReLU

# args.lr = args.lr / (512 / args.batch_size)

train_transforms = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.ToTensor()
])

# train_dataset = CIFAR10('/data/pz/sparsity/cifar10', True, transform=train_transforms, download=True)
# test_dataset = CIFAR10('/data/pz/sparsity/cifar10', False, transform=test_transforms, download=True)
train_dataset = ImageFolder('/data/pz/Office31/webcam', transform=train_transforms)
test_dataset = ImageFolder('/data/pz/Office31/webcam', transform=test_transforms)


def make_dataloaders(dataset: ImageFolder):
    # indices = [i for i, c in enumerate(dataset.targets) if c < args.num_classes]
    # subset = WrapperDataset(Subset(dataset, indices))
    subset = WrapperDataset(dataset)

    dataloader = DataLoader(subset, args.batch_size, True, num_workers=8, drop_last=True)

    return dataloader

train_dataloader = make_dataloaders(train_dataset)
test_dataloader = make_dataloaders(test_dataset)

observation = ActivationObservationPlugin(p=args.p, depth=12, batchwise_reported=bool(args.batchwise_reported))

vit = Model(
    Wrapper(
        VisionTransformer(
            image_size=args.image_size,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=args.num_classes,
            dropout=args.dropout
        )
    ),
    ERM(),
    observation
).to(device)

def print_and_return(x):
    print(x)
    return x


training = Training(
    n_epoch=args.n_epoch,
    model=DeviceSetter(vit),
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    lr=args.lr,
    writer=writer,
    test_every_epoch=5,
    save_every_epoch=None,
    optimizer=args.optimizer,
    weight_decay=args.weight_decay,
    gradient_clipping=args.grad_clipping,
    scheduler_args={
        'lr_lambda': lambda epoch: args.initial_lr_ratio + min(epoch / len(train_dataloader) / args.warmup_epoch, 1) * (1 - args.initial_lr_ratio),
        'last_epoch': args.warmup_epoch,
    },
    meta_batch_size=len(train_dataloader) if args.no_noise else 1
)

start_tensorboard_server(writer.log_dir)
training.run()
writer.add_hparams(
    {**vars(args), 'reference_has': ref_hash},
    observation.get_results()
)