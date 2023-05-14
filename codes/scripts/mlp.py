import argparse
from torchvision.datasets import ImageFolder, CIFAR10, MNIST
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch import nn

from ..base import new_experiment, Training, Model, Wrapper, device,  WrapperDataset, ERM, DeviceSetter, start_tensorboard_server
from ..modules.hooks import ActivationObservationPlugin
from ..modules.vit import FeedForward

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'RMSprop'])
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--n_epoch', type=int, default=10)
parser.add_argument('--grad_clipping', type=float, default=None)
parser.add_argument('--hidden_dim', type=int, default=3072)
all_args = parser.parse_known_args()
args = all_args[0]

writer = new_experiment('mlp_' + str(args), args)

args.lr = args.lr / (512 / args.batch_size)

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

train_dataset = MNIST('/data/pz/sparsity/mnist', True, transform=train_transforms, download=True)
test_dataset = MNIST('/data/pz/sparsity/mnist', False, transform=test_transforms, download=True)


def make_dataloaders(dataset: ImageFolder):
    # indices = [i for i, c in enumerate(dataset.targets) if c < args.num_classes]
    # subset = WrapperDataset(Subset(dataset, indices))
    subset = WrapperDataset(dataset)

    dataloader = DataLoader(subset, args.batch_size, True, num_workers=8, drop_last=True)

    return dataloader

train_dataloader = make_dataloaders(train_dataset)
test_dataloader = make_dataloaders(test_dataset)

observation = ActivationObservationPlugin()

class MLP(nn.Module):
    def __init__(self, width_in, width_out, width_hidden=3072):
        super().__init__()
        self.network = FeedForward(dim=width_in, hidden_dim=width_hidden, dim_out=width_out)

    def forward(self, X):
        X = X.flatten(1)
        return self.network(X)

vit = Model(
    Wrapper(
        MLP(
            width_in=args.image_size ** 2, 
            width_out=args.num_classes, 
            width_hidden=args.hidden_dim
        )
    ),
    ERM(),
    observation
).to(device)


training = Training(
    n_epoch=args.n_epoch,
    model=DeviceSetter(vit),
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    lr=args.lr,
    writer=writer,
    test_every_epoch=1,
    save_every_epoch=None,
    optimizer=args.optimizer,
    weight_decay=args.weight_decay,
    gradient_clipping=args.grad_clipping
)

start_tensorboard_server(writer.log_dir)
training.run()
writer.add_hparams(
    vars(args),
    {
        **{f'hparam/diagonal_gradients/mean': float(observation.diagonal_gradients.mean())},
        **{f'hparam/diagonal_gradients/{i}': float(dg) for i, dg in enumerate(observation.diagonal_gradients)},
        **{f'hparam/gradient_ratios/mean': float(observation.gradient_ratios.mean())},
        **{f'hparam/gradient_ratios/{i}': r for i, r in enumerate(observation.gradient_ratios)}
    }
)