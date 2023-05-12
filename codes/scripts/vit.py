import argparse
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from ..base import new_experiment, Training, Model, Wrapper, device,  WrapperDataset, ERM, DeviceSetter, start_tensorboard_server
from ..modules.hooks import ActivationObservationPlugin
from vit_pytorch.vit import ViT 

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'RMSprop'])
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--n_epoch', type=int, default=10)
all_args = parser.parse_known_args()
args = all_args[0]

writer = new_experiment(str(args), args)

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

train_dataset = ImageFolder(
    root='/data/datasets/tiny-imagenet-200/train', 
    transform=train_transforms
)
test_dataset = ImageFolder(
    root='/data/datasets/tiny-imagenet-200/test', 
    transform=test_transforms
)

def make_dataloaders(dataset: ImageFolder):
    indices = [i for i, c in enumerate(dataset.targets) if c < args.num_classes]
    subset = WrapperDataset(Subset(dataset, indices))

    dataloader = DataLoader(subset, args.batch_size, True, num_workers=8, drop_last=True)

    return dataloader

train_dataloader = make_dataloaders(train_dataset)
test_dataloader = make_dataloaders(test_dataset)

vit = Model(
    Wrapper(
        ViT(
            image_size=args.image_size,
            patch_size=16,
            num_classes=args.num_classes,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dropout=args.dropout
        )
    ),
    ERM(),
    ActivationObservationPlugin()
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
    weight_decay=args.weight_decay
)

start_tensorboard_server(writer.log_dir)
training.run()