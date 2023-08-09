from .mymodel import get_model
import torch
import os
import argparse

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from .imagenet1k import load_data
from ...modules.relu_vit import MLPBlock
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description="PyTorch Classification Training")
parser.add_argument("--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
parser.add_argument("--from-scratch", action='store_true')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--title', type=str, default="weight_norms")
parser.add_argument('--device', type=str, required=True)
parser.add_argument('--log_per_step', type=int, default=1)
parser.add_argument('--path-to-checkpoints', type=str, required=True)
parser.add_argument('--physical-batch-size', type=int, default=32)
args = parser.parse_args()

checkpoints = [c for c in os.listdir(args.path_to_checkpoints) if '.pth' in c and 'model' in c]
sorted_checkpoints = sorted(checkpoints, key=lambda s: int(s[s.find('_')+1:s.find('.')]))
print(sorted_checkpoints)

train_dir = os.path.join(args.data_path, "train")
val_dir = os.path.join(args.data_path, "val")
dataset = ImageFolder(train_dir, transform=transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
]))

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    pin_memory=True,
    drop_last=True,
)

model = get_model(
    'sparsified',
    data_loader,
    args,
    100, 0
)

def resume(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    checkpoint = None
def weight_norms(i, checkpoint: str, model: torch.nn.Module):
    writer: SummaryWriter = model.losses.writer
    norms = []
    mlp_norms = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            norms.append(m.weight.norm(2)**2 / max(*m.weight.shape))
        if not isinstance(m, MLPBlock):
            continue
        for mm in m.modules():
            if isinstance(mm, torch.nn.Linear):
                weight = mm.weight
                mlp_norms.append(weight.norm(2)**2 / max(*weight.shape))
    mean_norm = torch.stack(norms).mean()
    mean_mlp_norm = torch.stack(mlp_norms).mean()
    writer.add_scalar('mean_norm', float(mean_norm), i)
    writer.add_scalar('mean_mlp_norm', float(mean_mlp_norm), i)
    print(i, checkpoint, float(mean_norm), float(mean_mlp_norm))



with torch.no_grad():
    for i, checkpoint in enumerate(tqdm(sorted_checkpoints)):
        resume(os.path.join(args.path_to_checkpoints, checkpoint))
        mean_norm = weight_norms(i, checkpoint, model)