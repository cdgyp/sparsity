import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP

from .toy import prepare
from ...base import new_experiment, LossManager

from tqdm.auto import tqdm


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '2333'

    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
    )

def cleanup():
    dist.destroy_process_group()


def demo():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Start running basic DDP example on rank {rank} of {world_size}.")

    writer = new_experiment('test', None, device=rank)
    losses = LossManager(writer=writer)

    model, dataloader, dataset = prepare(n_sample=120000)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    d_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=30, pin_memory=True, drop_last=False, sampler=d_sampler)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=1e-3)

    losses.observe(dist.get_rank(), "general")
    losses.observe(dist.get_rank(), "specific", dist.get_rank())
    losses.observe(dist.get_rank(), "specific-2", int(dist.get_rank() // 2))
    losses.log_losses(1)

    exit()

    for X in tqdm(dataloader):
        pred = ddp_model(X)
        Y = torch.randint(0, 2, [X.shape[0]], device=pred.device)
        loss_fn(pred, Y).backward()
        optimizer.step()

    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    

if __name__ == "__main__":
    demo()