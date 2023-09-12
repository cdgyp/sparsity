import torch
from torch import nn
from .toy import prepare, device


model, dataloader = prepare()

if torch.cuda.device_count() > 1:
    print("Using ", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)

model = model.to(device)


for data in dataloader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())