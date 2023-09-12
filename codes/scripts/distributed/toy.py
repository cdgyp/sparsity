import torch
from torch import nn
from torch.utils.data import  Dataset, DataLoader
from ...base import device



class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return self.len
    

def get_rand_dataloader(dim_input, n_sample, batch_size):
    dataset = RandomDataset(dim_input, n_sample)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    ), dataset


class Model(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(Model, self).__init__()
        self.fc = nn.Linear(dim_input, dim_output)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input shape", input.size(),
              "output shape", output.size())

        return output
    
class Hook:
    def __init__(self) -> None:
        self.input = []
    def __call__(self, module, args, out):
        assert isinstance(args, tuple)
        self.input.append(args[0])
        print(len(self.input))
    
def prepare(
    dim_input=5,
    dim_output=2,
    batch_size=30,
    n_sample=100,
    hooked=False
):
    model = Model(dim_input=dim_input, dim_output=dim_output)
    if hooked:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.register_forward_hook(Hook())
    dataloader, dataset = get_rand_dataloader(dim_input=dim_input, n_sample=n_sample, batch_size=batch_size)
    return model, dataloader, dataset