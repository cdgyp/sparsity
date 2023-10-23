import torch
from torch.utils.data import DataLoader, Dataset
import argparse
from ...procedures.adversarial import FGSMExample, AdversarialObservation
import torchvision
from torchvision.datasets import ImageNet, ImageFolder
from torchvision import transforms
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from torchvision.transforms._presets import ImageClassification
from ...base import Model, Wrapper, LossManager, new_experiment, start_tensorboard_server
from tqdm.auto import tqdm
from math import ceil

def accuracy(pred: torch.Tensor, Y: torch.Tensor):
    return (pred.argmax(dim=-1) == Y).float()

class BaselineAdversarialPairDataset(Dataset):
    def __init__(self, baseline: torch.Tensor, adversarial: torch.Tensor, Y: torch.Tensor) -> None:
        super().__init__()
        self.baseline = baseline
        self.adversarial = adversarial
        self.Y = Y
        assert len(self.baseline) == len(self.adversarial) and len(self.adversarial) == len(self.Y)

    def __getitem__(self, index):
        return torch.stack([self.baseline[index], self.adversarial[index]], dim=0), torch.stack([self.Y[index], self.Y[index]])
    def __len__(self):
        return len(self.Y)
    def collate_fn(l: 'list[tuple[torch.Tensor, torch.Tensor]]'):
        baseline_X = torch.stack([X[0] for X, Y in l])
        adv_X = torch.stack([X[1] for X, Y in l])
        Y = torch.stack([Y[0] for X, Y in l])
        return baseline_X, adv_X, Y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epsilon', required=True)
    parser.add_argument('--title', type=str, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--n-samples', type=int, required=True)
    parser.add_argument('--do-filtering', action='store_true')
    parser.add_argument('--log-per-step', type=int, default=10)
    parser.add_argument('--filter-threshold', type=float, default=None, help="Defaults to None, meaning using soft spectral filtering")
    args = parser.parse_args()

    if isinstance(args.epsilon, str):
        args.epsilon = eval(args.epsilon)



    weight = ViT_B_16_Weights.DEFAULT
    
    model = vit_b_16(weight).to('cuda')

    preprocessing = weight.transforms()
    preprocessing = torchvision.transforms.Compose([preprocessing])
    dataset = ImageFolder(args.dataset, transform=preprocessing)
    dataloader = DataLoader(dataset, args.batch_size, num_workers=8, shuffle=True, drop_last=True, pin_memory=True)

    
    adv = FGSMExample(dataloader, model, torch.nn.CrossEntropyLoss(), epsilon=args.epsilon, test_fn=accuracy, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    writer, logdir = new_experiment('covering/' + args.title, args, dir_to_runs='runs')
    start_tensorboard_server(writer.logdir)

    model.eval()

    adversarial_plugin = AdversarialObservation(do_filtering=args.do_filtering, filter_threshold=args.filter_threshold)
    adversarial_plugin.activated = False
    full_model = Model(
        Wrapper(model),
        adversarial_plugin,
    ).to('cuda')
    full_model.losses = LossManager(writer=writer)

    all_adv_acc = []
    all_adv2_acc = []
    all_baseline_acc = []
    for t, (baseline_X, adv_X, Y, bl_test, adv_test) in enumerate(tqdm(adv.run(args.n_samples, True), total=min(ceil(args.n_samples / dataloader.batch_size), len(dataloader)))):
        with torch.no_grad():
            adversarial_plugin.activated = True
            all_X = torch.cat([baseline_X, adv_X])
            pred = full_model(all_X.to('cuda'))
            adv2_acc = accuracy(pred[len(pred) // 2: ], Y)
            print("baseline_acc:", bl_test.mean(), "adv_acc:", adv_test.mean(), "adv2_acc:", adv2_acc.mean())
            all_adv_acc.append(adv_test)
            all_adv2_acc.append(adv2_acc)
            all_baseline_acc.append(bl_test)

            full_model.losses.observe(bl_test.mean(), "acc", "baseline")
            full_model.losses.observe(adv_test.mean(), "acc", "adversarial")
            full_model.losses.observe(adv2_acc.mean(), "acc", "adversarial2")
            
            full_model.losses.observe(args.epsilon, "eps")
            
            adversarial_plugin.activated = False
            for h in adversarial_plugin.hooks:
                h.activated = False
        
        if t % args.log_per_step == 0:
            full_model.losses.log_losses(t) # ! don't call .reset()

    print("all_baseline_acc:", float(torch.stack(all_baseline_acc).mean()), "all_adv_acc:", float(torch.stack(all_adv_acc).mean()), "all_adv2_acc:", float(torch.stack(all_adv2_acc).mean()))




if __name__ == "__main__":
    main()
