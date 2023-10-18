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
    parser.add_argument('--epsilon', type=float, required=True)
    parser.add_argument('--title', type=str, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--n-samples', type=int, required=True)
    parser.add_argument('--do-filtering', action='store_true')
    parser.add_argument('--filter-threshold', type=float, default=None, help="Defaults to None, meaning using soft spectral filtering")
    args = parser.parse_args()

    weight = ViT_B_16_Weights.DEFAULT
    
    model = vit_b_16(weight).to('cuda')

    preprocessing = weight.transforms()
    preprocessing = torchvision.transforms.Compose([preprocessing])
    dataset = ImageFolder(args.dataset, transform=preprocessing)
    dataloader = DataLoader(dataset, args.batch_size, num_workers=8, shuffle=True, drop_last=True, pin_memory=True)

    
    adv = FGSMExample(dataloader, model, torch.nn.CrossEntropyLoss(), epsilon=args.epsilon, test_fn=accuracy)
    X, adversarial_X, Ys, baseline_test, adversarial_test = adv.run(args.n_samples)
    baseline_acc = baseline_test.mean()
    adversarial_acc = adversarial_test.mean()
    print(f"epsilon={args.epsilon}, baseline_acc={float(baseline_acc)}, adv_acc={float(adversarial_acc)}")

    writer, logdir = new_experiment('covering/' + args.title, args, dir_to_runs='runs')

    model.eval()

    full_model = Model(
        Wrapper(model),
        AdversarialObservation(do_filtering=args.do_filtering, filter_threshold=args.filter_threshold),
    ).to('cuda')
    full_model.losses = LossManager(writer=writer)

    adv_dataset = BaselineAdversarialPairDataset(X, adversarial_X, Ys)

    with torch.no_grad():
        all_adv_acc = []
        all_baseline_acc = []
        for (baseline_X, adv_X, Y) in tqdm(DataLoader(adv_dataset, int(args.batch_size // 2), False, drop_last=True, collate_fn=BaselineAdversarialPairDataset.collate_fn)):
            all_X = torch.cat([baseline_X, adv_X])
            pred = full_model(all_X.to('cuda'))
            adv_acc = accuracy(pred[len(pred) // 2: ], Y)
            baseline_acc = accuracy(pred[:len(pred) // 2], Y)
            print("baseline_acc:", baseline_acc.mean(), "adv_acc:", adv_acc.mean())
            all_adv_acc.append(adv_acc)
            all_baseline_acc.append(baseline_acc)

        print("all_baseline_acc:", float(torch.stack(all_baseline_acc).mean()), "all_adv_acc:", float(torch.stack(all_adv_acc).mean()))

    full_model.losses.log_losses(1)
    start_tensorboard_server(writer.logdir)



if __name__ == "__main__":
    main()
