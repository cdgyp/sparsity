from datasets import load_dataset, Dataset, load_from_disk
from tqdm.auto import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--n-train', type=int, default=100000)
parser.add_argument('--n-val', type=int, default=100000)
args = parser.parse_args()


def download_subset(*names, split="train", sample_count=10000, save_path="."):
    """
    下载数据集的子集并保存到指定位置

    参数:
    *names (str): 数据集名称的各个部分，它们将被合并并用作load_dataset的名称参数。
    split (str): 数据集的分割部分，例如 "train", "test" 或 "validation"。
    sample_count (int): 需要的样本数量。
    save_path (str): 保存子集的位置。

    返回:
    None
    """

    dataset_streaming = load_dataset(*names, split=split, streaming=True)

    selected_samples = []
    for idx, sample in tqdm(enumerate(dataset_streaming), total=sample_count):
        if idx == sample_count:
            break
        selected_samples.append(sample)

    subset = Dataset.from_dict({k: [dic[k] for dic in selected_samples] for k in selected_samples[0]})
    subset.save_to_disk(save_path)

download_subset("c4", "en", split="train", sample_count=args.n_train, save_path='data/c4/train')
download_subset("c4", "en", split="validation", sample_count=args.n_val, save_path='data/c4/validation')

if args.n_train < 100:
    loaded = load_from_disk('data/c4/train')
    for s in loaded:
        print(s)