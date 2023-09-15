from datasets import load_dataset, Dataset, load_from_disk
from tqdm.auto import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--n-train', type=int, default=100000)
parser.add_argument('--n-val', type=int, default=100000)
parser.add_argument('--chunk-size', type=int, default=1000000)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

def download_subset(*names, split="train", sample_count=10000, save_path=".", chunk_size=10000000, seed=42):
    """
    下载数据集的子集并保存到指定位置

    参数:
    *names (str): 数据集名称的各个部分，它们将被合并并用作load_dataset的名称参数。
    split (str): 数据集的分割部分，例如 "train", "test" 或 "validation"。
    sample_count (int): 需要的样本数量。
    save_path (str): 保存子集的位置。
    chunk_size (int): 每个子集的大小。

    """
    
    dataset_streaming = load_dataset(*names, split=split, streaming=True)
    if seed is not None:
        dataset_streaming = dataset_streaming.shuffle(seed=seed, buffer_size=400_000_000)
    selected_samples = []
    chunk_num = 0

    for idx, sample in tqdm(enumerate(dataset_streaming), total=sample_count):
        if idx == sample_count:
            break
        selected_samples.append(sample)
        
        # 当收集的样本达到 chunk_size 时，保存它们
        if len(selected_samples) == chunk_size:
            subset = Dataset.from_dict({k: [dic[k] for dic in selected_samples] for k in selected_samples[0]})
            subset_save_path = f"{save_path}_part_{chunk_num}"
            subset.save_to_disk(subset_save_path)
            selected_samples = []
            chunk_num += 1

    # 保存剩余的样本（如果有的话）
    if selected_samples:
        subset = Dataset.from_dict({k: [dic[k] for dic in selected_samples] for k in selected_samples[0]})
        subset_save_path = f"{save_path}_part_{chunk_num}"
        subset.save_to_disk(subset_save_path)

download_subset("c4", "en", split="train", sample_count=args.n_train, save_path='data/c4/train', chunk_size=args.chunk_size, seed=args.seed)
download_subset("c4", "en", split="validation", sample_count=args.n_val, save_path='data/c4/validation', chunk_size=args.chunk_size, seed=args.seed)

if args.n_train < 100:
    loaded = load_from_disk('data/c4/train_part_0')
    for s in loaded:
        print(s)
