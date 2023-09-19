import torch
from numpy.random import choice

def get_eval_dataset(eval_dataset, max_eval_samples):
    use_n_samples = max_eval_samples
    if use_n_samples < len(eval_dataset):
        eval_idx = choice(
            range(0, len(eval_dataset)), size=use_n_samples, replace=False)
        eval_dataset_subset = torch.utils.data.Subset(
            eval_dataset, eval_idx)
    else:
        eval_dataset_subset = eval_dataset
    return eval_dataset_subset