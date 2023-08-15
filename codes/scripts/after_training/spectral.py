import os
from argparse import ArgumentParser
import torch
from matplotlib import pyplot as plt
from typing import Any
import pandas as pd
import re

from ...base.inspect import scan_checkpoints
from ...modules.hooks import SpectralObservationPlugin

parser = ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--device', type=str, required=True)
parser.add_argument('--threshold-ratio', type=float, default=10)
parser.add_argument('--output-dir', type=str, required=True)
parser.add_argument('--histogram-dir', type=str, default=None)
args = parser.parse_args()
device = args.device


def filter(name: str, m: torch.nn.Module):
    if 'mlp.0.weight' not in name:
        return False
    return True

def final(weights):
    weight = torch.stack(weights)
    res = SpectralObservationPlugin.spectral_properties(
        torch.ones([1]).to(device),
        weight.to(device),
        args.threshold_ratio
    )['kkT']

    print(res)

    return res

def reduction(last_result: 'dict[str, dict[str, torch.Tensor]]', name: str, weight):
    if last_result is None: last_result = {}
    res: 'dict[str, torch.Tensor]' = SpectralObservationPlugin.spectral_properties(
        torch.ones([1]).to(device),
        weight.to(device),
        args.threshold_ratio
    )['kkT']
    # last_result[name] = res
    return {
        **last_result,
        name: res,
    }


models = [
    'model_299.pth'
]

def checkpoint_filter(name: str):
    basename = os.path.basename(name)
    return basename in models

results = scan_checkpoints(
    paths=args.path,
    filter=filter,
    reduction=reduction,
    map_location=device,
    checkpoint_filter=checkpoint_filter,
)

results_kkT = {int(p[p.find('model_') + len('model_'):-len('.pth')]): value for p, value in results.items()}



def permute(name):
    def get_depth(module_name):
        ints = re.findall(r'\d+', module_name)
        return int(ints[0])
        
    dict_module_epoch_value = {}
    for p, checkpoint in results_kkT.items():
        for module, value in checkpoint.items():
            if module not in dict_module_epoch_value:
                dict_module_epoch_value[get_depth(module)] = {}
            dict_module_epoch_value[get_depth(module)][p] = value[name]
    return dict_module_epoch_value
results_kkT = {
    name: permute(name) for name in next(iter(next(iter(results_kkT.values())).values()))
}

if args.histogram_dir is not None or True:
    eigenvalues = results_kkT['eigenvalues']
    df = pd.DataFrame.from_dict(
        {
            (epoch, depth) :   eigenvalues
                for depth in eigenvalues.keys()
                for epoch in eigenvalues[depth].keys()
        }
    )
    
    



def plot_curve(data_dict, title="Curve", x_label="X", y_label="Y", save_path=None):
    """
    使用给定的 {x: y} 字典绘制曲线
    
    参数:
        data_dict: {x: y} 格式的数据字典
        title: 图表的标题
        x_label: X轴标签
        y_label: Y轴标签
    """

    # 从字典中提取x和y的数据
    x = list(data_dict.keys())
    y = list(data_dict.values())

    # 使用matplotlib绘制曲线
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


output_dir = args.output_dir
os.makedirs(args.output_dir, exist_ok=True)
for name, d in results_kkT.items():
    plot_curve(
        d,
        title=name,
        x_label="Epoch",
        y_label=name,
        save_path=os.path.join(output_dir, name + '.jpg'),
    )
