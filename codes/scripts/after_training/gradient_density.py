# %%
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from tqdm.auto import tqdm

device='cuda:2'

# %%
root = os.path.expanduser('~/sparsity/cache/')
output_dir = os.path.expanduser('~/sparsity/dumps/')

# %%
def merge_splits(dir_to_merge: str):
    splits = []
    for split_name in os.listdir(dir_to_merge):
        split = torch.load(os.path.join(dir_to_merge, split_name), device).to(device)
        splits.append(split)
    return torch.stack(splits).sum(dim=0)

# %%
def get_histc(dir: str, checkpoint: int):
    n_layer = len(os.listdir(os.path.join(dir, 'g')))
    gs, gs_activated = [], []
    for i in tqdm(range(n_layer), desc=str(checkpoint)):
        g_directory = os.path.join(dir, 'g', str(i), str(checkpoint))
        g_activated_directory = os.path.join(dir, 'g_activated', str(i), str(checkpoint))
        
        gs.append(merge_splits(g_directory))
        gs_activated.append(merge_splits(g_activated_directory))

    return gs, gs_activated

# %%
def get_checkpoints(dir: str):
    g_directory = os.path.join(dir, 'g', '0')
    return sorted([int(s) for s in os.listdir(g_directory)])

# %%
n_bin = 1000
color='blue'

def histogram(gs, gs_activated, color, n_bin=1000, save_path=None, bin_range=[-10, 5], display_range=[-6, 3]):
    fig, axs = plt.subplots(len(gs_activated), 1, sharex=True)
    for i, ax in enumerate(axs):
        ax.set_yticks([])
        if i == len(axs) - 1:
            ax.set_xlabel("Log₁₀ squared value of entries in g")
    hist_c_bound = bin_range
    bin_width = (hist_c_bound[1] - hist_c_bound[0]) / (n_bin + 1)
    bin_edges = np.linspace(hist_c_bound[0], hist_c_bound[1], n_bin + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for ax, g, g_activated in tqdm(zip(axs, gs, gs_activated), total=len(gs), desc="histogram"):
        g = g.reshape(n_bin, -1).sum(dim=-1)
        g_activated = g_activated.reshape(n_bin, -1).sum(dim=-1) * (g.sum() / g_activated.sum() / 3)
        ax.bar(bin_centers, g.cpu(), width=bin_width, align='center', alpha=0.3, color=color)
        ax.bar(bin_centers, g_activated.cpu(), width=bin_width, align='center', color=color)
    plt.grid(False)
    plt.xlim(*display_range)
    if save_path:
        fig.savefig(save_path)
    else:
        fig.show()
    plt.close()

# histogram(gs, gs_activated, color='blue')

# %%
def get_means(all: torch.Tensor, hist_c_bound):
    means = []
    for checkpoint, gs in all:
        checkpoint_means = []
        for g in gs:
            assert len(g.shape) == 1
            prob = g / g.sum()
            bin_edges = np.linspace(hist_c_bound[0], hist_c_bound[1], len(g) + 1)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mean = (10 ** torch.tensor(centers, device=device) * prob).sum()
            checkpoint_means.append(mean)
        means.append((checkpoint, torch.stack(checkpoint_means)))

    return means

# %%
def get_ratio(all_gs, all_gs_activated, hist_c_bound):
    ratio = []
    for (checkpoint, gs_means), (checkpoint_, gs_activated_means) in zip(get_means(all_gs, hist_c_bound), get_means(all_gs_activated, hist_c_bound)):
        assert checkpoint == checkpoint_
        ratio.append((checkpoint, gs_activated_means / gs_means))

    res = []
    for l in range(len(ratio[0][1])):
        x = []
        y = []
        for i in range(len(ratio)):
            x.append(ratio[i][0])
            y.append(ratio[i][1][l].item())
        res.append((x, torch.tensor(y)))
    return res

# %%
def plot_multiple_lines_with_smoothing(lines, beta, alpha=0.5, labels=None, colors=None, title=None, xlabel=None, ylabel=None, save=None, y_lim=None, figsize=None, palette='Reds', fontsize=12, y_logscale=False, y_min=1e-3, show_legend=True, y_symlog=False, dont_close=False):

    plt.grid(True)

    if not isinstance(lines[0], list):
        all_lines = [lines]
        all_labels = [labels]
        all_palettes = [palette]
    else:
        all_lines = lines
        all_labels = labels
        all_palettes = palette

    print(all_palettes)
    if colors is not None:
        all_colors = [[colors for j in range(len(all_lines[i]))] for i, c in enumerate(all_colors)]
    else:
        all_colors = []
        for i in range(len(all_lines)):
            cmap = plt.get_cmap(all_palettes[i])
            colors = [cmap(1 - i) for i in np.linspace(0.1, 0.9, len(all_lines[i]))]
            all_colors.append(colors)

    

    lines, labels, colors = [], [], []
    for i in range(len(all_lines)):
        lines.extend(all_lines[i])
        labels.extend(all_labels[i])
        colors.extend(all_colors[i])
    
    if figsize is not None:
        if show_legend:
            figsize = [figsize[0] /4 * 5, figsize[1]]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [4, 1]})
        else:
            fig, ax1 = plt.subplots(figsize=figsize)
    else:
        if show_legend:
            fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]})
        else:
            fig, ax1 = plt.subplots()


    max_value = 0
    legend_lines = []
    for i, line in enumerate(lines):
        label = f'Line {i + 1}' if labels is None else labels[i]
        legend_line, = ax1.plot(line[0], line[1], color=colors[i], alpha=alpha, label=label)
        max_value = max(max_value, float(line[1].max()))
        legend_lines.append(legend_line)


    if y_logscale and max_value < 1:
        ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ticks = [t * 0.1 for t in ticks[:-1]] + ticks
        ax1.set_yticks(ticks)
        ax1.set_yticklabels([str(tick) for tick in ticks], fontsize=fontsize)

    if title:
        ax1.set_title(title, fontsize=fontsize)
    
    if xlabel:
        ax1.set_xlabel(xlabel, fontsize=fontsize)
    
    if ylabel:
        ax1.set_ylabel(ylabel, fontsize=fontsize)
    
    if y_lim is not None:
        if not y_symlog:
            ax1.set_ylim(*y_lim)
        else:
            ax1.set_ylim(top=y_lim[-1])
    elif y_min is not None:
        if not y_symlog:
            ax1.set_ylim(bottom=y_min)
    if y_symlog:
        ax1.set_ylim(bottom=-0.05)

    if y_logscale:
        ax1.set_yscale('log')
    if y_symlog:
        ax1.set_yscale('symlog')

    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1.grid()

    if show_legend:
        # Add legend to ax2 and hide everything else
        ax2.legend(handles=legend_lines, loc='center', fontsize=fontsize*0.75)
        ax2.axis('off')


    plt.tight_layout(pad=1.0)
    if save:
        plt.savefig(save)
    else:
        plt.show()
    if not dont_close:
        plt.close()

# %%
for task in ["imagenet1k", "T5"]:
    for model_type in ["sparsified", "vanilla"]:
        dir = os.path.join(root, task, 'gradient_density', model_type, "pth")
        checkpoints = get_checkpoints(dir)
        output_directory = os.path.join(output_dir, task, 'gradient_density', model_type)
        bin_range=[-80, 0] if task == "T5" else [-20, 10]
        display_range = [-80, -12] if task == "T5" else [-12, 6]
        all_gs, all_gs_activated = [], []
        for checkpoint in tqdm(sorted(checkpoints), f"{task}-{model_type}"):
            gs, gs_activated = get_histc(dir, checkpoint)
            save_path = os.path.join(output_directory, str(checkpoint)+'.jpg')
            os.makedirs(output_directory, exist_ok=True)
            histogram(gs, gs_activated, 'red' if model_type == 'sparsified' else 'blue', save_path=save_path, bin_range=bin_range, display_range=display_range)
            all_gs.append((checkpoint, gs))
            all_gs_activated.append((checkpoint, gs_activated))
        ratios = get_ratio(all_gs, all_gs_activated, bin_range)
        save_path = os.path.join(output_directory, 'ratio' + '.jpg')
        print(save_path)
        plot_multiple_lines_with_smoothing(
            lines=ratios,
            beta=1.0, alpha=1.0,
            labels=[f'Layer {i+1}' for i in range(len(ratios))],
            xlabel='Epochs' if task == "imagenet1k" else "Steps",
            ylabel='Log₁₀ E[g² | α>0] / E[g²]',
            palette='autumn' if model_type == "sparsified" else 'winter',
            y_logscale=True,
            y_min=10**(-1.5),
            save=save_path,
        )
        


