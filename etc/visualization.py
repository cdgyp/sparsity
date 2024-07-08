import torch
import torch.functional as F
import matplotlib.pyplot as plt
import numpy as np

from codes.modules.activations import SReLU, WeirdLeakyReLU, Shift, SquaredReLU, SShaped, JumpingSquaredReLU

def save_activation_function_plot(activation_fn, save_path, x_range=[-2, 2], y_range=None, fontsize=58):
    x = torch.linspace(*x_range, 10000, requires_grad=True)
    y = activation_fn(x)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.plot(x.detach().numpy(), y.detach().numpy(), linewidth=4)
    # ax.set_xlabel('Input', fontsize=fontsize)
    # ax.set_ylabel('Activation or Derivative', fontsize=fontsize)

    ax.set_xlim(*x_range)

    ax.axhline(0, color='black', linewidth=1.5)  # 绘制y=0参考线
    ax.axvline(0, color='black', linewidth=1.5)  # 绘制x=0参考线

    x_grad = torch.autograd.grad(y, x, torch.ones_like(x), create_graph=True)[0]
    ax.plot(x.detach().numpy(), x_grad.detach().numpy(), color='red', linestyle='dashed', linewidth=4)

    if y_range is not None:
        ax.set_ylim(*y_range)
    else:
        ax.set_ylim(-4,+4)

    ax.set_aspect('equal')

    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.savefig(save_path)
    plt.close()


# leaky_relu = torch.nn.LeakyReLU(0.1)
# srelu = SReLU(1)
# weird_leaky_relu = WeirdLeakyReLU(0.1, 1)

save_activation_function_plot(torch.relu, 'pic/relu.jpg', [-2, 2], [-1, 2])
# save_activation_function_plot(leaky_relu, 'pic/leaky_relu.jpg')
save_activation_function_plot(SquaredReLU(), 'pic/squared_relu.jpg', [-2, 2], [-1, 2])
# save_activation_function_plot(weird_leaky_relu, 'pic/wired_leaky_relu.jpg')
# weird = Shift(SReLU(1.5), -1.6, +1.6)
save_activation_function_plot(JumpingSquaredReLU(), 'pic/jsrelu.jpg', [-2, 2], [-1, 2])

for dx in [-1.6,+1.6]:
    for dy in [-1.6, +1.6]:
        weird_relu2 = Shift(SShaped(SquaredReLU(), 1.5), dx, +dy)
        weird_jrelu2 = Shift(SShaped(JumpingSquaredReLU(), 1.5), dx, +dy)
        print(weird_relu2.get_habitat())
        save_activation_function_plot(weird_relu2, f'pic/wired_relu2_{dx}_{dy}.jpg', [-5, 5])
        save_activation_function_plot(weird_jrelu2, f'pic/wired_jrelu2_{dx}_{dy}.jpg', [-5, 5])
        # save_activation_function_plot(JumpingSquaredReLU(), 'pic/jumping_relu2_{dx}_{dy}.jpg', [-4, 4])
