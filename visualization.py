import torch
import torch.functional as F
import matplotlib.pyplot as plt
import numpy as np

from codes.modules.activations import SReLU, WeirdLeakyReLU, Shift, SquaredReLU, SShaped

def save_activation_function_plot(activation_fn, save_path, x_range=[-2, 2]):
    x = torch.linspace(*x_range, 100, requires_grad=True)
    y = activation_fn(x)

    plt.plot(x.detach().numpy(), y.detach().numpy())
    plt.xlabel('x')
    plt.ylabel('Activation')

    plt.axhline(0, color='black', linewidth=0.5)  # 绘制y=0参考线
    plt.axvline(0, color='black', linewidth=0.5)  # 绘制x=0参考线

    x_grad = torch.autograd.grad(y, x, torch.ones_like(x), create_graph=True)[0]
    plt.plot(x.detach().numpy(), x_grad.detach().numpy(), color='red', linestyle='dashed')

    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


leaky_relu = torch.nn.LeakyReLU(0.1)
srelu = SReLU(1)
weird_leaky_relu = WeirdLeakyReLU(0.1, 1)

save_activation_function_plot(torch.relu, 'pic/relu.jpg')
save_activation_function_plot(leaky_relu, 'pic/leaky_relu.jpg')
save_activation_function_plot(srelu, 'pic/srelu.jpg')
save_activation_function_plot(weird_leaky_relu, 'pic/wired_leaky_relu.jpg')
weird = Shift(SReLU(1.5), -1.6, +1.6)
save_activation_function_plot(weird, 'pic/wired.jpg', [-4, 4])

weird_relu2 = Shift(SShaped(SquaredReLU(), 1.5), +1.6, +1.6)
print(weird_relu2.get_habitat())
save_activation_function_plot(weird_relu2, 'pic/wired_relu2.jpg', [-4, 4])
