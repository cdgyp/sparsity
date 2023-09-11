#include <torch/extension.h>

torch::Tensor jsrelu(torch::Tensor x) {
    auto y = x + 1;
    return y.square_().div_(2).add_(-1).mul_(x >= 0);
}

torch::Tensor jsrelu(torch::Tensor x) {
    return (x + 1).mul_(x >= 0)
}