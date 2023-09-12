#include <torch/extension.h>

torch::Tensor jsrelu(torch::Tensor x) {
    auto y = x + 1;
    return y.square_().add_(-1).div_(2).mul_(x >= 0);
}

torch::Tensor jsrelu_derivative(torch::Tensor x) {
    return (x + 1).mul_(x >= 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &jsrelu, "J-SquaredReLU forward");
    m.def("derivative", &jsrelu_derivative, "J-SquaredReLU derivative");
}
