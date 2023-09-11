from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='activations_ext',
    ext_modules=[
        cpp_extension.CppExtension('activations_ext', ['jsrelu.cpp']),
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)

