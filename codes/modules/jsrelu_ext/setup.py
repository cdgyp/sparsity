from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='jsrelu_ext',
    ext_modules=[
        cpp_extension.CppExtension('jsrelu_ext', ['jsrelu.cpp']),
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    version="0.0"
)

