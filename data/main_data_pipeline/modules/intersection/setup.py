from distutils.extension import Extension

# from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy
from Cython.Build import cythonize
from setuptools import setup, find_packages

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'triangle_hash',
    sources=[
        'triangle_hash.pyx'
    ],
    include_dirs=[numpy.get_include()],
    libraries=['m']  # Unix-like specific
)

# Gather all extension modules
ext_modules = [triangle_hash_module]

setup(
    name="fast_intersection",
    version="0.0.1",
    author="Steve",
    ext_modules=cythonize(ext_modules),
    zip_safe=False,
    packages=find_packages(),
    py_modules=['fast_intersection'],
    install_requires=[
        'numpy',
        'trimesh'
    ]
)
