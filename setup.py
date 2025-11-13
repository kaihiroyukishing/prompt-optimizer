"""
Setup script for building C++ extension modules with pybind11.
"""

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        'normalization',
        [
            'cpp/src/normalization.cpp',
            'cpp/bindings/normalization_bindings.cpp'
        ],
        include_dirs=['cpp/include'],
        cxx_std=17,
        language='c++'
    ),
    Pybind11Extension(
        'cache_serialization',
        [
            'cpp/src/cache_serialization.cpp',
            'cpp/bindings/cache_serialization_bindings.cpp'
        ],
        include_dirs=['cpp/include'],
        cxx_std=17,
        language='c++'
    ),
]

setup(
    name='prompt-optimizer',
    version='1.0.0',
    description='Context-aware AI prompt optimization system with C++ performance modules',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    python_requires='>=3.8',
)

