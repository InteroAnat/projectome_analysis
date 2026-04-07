# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
ext_modules = [
    Extension(
        'foo',
        sources=['foo.pyx'],
        language='c++',
        extra_compile_args=['-g']
    )
]

setup(
    name='foo',
    ext_modules=cythonize(ext_modules, annotate=True),
    include_dirs=[np.get_include()]
)
