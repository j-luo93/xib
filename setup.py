from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize
from setuptools import find_packages, setup

ext_modules = [
    Extension(
        'xib.extract_words_impl',
        ['xib/extract_words_impl.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        'xib.check_in_vocab_impl',
        ['xib/check_in_vocab_impl.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]
setup(
    name='xib',
    version='0.1',
    packages=find_packages(),
    zip_safe=False,
    ext_modules=cythonize(
        ext_modules,
        annotate=True),  # DEBUG(j_luo) remove this
    include_dirs=[np.get_include()]
)
