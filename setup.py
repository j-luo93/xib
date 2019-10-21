from distutils.core import setup

import numpy as np
from Cython.Build import cythonize
from setuptools import find_packages, setup

setup(
    name='xib',
    version='0.1',
    packages=find_packages(),
    ext_modules=cythonize('xib/extract_words_impl.pyx', annotate=True),
    include_dirs=[np.get_include()]
)
