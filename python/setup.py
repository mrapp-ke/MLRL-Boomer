import sys

import numpy
import setuptools
from Cython.Build import cythonize

# True, if annotated Cython source files that highlight Python interactions should be created
ANNOTATE = False

# True, if all Cython compiler optimizations should be disabled
DEBUG = False

# The compiler/linker argument to enable OpenMP support
COMPILE_FLAG_OPEN_MP = '/openmp' if sys.platform.startswith('win') else '-fopenmp'

sources = [
    '**/*.pyx'
]

extensions = [
    setuptools.Extension(name='*', sources=sources, language='c++', extra_compile_args=[COMPILE_FLAG_OPEN_MP],
                         extra_link_args=[COMPILE_FLAG_OPEN_MP],
                         define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
]

compiler_directives = {
    'boundscheck': DEBUG,
    'wraparound': DEBUG,
    'cdivision': not DEBUG,
    'initializedcheck': DEBUG
}

setuptools.setup(
    name='boomer',
    version='0.4.0',
    description='BOOMER - An algorithm for learning gradient boosted multi-label classification rules',
    url='https://github.com/mrapp-ke/Boomer',
    author='Michael Rapp',
    author_email='mrapp@ke.tu-darmstadt.de',
    license='MIT',
    packages=['boomer'],
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.5.0',
        'Cython>=0.29.0',
        'scikit-learn>=0.23.0',
        'liac-arff>=2.5.0',
        'requests>=2.25.0'
    ],
    python_requires='>=3.7',
    ext_modules=cythonize(extensions, language_level='3', annotate=ANNOTATE, compiler_directives=compiler_directives),
    include_dirs=[numpy.get_include()])
