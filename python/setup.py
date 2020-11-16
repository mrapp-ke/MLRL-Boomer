import sys

import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

# True, if annotated Cython source files that highlight Python interactions should be created
ANNOTATE = True

# True, if all Cython compiler optimizations should be disabled
DEBUG = False

# The compiler/linker argument to enable OpenMP support
sources = [
    '**/*.pyx',
    'boomer/common/cpp/random.cpp',
    'boomer/common/cpp/data.cpp',
    'boomer/common/cpp/indices.cpp',
    'boomer/common/cpp/input_data.cpp',
    'boomer/common/cpp/predictions.cpp',
    'boomer/common/cpp/rules.cpp',
    'boomer/common/cpp/post_processing.cpp',
    'boomer/common/cpp/rule_evaluation.cpp',
    'boomer/common/cpp/sub_sampling.cpp',
    'boomer/common/cpp/statistics.cpp',
    'boomer/common/cpp/thresholds.cpp',
    'boomer/common/cpp/thresholds_exact.cpp',
    'boomer/common/cpp/thresholds_approximate.cpp',
    'boomer/common/cpp/head_refinement.cpp',
    'boomer/common/cpp/rule_refinement.cpp',
    'boomer/common/cpp/pruning.cpp',
    'boomer/common/cpp/binning.cpp',
    'boomer/boosting/cpp/blas.cpp',
    'boomer/boosting/cpp/lapack.cpp',
    'boomer/boosting/cpp/data.cpp',
    'boomer/boosting/cpp/data_label_wise.cpp',
    'boomer/boosting/cpp/data_example_wise.cpp',
    'boomer/boosting/cpp/losses_label_wise.cpp',
    'boomer/boosting/cpp/losses_example_wise.cpp',
    'boomer/boosting/cpp/statistics_label_wise.cpp',
    'boomer/boosting/cpp/statistics_example_wise.cpp',
    'boomer/boosting/cpp/rule_evaluation_label_wise.cpp',
    'boomer/boosting/cpp/rule_evaluation_example_wise.cpp',
    'boomer/boosting/cpp/post_processing.cpp',
    'boomer/seco/cpp/heuristics.cpp',
    'boomer/seco/cpp/lift_functions.cpp',
    'boomer/seco/cpp/head_refinement.cpp',
    'boomer/seco/cpp/statistics_label_wise.cpp',
    'boomer/seco/cpp/rule_evaluation_label_wise.cpp'
]
COMPILE_FLAG_OPEN_MP = '/openmp' if sys.platform.startswith('win') else '-fopenmp'


extensions = [
    Extension(name='*', sources=sources, language='c++', extra_compile_args=[COMPILE_FLAG_OPEN_MP],
              extra_link_args=[COMPILE_FLAG_OPEN_MP], define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
]

compiler_directives = {
    'boundscheck': DEBUG,
    'wraparound': DEBUG,
    'cdivision': not DEBUG,
    'initializedcheck': DEBUG
}

setup(name='boomer',
      version='0.4.0',
      description='BOOMER - An algorithm for learning gradient boosted multi-label classification rules',
      url='https://github.com/mrapp-ke/Boomer',
      author='Michael Rapp',
      author_email='mrapp@ke.tu-darmstadt.de',
      license='MIT',
      packages=['boomer'],
      install_requires=[
          "numpy>=1.19.0",
          "scipy>=1.5.0",
          "Cython>=0.29.0",
          'scikit-learn>=0.23.0',
          'scikit-multilearn>=0.2.0',
          'liac-arff>=2.4.0',
          'requests>=2.23.0'
      ],
      python_requires='>=3.8',
      ext_modules=cythonize(extensions, language_level='3', annotate=ANNOTATE, compiler_directives=compiler_directives),
      include_dirs=[numpy.get_include()],
      zip_safe=False)
