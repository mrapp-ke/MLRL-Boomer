import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

extensions = [
    Extension(name='*', sources=['**/*.pyx'], define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")])
]

setup(name='boomer',
      version='0.1.0',
      description='BOOMER - An algorithm for learning gradient boosted multi-label classification rules',
      url='https://github.com/mrapp-ke/Boomer',
      author='Michael Rapp',
      author_email='mrapp@ke.tu-darmstadt.de',
      license='MIT',
      packages=['boomer'],
      install_requires=[
          "numpy>=1.18.0",
          "scipy>=1.4.0",
          "Cython>=0.29.0",
          'scikit-learn>=0.22.0',
          'scikit-multilearn>=0.2.0',
          'liac-arff>=2.4.0',
          'requests>=2.23.0',
          'matplotlib>=3.2.0',
          'xgboost>=1.0.0'
      ],
      python_requires='>=3.7',
      ext_modules=cythonize(extensions, language_level='3', annotate=True),
      include_dirs=[numpy.get_include()],
      zip_safe=False)
