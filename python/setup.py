from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(name='boomer',
      version='0.1.0',
      description='BOOMER - An algorithm for learning gradient boosted multi-label classification rules',
      url='https://github.com/mrapp-ke/Boomer',
      author='Michael Rapp',
      author_email='mrapp@ke.tu-darmstadt.de',
      license='MIT',
      packages=['boomer'],
      install_requires=[
          'liac-arff',
          'numpy',
          'scikit-learn',
          'scipy',
          'sklearn',
          'requests',
          'matplotlib',
          'Cython'
      ],
      python_requires='>=3.7',
      ext_modules=cythonize('**/*.pyx', language_level='3', annotate=True),
      include_dirs=[numpy.get_include()],
      zip_safe=False)
