#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)
"""
import setuptools

setuptools.setup(
    name='boomer',
    version='0.7.0',
    description='BOOMER - An algorithm for learning gradient boosted multi-label classification rules',
    url='https://github.com/mrapp-ke/Boomer',
    author='Michael Rapp',
    author_email='michael.rapp.ml@gmail.com',
    license='MIT',
    packages=['mlrl.common', 'mlrl.boosting', 'mlrl.seco', 'mlrl.testbed'],
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'liac-arff>=2.5.0'
    ],
    python_requires='>=3.7')
