# README

This project implements "BOOMER" - an algorithm for learning gradient boosted multi-label classification rules.

## Project structure

```
|-- data                Directory that contains several benchmark data sets
    |-- ...
|-- python              Directory that contains the project's Python code
    |-- boomer          Directory that contains the Python code of the actual library
        |-- ...
    |-- setup.py        Distutil definition of the library for installation via pip
    |-- main.py         Implements a main function that can be used to start an experiment, i.e., to train a model
    |-- plots.py        Implements another main function that can be used to create various plots for an existing model
|-- Makefile            Makefile for compiling the Cython source files and installing a Python virtual environment
|-- README.md           This file
```

## Project setup

This project requires Python 3.7 and uses C extensions for Python using [Cython](https://cython.org) to speed up computation. Prior to running the Python application, the Cython source files (`.pyx` and `.pxd` files) must be compiled (see [here](http://docs.cython.org/en/latest/src/quickstart/build.html) for further details). The project comes with a Makefile that eases the compilation by running
```
make compile
```
This should result in `.c` files, as well as `.so` files (on Linux) or `.pyd` files (on Windows) be placed in the directory `python/boomer/algorithm/`.

**Syntax highlighting in PyCharm:** Unfortunately, the Community Edition of PyCharm does not come with Cython support. To enable proper syntax highlighting of Cython source files, the file `settings.jar` in the project's root directory can be imported via `File -> Import Settings`.
