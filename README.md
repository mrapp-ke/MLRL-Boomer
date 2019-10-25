# README

This project implements "BOOMER" -- an algorithm for learning gradient boosted multi-label classification rules.

## Project structure

```
|-- data                Directory that contains several benchmark data sets
    |-- ...
|-- python              Directory that contains the project's Python code
    |-- boomer          Directory that contains the Python code of the actual library
        |-- ...
    |-- setup.py        Distutil definition of the library for installation via pip
    |-- main.py         Implements the main function that is executed when starting an experiment
|-- install_venv.sh     Bash script for installing a Python virtual environment including the project's library and all of its dependencies
|-- README.md           This file
```

## Project setup

This project uses C extensions for Python using [Cython](https://cython.org) to speed up computation (see source .pyx files). 

To enable proper syntax highlighting of Cython source files in PyCharm, the file `settings.jar` in the project's root directory should be imported via `File -> Import Settings`.

Prior to running the Python application, the Cython source files must be compiled by running the script `compile_cython.sh`.