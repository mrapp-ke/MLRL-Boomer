(documentation)=

# Generating the Documentation

The documentation of the BOOMER algorithm and other software provided by this project is publicly available at [https://mlrl-boomer.readthedocs.io](https://mlrl-boomer.readthedocs.io/en/latest/). This website should be the primary source of information for everyone who wants to learn about our work. However, if you want to generate the documentation from scratch, e.g., for offline use on your own computer, follow the instructions below.

## Prerequisites

In order to generate the documentation, [Doxygen](https://doxygen.nl) must be installed on the host system beforehand. It is used to generate an API documentation from the C++ source files. In addition, the [Roboto](https://fonts.google.com/specimen/Roboto) font should be available on your system. If this is not the case, another font is used as a fallback.

`````{tip}
It is not necessary to execute the steps below one after the other. Instead, running the following command should suffice to create the entire documentation, including files that describe the C++ and Python API.

````{tab} Linux
   ```text
   ./build doc
   ```
````

````{tab} macOS
   ```text
   ./build doc
   ```
````

````{tab} Windows
   ```text
   build.bat doc
   ```
````
`````

Whenever the documentation was updated or any C++ or Python source files have been modified, the above command must be run again in order to generate an updated version of the documentation that reflects the respective changes.

## Building the C++ API Documentation

By running the following command, the C++ API documentation is generated via Doxygen:

````{tab} Linux
   ```text
   ./build apidoc_cpp
   ```
````

````{tab} macOS
   ```text
   ./build apidoc_cpp
   ```
````

````{tab} Windows
   ```text
   build.bat apidoc_cpp
   ```
````

The resulting HTML files should be located in the directory `doc/developer_guide/api/cpp/`.

## Building the Python API Documentation

Similarly, the following command generates an API documentation from the project's Python code via [sphinx-apidoc](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html):

````{tab} Linux
   ```text
   ./build apidoc_python
   ```
````

````{tab} macOS
   ```text
   ./build apidoc_python
   ```
````

````{tab} Windows
   ```text
   build.bat apidoc_python
   ```
````

The files produced by the above command should be saved to the directory `doc/developer_guide/api/python/`.

```{note}
If you want to generate the API documentation for the C++ and Python code simultaneously, it is possible to use the build target `apidoc` instead of `apidoc_cpp` and `apidoc_python`.
```

## Building the Final Documentation

To generate the final documentation's HTML files via [sphinx](https://www.sphinx-doc.org/en/master/), the following command can be used:

````{tab} Linux
   ```text
   ./build doc
   ```
````

````{tab} macOS
   ```text
   ./build doc
   ```
````

````{tab} Windows
   ```text
   build.bat doc
   ```
````

Afterward, the generated files can be found in the directory `doc/_build/html/`.

## Cleaning up Build Files

Files that have been generated via the above steps can be removed by invoking the respective commands with the command line argument `--clean`. A more detailed description of how to use this command line argument can be found in the section {ref}`compilation`.
