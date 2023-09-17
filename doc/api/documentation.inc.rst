.. _documentation:

Generating the Documentation
----------------------------

**Prerequisites**

In order to generate the documentation (this document), `Doxygen <https://sourceforge.net/projects/doxygen/>`__ must be installed on the host system beforehand. It is used to generate an API documentation from the C++ source files. In addition, the `Roboto <https://fonts.google.com/specimen/Roboto>`__ font should be available on your system. If this is not the case, another font will be used as a fallback.

**Step 1: Generating the C++ API documentation**

By running the following command, the C++ API documentation is generated via Doxygen:

.. tab:: Linux

   .. code-block:: text

      ./build apidoc_cpp

.. tab:: MacOS

   .. code-block:: text

      ./build apidoc_cpp

.. tab:: Windows

   .. code-block:: text

      build.bat apidoc_cpp

The resulting HTML files should be located in the directory `doc/apidoc/api/cpp/`.

**Step 2: Generating the Python API documentation**

Similarly, the following command generates an API documentation from the project's Python code via `sphinx-apidoc <https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html>`__:

.. tab:: Linux

   .. code-block:: text

      ./build apidoc_python

.. tab:: MacOS

   .. code-block:: text

      ./build apidoc_python

.. tab:: Windows

   .. code-block:: text

      build.bat apidoc_python

.. note::
    If you want to generate the API documentation for the C++ and Python code simulatenously, it is possible to use the build target ``apidoc`` instead of ``apidoc_cpp`` and ``apidoc_python``.

**Step 3: Generating the final documentation**

To generate the final documentation's HTML files via `sphinx <https://www.sphinx-doc.org/en/master/>`__, the following command can be used:

.. tab:: Linux

   .. code-block:: text

      ./build doc

.. tab:: MacOS

   .. code-block:: text

      ./build doc

.. tab:: Windows

   .. code-block:: text

      build.bat doc

Afterwards, the generated files can be found in the directory `doc/build_/html/`.

It should further be noted that it is not necessary to run the above steps one after the other. Executing a single command with the build target ``doc`` should suffice to create the entire documentation, including files that describe the C++ and Python API.

Files that have been generated via the above steps can be removed by invoking the respective commands with the command line argument ``--clean``. A more detailed description can be found under :ref:`compilation`.
