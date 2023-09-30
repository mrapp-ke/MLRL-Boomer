.. _standards:

Coding Standards
================

As it is common for Open Source projects, where everyone is invited to contribute, we rely on coding standards to ensure that new code works as expected, does not break existing functionality, and adheres to best practices we agreed on. These coding standards are described in the following.

.. _testing:

Testing the Code
----------------

To be able to detect problems with the project's source code early during development, it comes with a large number of integration tests. Each of these tests runs a different configuration of the project's algorithms via the command line API and checks for unexpected results. If you want to execute the integrations tests on your own system, you can use the following command:

.. tab:: Linux

   .. code-block:: text

      ./build tests

.. tab:: MacOS

   .. code-block:: text

      ./build tests

.. tab:: Windows

   .. code-block:: text

      build.bat tests


The integration tests are also run automatically on a `CI server <https://en.wikipedia.org/wiki/Continuous_integration>`__ whenever relevant parts of the source code have been modified. For this purpose, we rely on the infrastructure provided by `Github Actions <https://docs.github.com/en/actions>`__. A track record of past test runs can be found in the `Github repository <https://github.com/mrapp-ke/Boomer/actions>`__.

.. _codestyle:

Code Style
----------

We aim to enforce a consistent code style across the entire project. For formatting the C++ code, we employ `clang-format <https://clang.llvm.org/docs/ClangFormat.html>`__. The desired C++ code style is defined in the file ``.clang-format`` in project's root directory. Accordingly, we use `YAPF <https://github.com/google/yapf>`__ to enforce the Python code style defined in the file ``.style.yapf``. In addition, `isort <https://github.com/PyCQA/isort>`__ is used to keep the ordering of imports in Python and Cython source files consistent according to the configuration file ``.isort.cfg`` and `pylint <https://pylint.org/>`__ is used to check for common issues in the Python code according to the configuration file ``.pylintrc``. If you have modified the project's source code, you can check whether it adheres to our coding standards via the following command:

.. tab:: Linux

   .. code-block:: text

      ./build test_format

.. tab:: MacOS

   .. code-block:: text

      ./build test_format

.. tab:: Windows

   .. code-block:: text

      build.bat test_format

.. note::
    If you want to check for compliance with the C++ or Python code style independently, you can use the build target ``test_format_cpp`` or ``test_format_python`` instead of ``test_format``.

In order to automatically format the project's source files according to our style guidelines, the following command can be used:

.. tab:: Linux

   .. code-block:: text

      ./build format

.. tab:: MacOS

   .. code-block:: text

      ./build format

.. tab:: Windows

   .. code-block:: text

      build.bat format

.. note::
    If you want to format only the C++ source files, you can specify the build target ``format_cpp`` instead of ``format``. Accordingly, the target ``format_python`` may be used to format only the Python source files.

Whenever any source files have been modified, a `Github Action <https://docs.github.com/en/actions>`__ is run automatically to verify if they adhere to our code style guidelines. The result of these runs can be found in the `Github repository <https://github.com/mrapp-ke/Boomer/actions>`__.
