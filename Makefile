default_target: install
.PHONY: clean_venv clean_cpp clean_cython clean_compile clean_cpp_install clean_cython_install clean_wheel \
        clean_install clean_doc clean test_format_cpp test_format_python test_format format_cpp format_python format \
        compile_cpp compile_cython compile install_cpp install_cython wheel install apidoc_cpp apidoc_python doc

UNAME = $(if $(filter Windows_NT,${OS}),Windows,$(shell uname))
IS_WIN = $(filter Windows,${UNAME})

VENV_DIR = venv
CPP_SRC_DIR = cpp
CPP_BUILD_DIR = ${CPP_SRC_DIR}/build
CPP_PACKAGE_DIR = ${CPP_SRC_DIR}/subprojects
PYTHON_SRC_DIR = python
PYTHON_BUILD_DIR = ${PYTHON_SRC_DIR}/build
PYTHON_PACKAGE_DIR = ${PYTHON_SRC_DIR}/subprojects
DIST_DIR = dist
DOC_DIR = doc
DOC_API_DIR = ${DOC_DIR}/apidoc
DOC_TMP_DIR = ${DOC_DIR}/python
DOC_BUILD_DIR = ${DOC_DIR}/_build

PS = powershell -Command
PYTHON = $(if ${IS_WIN},python,python3)
VENV_CREATE = ${PYTHON} -m venv ${VENV_DIR}
VENV_ACTIVATE = $(if ${IS_WIN},${PS} "${VENV_DIR}/Scripts/activate.bat;",. ${VENV_DIR}/bin/activate)
VENV_DEACTIVATE = $(if ${IS_WIN},${PS} "${VENV_DIR}/Scripts/deactivate.bat;",deactivate)
PIP_INSTALL = python -m pip install --prefer-binary
YAPF = yapf -r -p --style=.style.yapf --verbose
YAPF_DRYRUN = ${YAPF} --diff
YAPF_INPLACE = ${YAPF} -i
CLANG_FORMAT = clang-format --style=file --verbose
CLANG_FORMAT_DRYRUN = ${CLANG_FORMAT} -n --Werror
CLANG_FORMAT_INPLACE = ${CLANG_FORMAT} -i
MESON_SETUP = meson setup
MESON_COMPILE = meson compile
MESON_INSTALL = meson install
WHEEL_BUILD = python -m build --wheel
WHEEL_INSTALL = python -m pip install --force-reinstall --no-deps
PYTHON_UNITTEST = python -m unittest discover -v -f -s
DOXYGEN = $(if ${IS_WIN},for /f %%i in (./../VERSION) do set PROJECT_NUMBER=%%i && doxygen,PROJECT_NUMBER=${file < VERSION} doxygen)
SPHINX_APIDOC = sphinx-apidoc --tocfile index -f
SPHINX_BUILD = sphinx-build -M html

define delete_dir
	$(if ${IS_WIN},\
	${PS} "if (Test-Path ${1}) {rm ${1} -Recurse -Force}",\
	rm -rf ${1})
endef

define delete_files_recursively
	$(if ${IS_WIN},\
	${PS} "rm ${1} -Recurse -Force -Include ${2}",\
	rm -f ${1}/**/${2})
endef

define delete_dirs_recursively
	$(if ${IS_WIN},\
	${PS} "rm ${1} -Recurse -Force -Include ${2}",\
	rm -rf ${1}/**/${2})
endef

define install_wheels
	$(if ${IS_WIN},\
	${PS} "${WHEEL_INSTALL} (Get-ChildItem -Path ${1} | Where Name -Match '\.whl' | Select-Object -ExpandProperty FullName);",\
	${WHEEL_INSTALL} ${1}/*.whl)
endef

define create_dir
	$(if ${IS_WIN},\
	${PS} "New-Item -Path ${1} -ItemType "directory" -Force",\
	mkdir -p ${1})
endef

define clang_format_dryrun_recursively
	$(if ${IS_WIN},\
	(${PS} "Get-ChildItem -Path ${1} -Recurse | Where Name -Match '\.hpp|\.cpp' | Select-Object -ExpandProperty FullName | Out-File .cpp_files.tmp -Encoding utf8";\
	    ${CLANG_FORMAT_DRYRUN} --files=.cpp_files.tmp;\
	    ${PS} "rm .cpp_files.tmp -Force"),\
	find ${1} -type f \( -iname "*.hpp" -o -iname "*.cpp" \) -exec ${CLANG_FORMAT_DRYRUN} {} +)
endef

define clang_format_inplace_recursively
	$(if ${IS_WIN},\
	(${PS} "Get-ChildItem -Path ${1} -Recurse | Where Name -Match '\.hpp|\.cpp' | Select-Object -ExpandProperty FullName | Out-File .cpp_files.tmp -Encoding utf8";\
	    ${CLANG_FORMAT_INPLACE} --files=.cpp_files.tmp;\
	    ${PS} "rm .cpp_files.tmp -Force"),\
	find ${1} -type f \( -iname "*.hpp" -o -iname "*.cpp" \) -exec ${CLANG_FORMAT_INPLACE} {} +)
endef

clean_venv:
	@echo Removing virtual Python environment...
	$(call delete_dir,${VENV_DIR})

clean_cpp:
	@echo Removing C++ compilation files...
	$(call delete_dir,${CPP_BUILD_DIR})

clean_cython:
	@echo Removing Cython compilation files...
	$(call delete_dir,${PYTHON_BUILD_DIR})

clean_compile: clean_cpp clean_cython

clean_install:
	@echo Removing shared libraries and extension modules from source tree...
	$(call delete_files_recursively,${PYTHON_PACKAGE_DIR},*.so*)
	$(call delete_files_recursively,${PYTHON_PACKAGE_DIR},*.dylib)
	$(call delete_files_recursively,${PYTHON_PACKAGE_DIR},*.dll)
	$(call delete_files_recursively,${PYTHON_PACKAGE_DIR},*.lib)
	$(call delete_files_recursively,${PYTHON_PACKAGE_DIR},*.pyd)

clean_wheel:
	@echo Removing Python build files...
	$(call delete_dirs_recursively,${PYTHON_PACKAGE_DIR},build)
	$(call delete_dirs_recursively,${PYTHON_PACKAGE_DIR},${DIST_DIR})
	$(call delete_dirs_recursively,${PYTHON_PACKAGE_DIR},*.egg-info)

clean_doc:
	@echo Removing documentation...
	$(call delete_dir,${DOC_BUILD_DIR})
	$(call delete_dir,${DOC_API_DIR})
	$(call delete_files_recursively,${DOC_TMP_DIR},*.rst)

clean: clean_doc clean_wheel clean_compile clean_install clean_venv

venv:
	@echo Creating virtual Python environment...
	${VENV_CREATE}
	${VENV_ACTIVATE} \
	    && ${PIP_INSTALL} -r ${PYTHON_SRC_DIR}/requirements.txt \
	    && ${VENV_DEACTIVATE}

test_format_python: venv
	@echo Checking Python code style...
	${VENV_ACTIVATE} \
	    && ${YAPF_DRYRUN} ${PYTHON_PACKAGE_DIR} \
	    && ${VENV_DEACTIVATE}

test_format_cpp: venv
	@echo Checking C++ code style...
	${VENV_ACTIVATE} \
	    && $(call clang_format_dryrun_recursively,${CPP_PACKAGE_DIR}) \
	    && ${VENV_DEACTIVATE}

test_format: test_format_python test_format_cpp

format_python: venv
	@echo Formatting Python code...
	${VENV_ACTIVATE} \
	    && ${YAPF_INPLACE} ${PYTHON_PACKAGE_DIR} \
	    && ${VENV_DEACTIVATE}

format_cpp: venv
	@echo Formatting C++ code...
	${VENV_ACTIVATE} \
	    && $(call clang_format_inplace_recursively,${CPP_PACKAGE_DIR}) \
	    && ${VENV_DEACTIVATE}

format: format_python format_cpp

compile_cpp: venv
	@echo Compiling C++ code...
	${VENV_ACTIVATE} \
	    && ${MESON_SETUP} ${CPP_BUILD_DIR} ${CPP_SRC_DIR} \
	    && ${MESON_COMPILE} -C ${CPP_BUILD_DIR} \
	    && ${VENV_DEACTIVATE}

compile_cython: venv
	@echo Compiling Cython code...
	${VENV_ACTIVATE} \
	    && ${MESON_SETUP} ${PYTHON_BUILD_DIR} ${PYTHON_SRC_DIR} \
	    && ${MESON_COMPILE} -C ${PYTHON_BUILD_DIR} \
	    && ${VENV_DEACTIVATE}

compile: compile_cpp compile_cython

install_cpp: compile_cpp
	@echo Installing shared libraries into source tree...
	${VENV_ACTIVATE} \
	    && ${MESON_INSTALL} -C ${CPP_BUILD_DIR} \
	    && ${VENV_DEACTIVATE}

install_cython: compile_cython
	@echo Installing extension modules into source tree...
	${VENV_ACTIVATE} \
	    && ${MESON_INSTALL} -C ${PYTHON_BUILD_DIR} \
	    && ${VENV_DEACTIVATE}

wheel: install_cpp install_cython
	@echo Building wheel packages...
	${VENV_ACTIVATE} \
	    && ${WHEEL_BUILD} ${PYTHON_PACKAGE_DIR}/common \
	    && ${WHEEL_BUILD} ${PYTHON_PACKAGE_DIR}/boosting \
	    && ${WHEEL_BUILD} ${PYTHON_PACKAGE_DIR}/seco \
	    && ${WHEEL_BUILD} ${PYTHON_PACKAGE_DIR}/testbed \
	    && ${VENV_DEACTIVATE}

install: wheel
	@echo Installing wheel packages into virtual environment...
	${VENV_ACTIVATE} \
	    && $(call install_wheels,${PYTHON_PACKAGE_DIR}/common/${DIST_DIR}) \
	    && $(call install_wheels,${PYTHON_PACKAGE_DIR}/boosting/${DIST_DIR}) \
	    && $(call install_wheels,${PYTHON_PACKAGE_DIR}/seco/${DIST_DIR}) \
	    && $(call install_wheels,${PYTHON_PACKAGE_DIR}/testbed/${DIST_DIR}) \
	    && ${VENV_DEACTIVATE}

tests: install
	@echo Running integration tests...
	${VENV_ACTIVATE} \
	    && ${PYTHON_UNITTEST} ${PYTHON_PACKAGE_DIR}/testbed/tests \
	    && ${VENV_DEACTIVATE}

apidoc_cpp:
	@echo Generating C++ API documentation via Doxygen...
	$(call create_dir,${DOC_API_DIR}/api/cpp/common)
	cd ${DOC_DIR} && ${DOXYGEN} Doxyfile_common
	$(call create_dir,${DOC_API_DIR}/api/cpp/boosting)
	cd ${DOC_DIR} && ${DOXYGEN} Doxyfile_boosting

apidoc_python: install
	@echo Installing documentation dependencies into virtual environment...
	${VENV_ACTIVATE} \
	    && ${PIP_INSTALL} -r ${DOC_DIR}/requirements.txt \
	    && ${VENV_DEACTIVATE}
	@echo Generating Python API documentation via Sphinx-Apidoc...
	${VENV_ACTIVATE} \
	    && ${SPHINX_APIDOC} -o ${DOC_TMP_DIR}/common ${PYTHON_PACKAGE_DIR}/common/mlrl **/cython \
	    && ${SPHINX_BUILD} ${DOC_TMP_DIR}/common ${DOC_API_DIR}/api/python/common \
	    && ${SPHINX_APIDOC} -o ${DOC_TMP_DIR}/boosting ${PYTHON_PACKAGE_DIR}/boosting/mlrl **/cython \
	    && ${SPHINX_BUILD} ${DOC_TMP_DIR}/boosting ${DOC_API_DIR}/api/python/boosting \
	    && ${SPHINX_APIDOC} -o ${DOC_TMP_DIR}/testbed ${PYTHON_PACKAGE_DIR}/testbed/mlrl \
	    && ${SPHINX_BUILD} ${DOC_TMP_DIR}/testbed ${DOC_API_DIR}/api/python/testbed \
	    && ${VENV_DEACTIVATE}

doc: apidoc_cpp apidoc_python
	@echo Generating Sphinx documentation...
	${VENV_ACTIVATE} \
	    && ${SPHINX_BUILD} ${DOC_DIR} ${DOC_BUILD_DIR} \
	    && ${VENV_DEACTIVATE}
