default_target: install
.PHONY: clean_venv clean_cpp clean_cython clean_compile clean_cpp_install clean_cython_install clean_wheel \
        clean_install clean_doc clean compile_cpp compile_cython compile install_cpp install_cython wheel install doc

VENV_DIR = venv
CPP_SRC_DIR = cpp
CPP_BUILD_DIR = ${CPP_SRC_DIR}${SEP}build
PYTHON_SRC_DIR = python
PYTHON_BUILD_DIR = ${PYTHON_SRC_DIR}${SEP}build

VENV_CREATE = python3 -m venv ${VENV_DIR}
VENV_ACTIVATE = . ${VENV_DIR}/bin/activate
VENV_DEACTIVATE = deactivate
PIP_INSTALL = python -m pip install --prefer-binary
PIP_UPGRADE = ${PIP_INSTALL} --upgrade
MESON_SETUP = meson setup
MESON_COMPILE = meson compile
MESON_INSTALL = meson install
WHEEL_BUILD = python -m build --wheel
WHEEL_INSTALL = python -m pip install --force-reinstall --no-deps
DOXYGEN = doxygen
SPHINX_APIDOC = sphinx-apidoc --tocfile index -f
SPHINX_BUILD = sphinx-build -M html

clean_venv:
	@echo Removing virtual Python environment...
	rm -rf ${VENV_DIR}

clean_cpp:
	@echo Removing C++ compilation files...
	rm -rf ${CPP_BUILD_DIR}

clean_cython:
	@echo Removing Cython compilation files...
	rm -rf ${PYTHON_BUILD_DIR}

clean_compile: clean_cpp clean_cython

clean_install:
	rm -f ${PYTHON_SRC_DIR}/subprojects/**/mlrl/**/cython/*.so*
	rm -f ${PYTHON_SRC_DIR}/subprojects/**/mlrl/**/cython/*.dylib

clean_wheel:
	@echo Removing Python build files...
	rm -rf ${PYTHON_SRC_DIR}/subprojects/**/build
	rm -rf ${PYTHON_SRC_DIR}/subprojects/**/dist
	rm -rf ${PYTHON_SRC_DIR}/subprojects/**/*.egg-info

clean_doc:
	@echo Removing documentation...
	rm -rf doc/_build
	rm -rf doc/apidoc
	rm -f doc/python/**/*.rst

clean: clean_doc clean_wheel clean_compile clean_install clean_venv

venv:
	@echo Creating virtual Python environment...
	${VENV_CREATE}
	${VENV_ACTIVATE} && (\
	   ${PIP_UPGRADE} pip; \
	   ${PIP_UPGRADE} setuptools; \
	   ${PIP_INSTALL} -r ${PYTHON_SRC_DIR}/requirements.txt; \
	) && ${VENV_DEACTIVATE}

compile_cpp: venv
	@echo Compiling C++ code...
	${VENV_ACTIVATE} && (\
	    ${MESON_SETUP} ${CPP_BUILD_DIR} ${CPP_SRC_DIR}; \
	    ${MESON_COMPILE} -C ${CPP_BUILD_DIR}; \
	) && ${VENV_DEACTIVATE}

compile_cython: venv
	@echo Compiling Cython code...
	${VENV_ACTIVATE} && (\
	    ${MESON_SETUP} ${PYTHON_BUILD_DIR} ${PYTHON_SRC_DIR}; \
	    ${MESON_COMPILE} -C ${PYTHON_BUILD_DIR}; \
	) && ${VENV_DEACTIVATE}

compile: compile_cpp compile_cython

install_cpp: compile_cpp
	@echo Installing shared libraries into source tree...
	${VENV_ACTIVATE} && (\
	    ${MESON_INSTALL} -C ${CPP_BUILD_DIR}; \
	) && ${VENV_DEACTIVATE}

install_cython: compile_cython
	@echo Installing extension modules into source tree...
	${VENV_ACTIVATE} && (\
	    ${MESON_INSTALL} -C ${PYTHON_BUILD_DIR}; \
	) && ${VENV_DEACTIVATE}

wheel: install_cpp install_cython
	@echo Building wheel packages...
	${VENV_ACTIVATE} && (\
	    cd ${PYTHON_SRC_DIR}/subprojects; \
	    ${WHEEL_BUILD} common; \
	    ${WHEEL_BUILD} boosting; \
	    ${WHEEL_BUILD} seco; \
	    ${WHEEL_BUILD} testbed; \
	) && ${VENV_DEACTIVATE}

install: wheel
	@echo Installing wheel packages into virtual environment...
	${VENV_ACTIVATE} && (\
	    cd ${PYTHON_SRC_DIR}/subprojects; \
	    ${WHEEL_INSTALL} common/dist/*.whl; \
	    ${WHEEL_INSTALL} boosting/dist/*.whl; \
	    ${WHEEL_INSTALL} seco/dist/*.whl; \
	    ${WHEEL_INSTALL} testbed/dist/*.whl; \
	) && ${VENV_DEACTIVATE}

doc: install
	@echo Installing documentation dependencies into virtual environment...
	${VENV_ACTIVATE} && (\
	    ${PIP_INSTALL} -r doc/requirements.txt; \
	) && ${VENV_DEACTIVATE}
	@echo Generating C++ API documentation via Doxygen...
	cd doc && mkdir -p apidoc/api/cpp/common && PROJECT_NUMBER="${file < VERSION}" ${DOXYGEN} Doxyfile_common
	cd doc && mkdir -p apidoc/api/cpp/boosting && PROJECT_NUMBER="${file < VERSION}" ${DOXYGEN} Doxyfile_boosting
	@echo Generating Sphinx documentation...
	${VENV_ACTIVATE} && (\
	    ${SPHINX_APIDOC} -o doc/python/common python/subprojects/common/mlrl **/cython; \
	    ${SPHINX_BUILD} doc/python/common doc/apidoc/api/python/common; \
	    ${SPHINX_APIDOC} -o doc/python/boosting python/subprojects/boosting/mlrl **/cython; \
	    ${SPHINX_BUILD} doc/python/boosting doc/apidoc/api/python/boosting; \
	    ${SPHINX_APIDOC} -o doc/python/testbed python/subprojects/testbed/mlrl; \
	    ${SPHINX_BUILD} doc/python/testbed doc/apidoc/api/python/testbed; \
	    ${SPHINX_BUILD} doc/ doc/_build; \
	) && ${VENV_DEACTIVATE}
