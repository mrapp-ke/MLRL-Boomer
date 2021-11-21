default_target: install
.PHONY: clean_venv clean_cpp clean_cython clean_compile clean_cpp_install clean_cython_install clean_wheel \
        clean_install clean_doc clean compile_cpp compile_cython compile install_cpp install_cython wheel install doc

ACTIVATE_VENV = . venv/bin/activate
DEACTIVATE_VENV = deactivate
MESON_SETUP = meson setup
MESON_COMPILE = meson compile
MESON_INSTALL = meson install
BUILD_WHEEL = python -m build --wheel
INSTALL_WHEEL = pip install --force-reinstall --no-deps
DOXYGEN = doxygen
SPHINX_APIDOC = sphinx-apidoc --tocfile index -f
SPHINX_BUILD = sphinx-build -M html

clean_venv:
	@echo "Removing virtual Python environment..."
	rm -rf venv/

clean_cpp:
	@echo "Removing C++ compilation files..."
	rm -rf cpp/build/

clean_cython:
	@echo "Removing Cython compilation files..."
	rm -rf python/build/

clean_compile: clean_cpp clean_cython

clean_cpp_install:
	rm -f python/subprojects/**/mlrl/**/cython/lib*.so*

clean_cython_install:
	rm -f python/subprojects/**/mlrl/**/cython/*.so

clean_wheel:
	@echo "Removing Python build files..."
	rm -rf python/subprojects/**/build/
	rm -rf python/subprojects/**/dist/
	rm -rf python/subprojects/**/*.egg-info/

clean_install: clean_cpp_install clean_cython_install clean_wheel

clean_doc:
	@echo "Removing documentation..."
	rm -rf doc/_build/
	rm -rf doc/apidoc/
	rm -f doc/python/**/*.rst

clean: clean_doc clean_compile clean_install clean_venv

venv:
	@echo "Creating virtual Python environment..."
	python3 -m venv venv
	${ACTIVATE_VENV} && (\
	   pip install --upgrade pip; \
	   pip install --upgrade setuptools; \
	   pip install -r python/requirements.txt; \
	) && ${DEACTIVATE_VENV}

compile_cpp: venv
	@echo "Compiling C++ code..."
	${ACTIVATE_VENV} && (\
	    cd cpp/ && ${MESON_SETUP} build/; \
	    cd build/ && ${MESON_COMPILE}; \
	) && ${DEACTIVATE_VENV}

compile_cython: venv
	@echo "Compiling Cython code..."
	${ACTIVATE_VENV} && (\
	    cd python/ && ${MESON_SETUP} build/; \
	    cd build/ && ${MESON_COMPILE}; \
	) && ${DEACTIVATE_VENV}

compile: compile_cpp compile_cython

install_cpp: compile_cpp
	@echo "Installing shared libraries into source tree..."
	${ACTIVATE_VENV} && (\
	    cd cpp/build/ && ${MESON_INSTALL}; \
	) && ${DEACTIVATE_VENV}

install_cython: compile_cython
	@echo "Installing extension modules into source tree..."
	${ACTIVATE_VENV} && (\
	    cd python/build/ && ${MESON_INSTALL}; \
	) && ${DEACTIVATE_VENV}

wheel: install_cpp install_cython
	@echo "Building wheel packages..."
	${ACTIVATE_VENV} && (\
	    cd python/subprojects/; \
	    ${BUILD_WHEEL} common/; \
	    ${BUILD_WHEEL} boosting/; \
	    ${BUILD_WHEEL} seco/; \
	    ${BUILD_WHEEL} testbed/; \
	) && ${DEACTIVATE_VENV}

install: wheel
	@echo "Installing wheel packages into virtual environment..."
	${ACTIVATE_VENV} && (\
	    cd python/subprojects/; \
	    ${INSTALL_WHEEL} common/dist/*.whl; \
	    ${INSTALL_WHEEL} boosting/dist/*.whl; \
	    ${INSTALL_WHEEL} seco/dist/*.whl; \
	    ${INSTALL_WHEEL} testbed/dist/*.whl; \
	) && ${DEACTIVATE_VENV}

doc: install
	@echo "Installing dependencies into virtual environment..."
	${ACTIVATE_VENV} && (\
	    pip install -r doc/requirements.txt; \
	) && ${DEACTIVATE_VENV}
	@echo "Generating C++ API documentation via Doxygen..."
	cd doc/ && mkdir -p apidoc/api/cpp/common/ && ${DOXYGEN} Doxyfile_common
	cd doc/ && mkdir -p apidoc/api/cpp/boosting/ && ${DOXYGEN} Doxyfile_boosting
	@echo "Generating Sphinx documentation..."
	${ACTIVATE_VENV} && (\
	    ${SPHINX_APIDOC} -o doc/python/common/ python/subprojects/common/mlrl/ **/cython; \
	    ${SPHINX_BUILD} doc/python/common/ doc/apidoc/api/python/common/; \
	    ${SPHINX_APIDOC} -o doc/python/boosting/ python/subprojects/boosting/mlrl/ **/cython; \
	    LD_LIBRARY_PATH=cpp/build/subprojects/common/ \
	        ${SPHINX_BUILD} doc/python/boosting/ doc/apidoc/api/python/boosting/; \
	    ${SPHINX_APIDOC} -o doc/python/testbed/ python/subprojects/testbed/mlrl/; \
	    ${SPHINX_BUILD} doc/python/testbed/ doc/apidoc/api/python/testbed/; \
	    ${SPHINX_BUILD} doc/ doc/_build/; \
	) && ${DEACTIVATE_VENV}
