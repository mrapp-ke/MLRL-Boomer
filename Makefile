default_target: compile
.PHONY: clean_venv clean_cpp clean_cython clean_compile clean_doc clean install doc

ACTIVATE_VENV = . venv/bin/activate
DEACTIVATE_VENV = deactivate

clean_venv:
	@echo "Removing virtual Python environment..."
	rm -rf venv/

clean_cpp:
	@echo "Removing C++ compilation files..."
	rm -rf cpp/build/

clean_cython:
	@echo "Removing Cython compilation files..."
	rm -rf python/build/
	find python/ -type f -name "*.so" -delete

clean_compile: clean_cpp clean_cython

clean_doc:
	@echo "Removing documentation..."
	rm -rf doc/_build/
	rm -rf doc/doxygen/
	rm -rf doc/python_apidoc/
	rm -f doc/python/*.rst

clean: clean_doc clean_compile clean_venv

venv:
	@echo "Creating virtual Python environment..."
	python3 -m venv venv
	${ACTIVATE_VENV} && (\
	   python -m pip install --upgrade pip; \
	   pip install -r python/requirements.txt; \
	) && ${DEACTIVATE_VENV}

compile: venv
	@echo "Compiling C++ code..."
	${ACTIVATE_VENV} && (\
	    cd cpp/ && meson setup build/; \
	    cd build/ && meson compile; \
	) && ${DEACTIVATE_VENV}
	@echo "Compiling Cython code..."
	${ACTIVATE_VENV} && (\
	    cd python/ && meson setup build/; \
	    cd build/ && meson compile && meson install; \
	) && ${DEACTIVATE_VENV}

doc: compile
	@echo "Installing dependencies into virtual environment..."
	${ACTIVATE_VENV} && (\
	    pip install -r doc/requirements.txt; \
	) && ${DEACTIVATE_VENV}
	@echo "Generating C++ API documentation via Doxygen..."
	cd doc/ && mkdir -p doxygen/api/cpp/ && doxygen Doxyfile
	@echo "Generating Sphinx documentation..."
	${ACTIVATE_VENV} && (\
	    sphinx-apidoc --tocfile index -f -o doc/python python/mlrl **/seco **/cython; \
	    cd doc/python/ && LD_PRELOAD="../../cpp/build/mlrl/common/libmlrlcommon.so \
	        ../../cpp/build/mlrl/boosting/libmlrlboosting.so" sphinx-build -M html . ../python_apidoc/api/python; \
	    cd ../ && sphinx-build -M html . _build; \
	) && ${DEACTIVATE_VENV}
