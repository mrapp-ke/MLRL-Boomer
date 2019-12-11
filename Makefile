default_target: compile
.PHONY: clean_venv clean_cython clean install

clean: clean_cython clean_venv

clean_cython:
	@echo "Removing compiled C/C++ files..."
	find python/ -type f -name "*.so" -delete
	find python/ -type f -name "p*.c" -delete
	find python/ -type f -name "*.pyd" -delete
	find python/ -type f -name "*.cpp" -delete
	find python/ -type f -name "*.html" -delete

clean_venv:
	@echo "Removing virtual Python environment..."
	rm -rf venv/

compile:
	@echo "Compiling Cython code..."
	cd python/ && python setup.py build_ext --inplace

install: clean_venv
	@echo "Creating virtual Python environment..."
	python3.7 -m venv venv && venv/bin/pip3.7 install python/
