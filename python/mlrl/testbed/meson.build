project('testbed')

# Directory into which Python source files should be installed
lib_dir = subproject('common').get_variable('lib_dir')
package_dir = 'mlrl' / meson.project_name()
install_dir = lib_dir / package_dir

# Python source files
source_files = [
    'bbc_cv.py',
    'data.py',
    'data_characteristics.py',
    'evaluation.py',
    'experiments.py',
    'interfaces.py',
    'io.py',
    'model_characteristics.py',
    'parameters.py',
    'persistence.py',
    'training.py'
]

install_data(source_files, install_dir : install_dir)
