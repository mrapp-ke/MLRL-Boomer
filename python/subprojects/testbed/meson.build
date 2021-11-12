project('testbed')

# Directory into which Python source files should be installed
lib_dir = subproject('common').get_variable('lib_dir')
package_dir = 'mlrl' / meson.project_name()
install_dir = lib_dir / package_dir

# Python source files
source_files = [
    package_dir / 'args.py',
    package_dir / 'bbc_cv.py',
    package_dir / 'data.py',
    package_dir / 'data_characteristics.py',
    package_dir / 'evaluation.py',
    package_dir / 'experiments.py',
    package_dir / 'interfaces.py',
    package_dir / 'io.py',
    package_dir / 'main_boomer.py',
    package_dir / 'main_seco.py',
    package_dir / 'model_characteristics.py',
    package_dir / 'parameters.py',
    package_dir / 'persistence.py',
    package_dir / 'runnables.py',
    package_dir / 'training.py'
]

install_data(source_files, install_dir : install_dir)
