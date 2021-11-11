project('common', 'cython', 'cpp')

# Source files
cython_dir = 'cython'
cython_module_names = [
    'algorithm_builder',
    'feature_binning',
    'feature_sampling',
    'input',
    'instance_sampling',
    'label_sampling',
    'measures',
    'model',
    'output',
    'partition_sampling',
    'post_processing',
    'pruning',
    'rule_induction',
    'rule_model_assemblage',
    'statistics',
    'stopping',
    'thresholds',
    'thresholds_approximate',
    'thresholds_exact'
]

# Dependencies
python = import('python').find_installation()
python_dep = python.dependency()
numpy_cmd = run_command(python.full_path(), '-c', 'import numpy; print(numpy.get_include())', check : true )
numpy_dep = declare_dependency(include_directories : include_directories(numpy_cmd.stdout().strip()))

cpp = meson.get_compiler('cpp')
cpp_dir = '../../../cpp/'
package_dir = 'mlrl' / meson.project_name()
common_lib = cpp.find_library('mlrlcommon', dirs : meson.current_source_dir() / cpp_dir / 'build' / package_dir)
common_include = declare_dependency(include_directories : include_directories(cpp_dir / package_dir / 'include'))

dependencies = [
    python_dep,
    numpy_dep,
    common_lib,
    common_include
]

# Cython compiler options
cython_options = [
    'cython_language=cpp',
    'cython_version=3'
]

# Directory into which the extension modules and Python source files should be installed
python_version = python.language_version()
lib_dir = meson.current_source_dir() / '../../../venv/lib/python' + python_version + '/site-packages/'
install_dir = lib_dir / package_dir

# Extension modules
runtime_lib_dir = lib_dir / 'mlrl.libs'

foreach module_name : cython_module_names
    file_name = cython_dir / module_name + '.pyx'
    python.extension_module(module_name, file_name, dependencies : dependencies, override_options : cython_options,
                            install : true, install_dir : install_dir / cython_dir, install_rpath : runtime_lib_dir)
endforeach

# Python source files
source_files = [
    'arrays.py',
    'data_types.py',
    'learners.py',
    'options.py',
    'rule_learners.py',
    'strings.py'
]

install_data(source_files, install_dir : install_dir)
