project('common', 'cpp')

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

# Directory into which the extension modules and Python source files should be installed
python_version = python.language_version()
lib_dir = meson.current_source_dir() / '../../../venv/lib/python' + python_version + '/site-packages/'
install_dir = lib_dir / package_dir

# Extension modules
runtime_lib_dir = lib_dir / 'mlrl.libs'
cython_include_dir = meson.current_source_dir()

foreach module_name : cython_module_names
    pyx_file = package_dir / cython_dir / module_name + '.pyx'
    pxd_file = package_dir / cython_dir / module_name + '.pxd'
    command = ['cython', '--cplus', '-3', '--fast-fail', '@INPUT@', '-o', '@OUTPUT@']
    cython_output_file = custom_target(module_name, output : module_name + '.cpp', input : pyx_file,
                                       depend_files: pxd_file, command : command)
    python.extension_module(module_name, cython_output_file, dependencies : dependencies, install : true,
                            install_dir : install_dir / cython_dir, install_rpath : runtime_lib_dir)
endforeach

# Python source files
source_files = [
    package_dir / 'arrays.py',
    package_dir / 'data_types.py',
    package_dir / 'learners.py',
    package_dir / 'options.py',
    package_dir / 'rule_learners.py',
    package_dir / 'strings.py'
]

install_data(source_files, install_dir : install_dir)
