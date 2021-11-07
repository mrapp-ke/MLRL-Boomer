project('common', 'cython', 'cpp')

# Source files
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
common_lib = cpp.find_library('mlrlcommon', dirs : meson.current_source_dir() / cpp_dir / 'build/mlrl/common/')
common_include = declare_dependency(include_directories : include_directories(cpp_dir / 'mlrl/common/include/'))

dependencies = [
    python_dep,
    numpy_dep,
    common_lib,
    common_include
]

# Cython compiler options
cython_options = ['cython_language=cpp']

foreach module_name : cython_module_names
    file_name = 'cython' / module_name + '.pyx'
    python.extension_module(module_name, file_name, dependencies : dependencies, override_options : cython_options)
endforeach
