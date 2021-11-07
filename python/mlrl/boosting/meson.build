project('boosting', 'cython', 'cpp')

# Source files
cython_module_names = [
    'label_binning',
    'losses_example_wise',
    'losses_label_wise',
    'model',
    'output',
    'post_processing',
    'rule_evaluation_example_wise',
    'rule_evaluation_label_wise',
    'statistics_example_wise',
    'statistics_label_wise'
]

# Dependencies
common_project = subproject('common')
python_dep = common_project.get_variable('python_dep')
numpy_dep = common_project.get_variable('numpy_dep')
common_lib = common_project.get_variable('common_lib')
common_include = common_project.get_variable('common_include')

cpp = meson.get_compiler('cpp')
cpp_dir = common_project.get_variable('cpp_dir')
boosting_lib = cpp.find_library('mlrlboosting', dirs : meson.current_source_dir() / cpp_dir / 'build/mlrl/boosting/')
boosting_include = declare_dependency(include_directories : include_directories(cpp_dir / 'mlrl/boosting/include/'))

dependencies = [
    python_dep,
    numpy_dep,
    common_lib,
    common_include,
    boosting_lib,
    boosting_include
]

# Cython compiler options
cython_options = common_project.get_variable('cython_options')

python = common_project.get_variable('python')

foreach module_name : cython_module_names
    file_name = 'cython' / module_name + '.pyx'
    python.extension_module(module_name, file_name, dependencies : dependencies, override_options : cython_options)
endforeach
