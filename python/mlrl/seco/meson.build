project('seco', 'cython', 'cpp')

# Source files
cython_dir = 'cython'
cython_module_names = [
    'heuristics',
    'instance_sampling',
    'model',
    'output',
    'rule_evaluation_label_wise',
    'statistics_label_wise',
    'stopping'
]

# Dependencies
common_project = subproject('common')
python_dep = common_project.get_variable('python_dep')
numpy_dep = common_project.get_variable('numpy_dep')
common_lib = common_project.get_variable('common_lib')
common_include = common_project.get_variable('common_include')

cpp = meson.get_compiler('cpp')
cpp_dir = common_project.get_variable('cpp_dir')
lib_dir = meson.current_source_dir() / cpp_dir / 'build/mlrl/seco/'
seco_lib = cpp.find_library('mlrlseco', dirs : lib_dir)
seco_include = declare_dependency(include_directories : include_directories(cpp_dir / 'mlrl/seco/include/'))

dependencies = [
    python_dep,
    numpy_dep,
    common_lib,
    common_include,
    seco_lib,
    seco_include
]

# Cython compiler options
cython_options = common_project.get_variable('cython_options')

python = common_project.get_variable('python')
install_dir = meson.current_source_dir() / cython_dir

foreach module_name : cython_module_names
    file_name = cython_dir / module_name + '.pyx'
    python.extension_module(module_name, file_name, dependencies : dependencies, override_options : cython_options,
                            install : true, install_dir : install_dir, install_rpath : lib_dir)
endforeach
