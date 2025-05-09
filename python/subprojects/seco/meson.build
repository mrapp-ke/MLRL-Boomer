project('seco', 'cython', 'cpp')

# Source files
cython_dir = 'cython'
cython_module_names = [
    'heuristic',
    'learner_seco',
    'learner',
    'lift_function',
    'package_info',
    'stopping_criterion'
]

# Dependencies
common_project = subproject('common')
python_dep = common_project.get_variable('python_dep')
numpy_dep = common_project.get_variable('numpy_dep')
common_include = common_project.get_variable('common_include')

cpp = meson.get_compiler('cpp')
cpp_dir = common_project.get_variable('cpp_dir')
seco_lib = cpp.find_library('mlrlseco', dirs : meson.current_source_dir() / cpp_dir / 'build/subprojects/seco/')
seco_include = declare_dependency(include_directories : include_directories(cpp_dir / 'subprojects/seco/include/'))

dependencies = [
    python_dep,
    numpy_dep,
    common_include,
    seco_lib,
    seco_include
]

# Cython compiler arguments
cython_options = common_project.get_variable('cython_options')
cython_args = common_project.get_variable('cython_args')

# C++ compiler arguments
cpp_args = common_project.get_variable('cpp_args')

# Directory into which the extension modules and Python source files should be installed
package_dir = 'mlrl' / meson.project_name()
install_dir = meson.current_source_dir() / package_dir / cython_dir

# Extension modules
python = common_project.get_variable('python')
rpath = common_project.get_variable('rpath')
add_project_arguments(cython_args, language : 'cython')

foreach module_name : cython_module_names
    pyx_file = package_dir / cython_dir / module_name + '.pyx'
    python.extension_module(module_name, pyx_file, dependencies : dependencies, override_options : cython_options,
                            cpp_args : cpp_args, install : true, install_dir : install_dir, install_rpath : rpath)
endforeach
