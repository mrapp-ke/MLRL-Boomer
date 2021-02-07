project('common', 'cpp', version : '0.4.0')

# Source files
source_files = [

]

# Dependencies
openmp_dep = dependency('openmp')

dependencies = [
    openmp_dep
]

# Directory containing public headers
include_directories = include_directories('include')

common_lib = library('mlrlcommon', source_files, include_directories : include_directories, dependencies : dependencies,
                     version : '0.4.0')
common_dep = declare_dependency(include_directories : include_directories, link_with : common_lib)
