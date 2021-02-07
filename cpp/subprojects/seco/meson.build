project('seco', 'cpp', version : '0.4.0')

# Source files
source_files = [

]

# Dependencies
common_dep = subproject('common').get_variable('common_dep')

dependencies = [
    common_dep
]

# Directory containing public headers
include_directories = include_directories('include')

seco_lib = library('mlrlseco', source_files, include_directories : include_directories, dependencies : dependencies,
                   version : '0.4.0')
