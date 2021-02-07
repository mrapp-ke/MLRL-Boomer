project('boosting', 'cpp', version : '0.4.0')

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

boosting_lib = library('mlrlboosting', source_files, include_directories : include_directories,
                       dependencies : dependencies, version : '0.4.0')
