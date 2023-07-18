project('seco', 'cpp')

# Source files
source_files = [
    'src/seco/data/matrix_coverage_dense.cpp',
    'src/seco/data/vector_confusion_matrix_dense.cpp',
    'src/seco/heuristics/heuristic_accuracy.cpp',
    'src/seco/heuristics/heuristic_f_measure.cpp',
    'src/seco/heuristics/heuristic_laplace.cpp',
    'src/seco/heuristics/heuristic_m_estimate.cpp',
    'src/seco/heuristics/heuristic_precision.cpp',
    'src/seco/heuristics/heuristic_recall.cpp',
    'src/seco/heuristics/heuristic_wra.cpp',
    'src/seco/lift_functions/lift_function_kln.cpp',
    'src/seco/lift_functions/lift_function_no.cpp',
    'src/seco/lift_functions/lift_function_peak.cpp',
    'src/seco/model/decision_list_builder.cpp',
    'src/seco/prediction/predictor_binary_label_wise.cpp',
    'src/seco/rule_evaluation/head_type_partial.cpp',
    'src/seco/rule_evaluation/head_type_single.cpp',
    'src/seco/rule_evaluation/rule_evaluation_label_wise_partial.cpp',
    'src/seco/rule_evaluation/rule_evaluation_label_wise_single.cpp',
    'src/seco/statistics/statistics_provider_label_wise_dense.cpp',
    'src/seco/stopping/stopping_criterion_coverage.cpp',
    'src/seco/info.cpp',
    'src/seco/learner.cpp',
    'src/seco/learner_seco.cpp'
]

# Dependencies
common_project = subproject('common')
common_dep = common_project.get_variable('common_dep')

dependencies = [
    common_dep
]

# Directory containing public headers
include_directories = include_directories('include')

# Directory into which the library should be installed
install_root = common_project.get_variable('install_root')
install_dir = install_root / meson.project_name() / 'mlrl' / meson.project_name() / 'cython/'

# Obtain information about the library
lib_name = 'mlrl' + meson.project_name()
version = common_project.get_variable('version')
target_architecture = common_project.get_variable('target_architecture')

# Set configuration options
configuration = configuration_data()
configuration.set('mlrlseco_library_name', 'lib' + lib_name)
configuration.set('mlrlseco_library_version', version)
configuration.set('mlrlseco_target_architecture', target_architecture)
configure_file(input : 'include/seco/config.hpp.in', output : 'config.hpp', configuration: configuration)

# Library declaration
cpp_args = common_project.get_variable('cpp_args')
link_args = common_project.get_variable('link_args')

if host_machine.system() == 'windows'
    cpp_args += '-DMLRLSECO_EXPORTS'
endif

seco_lib = library(lib_name, source_files, include_directories : include_directories, dependencies : dependencies,
                   cpp_args : cpp_args, link_args : link_args, version : version, install : true,
                   install_dir : install_dir)
