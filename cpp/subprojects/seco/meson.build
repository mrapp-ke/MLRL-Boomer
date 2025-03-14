project('seco', 'cpp')

# Source files
source_files = [
    'src/mlrl/seco/data/matrix_coverage_dense.cpp',
    'src/mlrl/seco/data/vector_confusion_matrix_dense.cpp',
    'src/mlrl/seco/heuristics/heuristic_accuracy.cpp',
    'src/mlrl/seco/heuristics/heuristic_f_measure.cpp',
    'src/mlrl/seco/heuristics/heuristic_laplace.cpp',
    'src/mlrl/seco/heuristics/heuristic_m_estimate.cpp',
    'src/mlrl/seco/heuristics/heuristic_precision.cpp',
    'src/mlrl/seco/heuristics/heuristic_recall.cpp',
    'src/mlrl/seco/heuristics/heuristic_wra.cpp',
    'src/mlrl/seco/lift_functions/lift_function_kln.cpp',
    'src/mlrl/seco/lift_functions/lift_function_no.cpp',
    'src/mlrl/seco/lift_functions/lift_function_peak.cpp',
    'src/mlrl/seco/model/decision_list_builder.cpp',
    'src/mlrl/seco/prediction/predictor_binary_output_wise.cpp',
    'src/mlrl/seco/rule_evaluation/head_type_partial.cpp',
    'src/mlrl/seco/rule_evaluation/head_type_single.cpp',
    'src/mlrl/seco/rule_evaluation/rule_evaluation_decomposable_partial.cpp',
    'src/mlrl/seco/rule_evaluation/rule_evaluation_decomposable_single.cpp',
    'src/mlrl/seco/statistics/statistics_provider_decomposable_dense.cpp',
    'src/mlrl/seco/stopping/stopping_criterion_coverage.cpp',
    'src/mlrl/seco/info.cpp',
    'src/mlrl/seco/learner_seco_classifier.cpp'
]

# Test files
test_files = [
    'test/mlrl/seco/info.cpp'
]

# Dependencies
common_project = subproject('common')
common_dep = common_project.get_variable('common_dep')

dependencies = [
    common_dep
]

# Directory containing public headers
include_dir = include_directories('include')

# Directory into which the library should be installed
install_root = common_project.get_variable('install_root')
install_dir = install_root / meson.project_name() / 'mlrl' / meson.project_name() / 'cython/'

# Obtain information about the library
library_name = 'mlrl' + meson.project_name()
library_version = common_project.get_variable('library_version')
target_architecture = common_project.get_variable('target_architecture')

# Set configuration options
configuration = configuration_data()
configuration.set('mlrlseco_library_name', 'lib' + library_name)
configuration.set('mlrlseco_library_version', library_version)
configuration.set('mlrlseco_target_architecture', target_architecture)
configure_file(input : 'include/mlrl/seco/config.hpp.in', output : 'config.hpp', configuration: configuration)

# Library declaration
cpp_args = common_project.get_variable('cpp_args')
link_args = common_project.get_variable('link_args')

if host_machine.system() == 'windows'
    cpp_args += '-DMLRLSECO_EXPORTS'
endif

seco_lib = library(library_name, source_files, include_directories : include_dir, dependencies : dependencies,
                   cpp_args : cpp_args, link_args : link_args, version : library_version, install : true,
                   install_dir : install_dir)

# Test declaration
gtest_test_dep = common_project.get_variable('gtest_test_dep')

if gtest_test_dep.found()
    common_test_dep = common_project.get_variable('common_test_dep')

    test_dependencies = [
        gtest_test_dep,
        common_test_dep,
        declare_dependency(include_directories : [include_dir, include_directories('src')], link_with : seco_lib)
    ]

    test_name = library_name + 'tests'
    seco_tests = executable(test_name, test_files, dependencies : test_dependencies)
    test(test_name, seco_tests)
endif
